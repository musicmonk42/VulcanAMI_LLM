from __future__ import annotations
import json, threading, sys, subprocess, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement, CSIUEnforcementConfig, CSIUValidationError, CSIUMetricSnapshot, METRIC_ORDER

class Clock:
    def __init__(self): self.t=datetime(2026,1,1,tzinfo=timezone.utc)
    def __call__(self): return self.t
    def advance(self, seconds): self.t += timedelta(seconds=seconds)

def cfg(path, clock, hist=True, cap=0.10, single=0.05):
    Path(path).touch()
    return CSIUEnforcementConfig(durable_store_path=str(path), durable_accounting_required=True, clock=clock, history_tracking_enabled=hist, max_cumulative_influence_window=cap, max_single_influence=single)

def plan(): return {"objective_weights":{"a":1.0},"id":"p"}

def snap(enf, metrics=None):
    now=enf._clock(); n=len(enf._seen_snapshot_digests)
    return CSIUMetricSnapshot(metrics=metrics or {k:.5 for k in METRIC_ORDER}, window_start=(now-timedelta(minutes=5)).isoformat().replace("+00:00","Z"), window_end=now.isoformat().replace("+00:00","Z"), sample_count=30, aggregation_method="mean", metric_definition_version=enf.policy.metric_definition_version, provider_id="p", provenance_digest=(str(n%10))*64, policy_digest=enf.policy.policy_digest)
def apply(enf):
    prev=getattr(enf,"_last_snapshot",None)
    if prev is None:
        prev=snap(enf); enf.apply_regularization_from_snapshots(plan(), None, prev); enf.config.clock.advance(360) if hasattr(enf.config.clock, "advance") else None
    cur=snap(enf, {k:.6 if k in ("A","C","E","U") else .4 for k in METRIC_ORDER})
    return enf.apply_regularization_from_snapshots(plan(), prev, cur, plan_id="p")

def test_dependency_light_direct_imports_from_tmp():
    code='import sys; import vulcan.world_model.meta_reasoning.csiu_enforcement; import vulcan.world_model.meta_reasoning.self_improvement_drive; print(sorted({"numpy","torch","fastapi","aiohttp","networkx"}&set(sys.modules)))'
    out=subprocess.check_output([sys.executable,'-c',code], cwd='/tmp', env={**os.environ,'PYTHONPATH':str(Path.cwd()/'src')}, text=True)
    assert out.strip().endswith('[]')

def test_restart_spanning_budget_and_history_display_off(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'
    e=CSIUEnforcement(cfg(store,c)); apply(e); apply(e)
    assert e.check_cumulative_influence()['cumulative_influence'] > 0
    pol=e.policy; e.close(); e2=CSIUEnforcement(cfg(store,c, hist=False), policy=pol)
    unchanged, dec=apply(e2)
    assert isinstance(unchanged, dict)
    c.advance(3601); assert e2.check_cumulative_influence()['cumulative_influence']==0
    apply(e2); assert e2.check_cumulative_influence()['cumulative_influence'] == pytest.approx(0.05, abs=0.05)

def test_observe_telemetry_is_evaluation_only_and_persistent(tmp_path):
    c=Clock(); store=tmp_path/'obs.jsonl'; e=CSIUEnforcement(cfg(store,c))
    s1=snap(e); d1=e.observe_telemetry_snapshots(None,s1); assert d1.reason_code=="baseline_established"
    assert e.check_cumulative_influence()["cumulative_influence"]==0 and e.get_statistics()["total_applications"]==0
    c.advance(360); s2=snap(e,{k:.6 if k in ("A","C","E","U") else .4 for k in METRIC_ORDER})
    d2=e.observe_telemetry_snapshots(s1,s2); assert d2.reason_code in {"telemetry_observed","zero_pressure"}
    assert e.check_cumulative_influence()["cumulative_influence"]==0 and e.get_statistics()["total_applications"]==0
    pol=e.policy; e.close(); e2=CSIUEnforcement(cfg(store,c), policy=pol)
    assert e2.check_cumulative_influence()["cumulative_influence"]==0
    cur=s2
    for i in range(10):
        c.advance(360); nxt=snap(e2,{k:.6+(i+1)*.001 if k in ("A","C","E","U") else .4-(i+1)*.001 for k in METRIC_ORDER})
        e2.observe_telemetry_snapshots(cur,nxt); cur=nxt
    assert e2.check_cumulative_influence()["cumulative_influence"]==0 and e2.get_statistics()["total_applications"]==0
    assert e2.observe_telemetry_snapshots(cur,cur).reason_code=="replayed_snapshot"
    c.advance(360); real=snap(e2,{k:.8 if k in ("A","C","E","U") else .3 for k in METRIC_ORDER})
    _,dec=e2.apply_regularization_from_snapshots(plan(),cur,real,plan_id="real")
    assert dec.applied and e2.check_cumulative_influence()["cumulative_influence"]>0

def test_corruption_symlink_and_second_writer_rejected(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c)); apply(e)
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(store,c), policy=e.policy)
    e.close(); raw=store.read_text(); store.write_text(raw.replace('committed','aborted',1))
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(store,c), policy=e.policy)
    bad=tmp_path/'bad.jsonl'; target=tmp_path/'target'; target.touch(); bad.symlink_to(target)
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(bad,c))

def test_concurrent_remaining_budget_race(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'
    e=CSIUEnforcement(cfg(store,c,cap=0.05,single=0.05))
    prev=snap(e,{k:(0.0 if k in ("A","C","E","U") else 1.0) for k in METRIC_ORDER})
    e.observe_telemetry_snapshots(None,prev); c.advance(360)
    cur=prev
    for i in range(20):
        c.advance(360)
        high = (i % 2 == 0)
        nxt=snap(e,{k:((1.0 if high else 0.0) if k in ("A","C","E","U") else (0.0 if high else 1.0)) for k in METRIC_ORDER})
        _,d0=e.apply_regularization_from_snapshots(plan(),cur,nxt,plan_id=f"p{i}")
        if not d0.applied: break
        cur=nxt
        rem=e.check_cumulative_influence()["remaining"]
        if 0.012 < rem < 0.014: break
    first=e.check_cumulative_influence()['cumulative_influence']
    c.advance(360)
    base=e._last_snapshot
    barrier=threading.Barrier(2); results=[]
    def worker():
        barrier.wait()
        now=e._clock(); tag=("1" if threading.current_thread().name.endswith("0") else "2")
        to_high = base.metrics["A"] < 0.5
        nxt=CSIUMetricSnapshot(metrics={k:((1.0 if to_high else 0.0) if k in ("A","C","E","U") else (0.0 if to_high else 1.0)) for k in METRIC_ORDER}, window_start=(now-timedelta(minutes=5)).isoformat().replace("+00:00","Z"), window_end=now.isoformat().replace("+00:00","Z"), sample_count=30, aggregation_method="mean", metric_definition_version=e.policy.metric_definition_version, provider_id="p", provenance_digest=tag*64, policy_digest=e.policy.policy_digest)
        results.append(e.apply_regularization_from_snapshots(plan(),base,nxt,plan_id="p"))
    ts=[threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    applied=sum(1 for _,d in results if d.applied); blocked=sum(1 for _,d in results if d.blocked)
    assert applied == 1
    assert blocked == 1
    assert [d.reason_code for _,d in results if d.blocked] in (["previous_snapshot_digest_mismatch"], ["replayed_snapshot"])
    expected = first + abs(results[0][1].pressure or results[1][1].pressure)
    assert e.check_cumulative_influence()['cumulative_influence'] == pytest.approx(expected, abs=1e-12)
    pol=e.policy; e.close(); e2=CSIUEnforcement(cfg(store,c,cap=0.05,single=0.05), policy=pol)
    assert e2.check_cumulative_influence()['cumulative_influence'] == pytest.approx(expected, abs=1e-12)

def test_actual_effect_over_pressure_rejected(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c))
    # Pressure above single cap is capped and actual influence is defined as the larger
    # conservative reserved/validated effect that is accounted durably.
    with pytest.raises(CSIUValidationError):
        e.apply_regularization_with_enforcement(plan(), 9.0, {}, snapshot=snap(e))
    _, d=apply(e)
    assert d.actual_effect <= e.config.max_single_influence
    assert e.check_cumulative_influence()['cumulative_influence']>0
