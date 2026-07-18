from __future__ import annotations
import json, threading, sys, subprocess, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement, CSIUEnforcementConfig, CSIUValidationError

class Clock:
    def __init__(self): self.t=datetime(2026,1,1,tzinfo=timezone.utc)
    def __call__(self): return self.t
    def advance(self, seconds): self.t += timedelta(seconds=seconds)

def cfg(path, clock, hist=True):
    Path(path).touch()
    return CSIUEnforcementConfig(durable_store_path=str(path), durable_accounting_required=True, clock=clock, history_tracking_enabled=hist)

def plan(): return {"objective_weights":{"a":1.0},"id":"p"}

def apply(enf):
    return enf.apply_regularization_with_enforcement(plan(), 0.05, {}, plan_id="p")

def test_dependency_light_direct_imports_from_tmp():
    code='import sys; import vulcan.world_model.meta_reasoning.csiu_enforcement; import vulcan.world_model.meta_reasoning.self_improvement_drive; print(sorted({"numpy","torch","fastapi","aiohttp","networkx"}&set(sys.modules)))'
    out=subprocess.check_output([sys.executable,'-c',code], cwd='/tmp', env={**os.environ,'PYTHONPATH':str(Path.cwd()/'src')}, text=True)
    assert out.strip().endswith('[]')

def test_restart_spanning_budget_and_history_display_off(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'
    e=CSIUEnforcement(cfg(store,c)); apply(e); apply(e)
    assert e.check_cumulative_influence()['cumulative_influence'] == pytest.approx(0.10)
    pol=e.policy; e.close(); e2=CSIUEnforcement(cfg(store,c, hist=False), policy=pol)
    unchanged, dec=apply(e2)
    assert dec.blocked and dec.reason_code=='cumulative_cap_exceeded' and unchanged==plan()
    c.advance(3601); assert e2.check_cumulative_influence()['cumulative_influence']==0
    apply(e2); assert e2.check_cumulative_influence()['cumulative_influence']==pytest.approx(0.05)

def test_corruption_symlink_and_second_writer_rejected(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c)); apply(e)
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(store,c), policy=e.policy)
    e.close(); raw=store.read_text(); store.write_text(raw.replace('committed','aborted',1))
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(store,c), policy=e.policy)
    bad=tmp_path/'bad.jsonl'; target=tmp_path/'target'; target.touch(); bad.symlink_to(target)
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(bad,c))

def test_concurrent_remaining_budget_race(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c)); apply(e)
    barrier=threading.Barrier(2); results=[]
    def worker():
        barrier.wait(); results.append(apply(e))
    ts=[threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    applied=sum(1 for _,d in results if d.applied); blocked=sum(1 for _,d in results if d.blocked)
    assert (applied,blocked)==(1,1)
    assert e.check_cumulative_influence()['cumulative_influence']==pytest.approx(0.10)

def test_actual_effect_over_pressure_rejected(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c))
    # Pressure above single cap is capped and actual influence is defined as the larger
    # conservative reserved/validated effect that is accounted durably.
    _, d=e.apply_regularization_with_enforcement(plan(), 9.0, {})
    assert d.actual_effect <= e.config.max_single_influence
    assert e.check_cumulative_influence()['cumulative_influence']==pytest.approx(0.05)
