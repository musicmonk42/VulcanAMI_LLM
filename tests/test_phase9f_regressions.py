from __future__ import annotations
import ast, json
from pathlib import Path
import pytest
from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement, CSIUValidationError
from tests.test_phase9b_csiu_persistence import Clock, cfg, apply

PHASE9_FILES = [p for p in Path('tests').glob('test_phase9*.py')] + [Path('src/vulcan/tests/test_self_improvement_drive.py')]

def test_phase9_assertions_are_not_tautological():
    bad=[]
    for path in PHASE9_FILES:
        tree=ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            text=ast.get_source_segment(path.read_text(), node.test) or ''
            compact=' '.join(text.split())
            if 'len(' in compact and '>= 0' in compact: bad.append((path,node.lineno,compact))
            if 'applied + blocked' in compact: bad.append((path,node.lineno,compact))
            if ' or ' in compact and '!=' in compact and '==' in compact: bad.append((path,node.lineno,compact))
    assert bad == []

@pytest.mark.parametrize('field,value', [
    ('current_snapshot_digest','1'*64),
    ('previous_snapshot_digest','2'*64),
    ('decision_digest','3'*64),
    ('utility',0.123456),
    ('ewma_utility',0.234567),
    ('charged_influence',0.987),
    ('reason_code','tampered_reason'),
])
def test_version2_record_tamper_is_hash_bound(tmp_path, field, value):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c)); apply(e); e.close()
    lines=store.read_text().splitlines()
    rec=json.loads(lines[-1]); rec[field]=value
    lines[-1]=json.dumps(rec, sort_keys=True, separators=(',',':'))
    store.write_text('\n'.join(lines)+'\n')
    with pytest.raises(CSIUValidationError):
        CSIUEnforcement(cfg(store,c), policy=e.policy)

def test_version2_record_rejects_unknown_and_requires_exact_fields(tmp_path):
    c=Clock(); store=tmp_path/'c.jsonl'; e=CSIUEnforcement(cfg(store,c)); apply(e); e.close()
    lines=store.read_text().splitlines(); rec=json.loads(lines[-1]); rec['chain_digest']='0'*64
    lines[-1]=json.dumps(rec, sort_keys=True, separators=(',',':')); store.write_text('\n'.join(lines)+'\n')
    with pytest.raises(CSIUValidationError): CSIUEnforcement(cfg(store,c), policy=e.policy)
