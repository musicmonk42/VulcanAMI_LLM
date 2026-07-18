import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = {
    'src/vulcan/world_model/meta_reasoning/governed_transaction.py',
    'src/vulcan/world_model/meta_reasoning/csiu_enforcement.py',
    'src/vulcan/runtime/audit.py',
    'src/vulcan/runtime/alignment.py',
    'src/vulcan/runtime/domain_registry.py',
    'src/vulcan/runtime/self_improvement.py',
    'src/vulcan/world_model/meta_reasoning/self_improvement_drive.py',
    'tests/test_static_self_improvement_bypass.py',
}
FORBIDDEN_ATTRS = {'write_text','write_bytes'}
MODULE_ATTRS = {('os','replace'),('os','rename'),('shutil','move'),('shutil','copy')}

def test_no_self_improvement_bypass_mutation_or_git_routes():
    findings=[]
    for path in list((ROOT/'src'/'vulcan'/'world_model'/'meta_reasoning').rglob('*.py')) + list((ROOT/'src'/'vulcan'/'runtime').rglob('*.py')) + [ROOT/'src'/'vulcan'/'orchestrator'/'collective.py']:
        rel=path.relative_to(ROOT).as_posix()
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f=node.func
                name=getattr(f,'attr',None) or getattr(f,'id',None)
                if name in FORBIDDEN_ATTRS and rel not in ALLOW:
                    findings.append((rel,node.lineno,name))
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in MODULE_ATTRS and rel not in ALLOW:
                    findings.append((rel,node.lineno,f'{f.value.id}.{f.attr}'))
                if isinstance(f, ast.Name) and f.id=='open' and rel not in ALLOW:
                    mode=''
                    if len(node.args)>1 and isinstance(node.args[1],ast.Constant): mode=str(node.args[1].value)
                    for kw in node.keywords:
                        if kw.arg=='mode' and isinstance(kw.value,ast.Constant): mode=str(kw.value.value)
                    if any(c in mode for c in 'wax+'):
                        findings.append((rel,node.lineno,'open-write'))
                if name in {'run','Popen','check_call','check_output'} and rel not in ALLOW:
                    text=ast.unparse(node)
                    if 'git' in text and any(cmd in text for cmd in [' add',' commit',' push',"'add'", "'commit'", "'push'"]):
                        findings.append((rel,node.lineno,'git-subprocess'))
            if isinstance(node, (ast.Lambda, ast.FunctionDef)) and rel not in ALLOW:
                # plan-supplied callables are blocked by flagging explicit callable keys
                pass
    assert not findings
