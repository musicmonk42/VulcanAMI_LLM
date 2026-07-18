"""Offline gates for local language-adapter artifact manifests."""
import hashlib, json, unicodedata
import pytest
from vulcan.local_language import ReleaseRole, ReleaseVerificationError, verify_release, decode_generated_suffix, load_tokenizer_contract
from vulcan.local_language.release import REQ_CATS

def _sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def _j(o): return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",",":"))
def _tok():
    vocab=["<bos>","<eos>","<pad>","<unk>"]+sorted(set('{}[],:\"-0123456789').union({"schema_version","transformer-span-proposal/1","candidates","operation","span","argument_spans","confidence","start","end","arithmetic","lookup","memory_read","memory_write","memory_forget","unsupported","expression","key","request_span"}))
    return {"schema_version":"local-tokenizer/1","normalization":"NFC","vocabulary":vocab,"special_tokens":{"bos":0,"eos":1,"pad":2,"unk":3},"max_length":1000}
def _cfg(**kw):
    d={"schema_version":"local-gpt-span-provider/1","model_architecture":"tiny-gpt-span","vocabulary_size":64,"embedding_width":32,"layer_count":2,"attention_head_count":4,"feed_forward_width":64,"approved_context_length":512,"max_generated_proposal_tokens":128,"special_token_ids":{"bos":0,"eos":1,"pad":2,"unk":3},"dropout":0.0,"generation_method":"greedy"}; d.update(kw); return d
def _eval(release_id="adapter-01", passed=True, bad=False):
    metrics={c:1.0 for c in REQ_CATS}; thresholds={c:0.9 for c in REQ_CATS}
    if bad: metrics["fallback_behavior"]=0.0
    return {"schema_version":"local-language-evaluation/1","release_id":release_id,"runtime_abi":"vulcan-transformer-span/1","dataset_digest":"0"*64,"categories":sorted(REQ_CATS),"metrics":metrics,"thresholds":thresholds,"passed":passed,"evaluated_at":"2026-07-18T00:00:00Z","evaluator":"eval-tool"}
def _release(root, *, cfg=None, tok=None, ev=None, role="input-language-adapter", abi="vulcan-transformer-span/1"):
    files={"weights.safetensors":b"SAFE","tokenizer.json":_j(tok or _tok()).encode(),"config.json":_j(cfg or _cfg()).encode(),"evaluation.json":_j(ev or _eval()).encode()}
    for n,b in files.items(): (root/n).write_bytes(b)
    arts=[]
    for name,path in [("weights","weights.safetensors"),("tokenizer","tokenizer.json"),("config","config.json"),("evaluation_report","evaluation.json")]: arts.append({"name":name,"path":path,"sha256":_sha(files[path]),"byte_size":len(files[path])})
    manifest={"schema_version":"local-language-release/1","release_id":"adapter-01","version":"1.0.0","role":role,"runtime_abi":abi,"provider_implementation":"local-gpt-span-provider","provider_config_artifact":"config.json","tokenizer_artifact":"tokenizer.json","weights_artifact":"weights.safetensors","evaluation_report_artifact":"evaluation.json","release_created":"2026-07-18T00:00:00Z","canonical_manifest_digest":"0"*64,"approval":{"state":"approved","approval_id":"review-01","approved_by":"release-authority","approved_at":"2026-07-18T00:00:00Z","manifest_digest":"0"*64},"evaluation":{"passed":True,"report_sha256":arts[-1]["sha256"]},"artifacts":arts}
    b=_j(manifest).encode(); digest=_sha(b); manifest["canonical_manifest_digest"]=digest; manifest["approval"]["manifest_digest"]=digest; (root/"manifest.json").write_text(_j(manifest),encoding="utf-8")

def test_verify_release_binds_complete_approved_role_specific_artifacts(tmp_path):
    root=tmp_path/"release"; root.mkdir(); _release(root); release=verify_release(root)
    assert release.role is ReleaseRole.INPUT and release.runtime_abi == "vulcan-transformer-span/1"

def test_duplicate_manifest_keys_are_rejected(tmp_path):
    root=tmp_path/"release"; root.mkdir(); _release(root); (root/"manifest.json").write_text('{"schema_version":"local-language-release/1","schema_version":"local-language-release/1"}')
    with pytest.raises(ReleaseVerificationError, match="duplicate JSON key"): verify_release(root)

@pytest.mark.parametrize("mutate", ["extra","missing","digest","size","absolute","traversal","symlink","role","abi","unpassed_eval","eval_inconsistent","dims","dropout","sampling","tok_dup","tok_control","tok_missing","special"])
def test_strict_release_rejections(tmp_path, mutate):
    cfg=_cfg(); tok=_tok(); ev=_eval(); role="input-language-adapter"; abi="vulcan-transformer-span/1"
    if mutate=="dims": cfg["embedding_width"]=30
    if mutate=="dropout": cfg["dropout"]=0.1
    if mutate=="sampling": cfg["generation_method"]="sampling"
    if mutate=="tok_dup": tok["vocabulary"][4]=tok["vocabulary"][5]
    if mutate=="tok_control": tok["vocabulary"][4]="bad\x00"
    if mutate=="tok_missing": tok["vocabulary"].remove("domain_hint") if "domain_hint" in tok["vocabulary"] else tok["vocabulary"].remove("arithmetic")
    if mutate=="special": tok["special_tokens"]["unk"]=tok["special_tokens"]["pad"]
    if mutate=="unpassed_eval": ev=_eval(passed=False)
    if mutate=="eval_inconsistent": ev=_eval(bad=True)
    if mutate=="role": role="output-language-adapter"
    if mutate=="abi": abi="semantic-ingress/2"
    root=tmp_path/"release"; root.mkdir(); _release(root,cfg=cfg,tok=tok,ev=ev,role=role,abi=abi)
    if mutate=="extra": (root/"extra.bin").write_bytes(b"x")
    if mutate=="missing": (root/"weights.safetensors").unlink()
    if mutate=="digest": (root/"weights.safetensors").write_bytes(b"BAD")
    if mutate=="size":
        m=json.loads((root/"manifest.json").read_text()); m["artifacts"][0]["byte_size"]+=1; (root/"manifest.json").write_text(_j(m))
    if mutate in {"absolute","traversal"}:
        m=json.loads((root/"manifest.json").read_text()); m["artifacts"][0]["path"]="/tmp/x" if mutate=="absolute" else "../x"; (root/"manifest.json").write_text(_j(m))
    if mutate=="symlink": (root/"weights.safetensors").unlink(); (root/"weights.safetensors").symlink_to(root/"config.json")
    with pytest.raises(ReleaseVerificationError): verify_release(root)

def test_exact_decoder_rejects_special_misuse(tmp_path):
    root=tmp_path/"release"; root.mkdir(); _release(root); verify_release(root); tok=load_tokenizer_contract(root/"tokenizer.json")
    with pytest.raises(ReleaseVerificationError): decode_generated_suffix([tok.special_tokens["unk"], tok.special_tokens["eos"]], tok, max_tokens=4)
    with pytest.raises(ReleaseVerificationError): decode_generated_suffix([tok.special_tokens["eos"], 4], tok, max_tokens=4)
