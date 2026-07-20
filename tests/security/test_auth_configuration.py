from __future__ import annotations
import base64, hashlib, hmac, json, os, subprocess, time
from pathlib import Path

import pytest

from vulcan.runtime.auth import AuthConfig, AuthError, authenticate_bearer
from vulcan.runtime.settings import SettingsError, load_runtime_settings

SECRET="AbCdEfGhIjKlMnOpQrStUvWxYz7890+/rotation"
OLD="OldSecretAbCdEfGhIjKlMnOpQrStUvWxYz7890+/"
NOW=1_700_000_000.0

def b64(o):
    raw=json.dumps(o,separators=(",",":"),sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
def token(secret=SECRET, *, kid="v1", **claims):
    h={"alg":"HS256","typ":"JWT","kid":kid}
    p={"iss":"vulcan","aud":"vulcan-runtime","sub":"alice","tenant":"t1","scope":"reason:write","iat":NOW,"exp":NOW+300,"jti":"jti_0123456789abcdef"}
    p.update(claims)
    msg=f"{b64(h)}.{b64(p)}"
    sig=base64.urlsafe_b64encode(hmac.new(secret.encode(),msg.encode(),hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"{msg}.{sig}"
def env(tmp_path: Path, **overrides: str):
    e={"VULCAN_ENV":"development","VULCAN_JWT_SECRET":SECRET,"VULCAN_RUNTIME_DURABLE_ROOT":str(tmp_path/"rt"),"VULCAN_MEMORY_SQLITE_PATH":str(tmp_path/"rt"/"memory.sqlite")}
    e.update(overrides); return e

@pytest.mark.parametrize("name", ["VULCAN_JWT_SECRET","GRAPHIX_JWT_SECRET","JWT_SECRET_KEY","JWT_SECRET"])
def test_runtime_settings_accept_exactly_one_supported_name(tmp_path, name):
    e=env(tmp_path); e.pop("VULCAN_JWT_SECRET"); e[name]=SECRET
    assert load_runtime_settings(e).jwt_secret.reveal()==SECRET

def test_runtime_settings_accepts_matching_alias_overlap(tmp_path):
    assert load_runtime_settings(env(tmp_path, JWT_SECRET=SECRET, JWT_SECRET_KEY=SECRET)).jwt_secret.reveal()==SECRET

def test_runtime_settings_rejects_conflicts_and_weak_values(tmp_path):
    with pytest.raises(SettingsError): load_runtime_settings(env(tmp_path, JWT_SECRET="DifferentSecretAbCdEfGhIjKlMnOpQrStUvWxYz012345"))
    with pytest.raises(SettingsError): load_runtime_settings(env(tmp_path, VULCAN_JWT_SECRET="short"))
    with pytest.raises(SettingsError): load_runtime_settings(env(tmp_path, VULCAN_JWT_SECRET="dev-jwt-secret-key-change-in-production-minimum-32-chars"))

@pytest.mark.parametrize("name", ["VULCAN_JWT_SECRET","GRAPHIX_JWT_SECRET","JWT_SECRET_KEY","JWT_SECRET"])
def test_entrypoint_accepts_each_supported_alias(name):
    e={"PATH":os.environ["PATH"], name:SECRET}
    cp=subprocess.run(["sh","entrypoint.sh","sh","-c","test \"$VULCAN_JWT_SECRET\" = \"$EXPECTED\""], env=e|{"EXPECTED":SECRET}, text=True, capture_output=True, timeout=5)
    assert cp.returncode==0, cp.stderr

def test_entrypoint_rejects_conflicts_and_weak_values():
    base={"PATH":os.environ["PATH"],"VULCAN_JWT_SECRET":SECRET,"JWT_SECRET":OLD}
    cp=subprocess.run(["sh","entrypoint.sh","true"], env=base, text=True, capture_output=True, timeout=5)
    assert cp.returncode==78
    assert SECRET not in cp.stderr and OLD not in cp.stderr
    cp=subprocess.run(["sh","entrypoint.sh","true"], env={"PATH":os.environ["PATH"],"VULCAN_JWT_SECRET":"short"}, text=True, capture_output=True, timeout=5)
    assert cp.returncode==78

def test_jwt_requires_jti_kid_issuer_audience_and_max_lifetime():
    cfg=AuthConfig(SECRET,"vulcan","vulcan-runtime",key_version="v1")
    p=authenticate_bearer("Bearer "+token(), cfg, clock=lambda: NOW)
    assert p.jti=="jti_0123456789abcdef" and p.key_version=="v1"
    for bad in [token(jti="not canonical spaces"), token(iss="evil"), token(aud="evil"), token(exp=NOW+7200)]:
        with pytest.raises(AuthError): authenticate_bearer("Bearer "+bad, cfg, clock=lambda: NOW)

def test_rotation_overlap_accepts_old_and_new_key_versions_with_bounded_timing():
    cfg=AuthConfig(SECRET,"vulcan","vulcan-runtime",key_version="v2",keyring={"v1":OLD,"v2":SECRET})
    assert authenticate_bearer("Bearer "+token(OLD,kid="v1"), cfg, clock=lambda: NOW).key_version=="v1"
    assert authenticate_bearer("Bearer "+token(SECRET,kid="v2"), cfg, clock=lambda: NOW).key_version=="v2"
    start=time.perf_counter()
    for _ in range(100):
        with pytest.raises(AuthError): authenticate_bearer("Bearer "+token(OLD,kid="missing"), cfg, clock=lambda: NOW)
    assert time.perf_counter()-start < 1.0
