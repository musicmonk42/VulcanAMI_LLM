import base64
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
# Crypto for signature verification
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS
from flask_jwt_extended import (JWTManager, create_access_token, get_jwt,
                                get_jwt_identity, jwt_required)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import select

# ============================================================
# Environment & Configuration
# ============================================================
load_dotenv()

APP_VERSION = "1.3.0"  # Version bumped to reflect additional security & correctness hardening
BOOTSTRAP_KEY = os.environ.get("BOOTSTRAP_KEY")  # Optional bootstrap protection
MAX_CONTENT_LENGTH_BYTES = int(os.environ.get("MAX_CONTENT_LENGTH_BYTES", 16 * 1024 * 1024))  # 16MB default
JWT_EXP_MINUTES = int(os.environ.get("JWT_EXP_MINUTES", 30))
# Require TLS for bootstrap endpoint unless explicitly disabled (do not disable in production)
ENFORCE_HTTPS_BOOTSTRAP = os.environ.get("ENFORCE_HTTPS_BOOTSTRAP", "true").lower() != "false"
JWT_ISSUER = os.environ.get("JWT_ISSUER", "graphix-registry")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "graphix-agents")
JWT_IDENTITY_CLAIM = os.environ.get("JWT_IDENTITY_CLAIM", "sub")

# Revocation settings
JWT_REVOCATION_PREFIX = os.environ.get("JWT_REVOCATION_PREFIX", "jwt:revoked:")
JWT_ENABLE_LOGOUT = os.environ.get("JWT_ENABLE_LOGOUT", "true").lower() == "true"

# IR size byte cap (combined graph JSON serialized size)
IR_MAX_BYTES = int(os.environ.get("IR_MAX_BYTES", 2 * 1024 * 1024))  # 2 MiB default

# Instantiate Flask first so config can be bound
app = Flask(__name__)

# Enforce hard request body size limit
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_BYTES
app.config['JWT_IDENTITY_CLAIM'] = JWT_IDENTITY_CLAIM

# Security / JWT (fail fast if unset/weak)
jwt_secret = os.environ.get("JWT_SECRET_KEY")
if not jwt_secret or jwt_secret.strip() in {"super-secret-key", "insecure-dev-secret", "default-super-secret-key-change-me"}:
    raise RuntimeError("Refusing to start: JWT_SECRET_KEY must be set to a strong, non-default value in the environment.")
app.config['JWT_SECRET_KEY'] = jwt_secret
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=JWT_EXP_MINUTES)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DB_URI", "sqlite:///graphix_registry.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Instantiate core extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Rate limiter with Redis fallback
redis_storage_uri = None
redis_client = None
try:
    import redis
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_LIMITER_DB', 1))
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2
    )
    redis_client.ping()
    redis_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
    print(f"✅ Rate limiter connected to Redis: {redis_storage_uri}")
except Exception as e:
    print(f"⚠️  Redis not available for rate limiting: {e}")
    print("⚠️  Falling back to in-memory storage (NOT recommended for production)")
    redis_storage_uri = "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=redis_storage_uri,
    storage_options=(
        {"socket_connect_timeout": 2, "socket_timeout": 2}
        if redis_storage_uri != "memory://" else {}
    )
)

if redis_storage_uri != "memory://":
    print("✅ Rate limiter using Redis backend (production-ready)")
else:
    print("⚠️ Rate limiter using in-memory backend (development only)")

# CORS configuration (explicit allowlist; no wildcard)
cors_origins_env = os.environ.get("CORS_ORIGINS", "")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
if cors_origins:
    CORS(
        app,
        resources={r"/*": {"origins": cors_origins}},
        supports_credentials=False,
        max_age=3600,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Bootstrap-Key"]
    )
else:
    # No CORS if allowlist not configured
    pass

# Logging setup (structured audit logging secondary)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GraphixRegistry")
audit_logger = logging.getLogger("GraphixRegistryAudit")

# ============================================================
# Database Models
# ============================================================
class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(50), unique=True, nullable=False)
    pubkey = db.Column(db.Text, nullable=False, index=True)
    roles = db.Column(db.JSON, nullable=False)  # list of roles
    trust = db.Column(db.Float, default=0.5, nullable=False)

class IRProposal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(50), nullable=False)
    ir_json = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(50))
    event = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    # Could add metadata/json field for structured expansion later

with app.app_context():
    db.create_all()
    # Ensure index uniqueness for pubkey not enforced at DB layer; logical uniqueness enforced in code.

# ============================================================
# In-memory Security Stores (fallbacks when Redis unavailable)
# ============================================================
_NONCE_STORE: Dict[str, Tuple[str, float]] = {}  # agent_id -> (nonce, expires_at_epoch)
_BACKOFF_STORE: Dict[str, float] = {}            # ip-derived keys or counters
_BOOTSTRAP_KEY_USED_FLAG: bool = False           # in-memory fallback flag for bootstrap key used
_REVOKED_TOKENS: set = set()                     # fallback JWT jti blacklist

NONCE_TTL_SECONDS = int(os.environ.get("NONCE_TTL_SECONDS", 300))  # 5 minutes
MAX_ROLES = int(os.environ.get("MAX_ROLES", 10))

# ============================================================
# Helpers
# ============================================================
def safe_json():
    """Return parsed JSON or abort with 400 if missing/invalid."""
    if not request.is_json:
        abort(400, description="Content-Type must be application/json")
    data = request.get_json(silent=True)
    if data is None:
        abort(400, description="Invalid or empty JSON body")
    return data

AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{3,50}$")

def validate_agent_id(agent_id: Optional[str]) -> str:
    if not isinstance(agent_id, str):
        abort(400, description="agent_id must be a string")
    agent_id = agent_id.strip()
    if not AGENT_ID_PATTERN.match(agent_id):
        abort(400, description="agent_id must be 3-50 chars; allowed [A-Za-z0-9_.:-]")
    return agent_id

def validate_roles(roles: List[str]) -> List[str]:
    if not isinstance(roles, list):
        abort(400, description="roles must be a list")
    if len(roles) > MAX_ROLES:
        abort(400, description=f"Too many roles (max {MAX_ROLES})")
    cleaned = []
    role_pattern = re.compile(r"^[a-z0-9_:\-]{1,32}$")
    for r in roles:
        if not isinstance(r, str):
            abort(400, description=f"Invalid role entry: {r}")
        r_clean = r.strip().lower()
        if not role_pattern.match(r_clean):
            abort(400, description=f"Invalid role format: {r}")
        cleaned.append(r_clean)
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned

def validate_pubkey(pubkey: Optional[str]) -> str:
    if not isinstance(pubkey, str):
        abort(400, description="pubkey must be a string")
    pubkey = pubkey.strip()
    if len(pubkey) < 32 or len(pubkey) > 8192:
        abort(400, description="pubkey length invalid")
    return pubkey

def log_audit(actor: Optional[str], event: str, meta: Optional[Dict[str, Any]] = None):
    record = {
        "actor": actor,
        "event": event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "meta": meta or {}
    }
    audit = AuditLog(agent_id=actor, event=json.dumps(record))
    db.session.add(audit)
    db.session.commit()
    # Structured log
    audit_logger.info(json.dumps(record))

def response_ok(**kwargs):
    return jsonify({"status": "ok", "data": kwargs})

def _b64decode_any(s: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(_with_b64_padding(s))
    except Exception:
        return base64.b64decode(_with_b64_padding(s))

def _with_b64_padding(s: str) -> str:
    pad = (-len(s)) % 4
    return s + ("=" * pad)

def _load_public_key(pubkey_str: str):
    # Try PEM first
    try:
        pub = serialization.load_pem_public_key(pubkey_str.encode("utf-8"))
        return _enforce_key_constraints(pub)
    except Exception:
        pass
    # Try raw base64 for Ed25519
    try:
        raw = _b64decode_any(pubkey_str)
        if len(raw) == 32:
            pub = ed25519.Ed25519PublicKey.from_public_bytes(raw)
            return _enforce_key_constraints(pub)
    except Exception:
        pass
    # Try OpenSSH
    try:
        pub = serialization.load_ssh_public_key(pubkey_str.encode("utf-8"))
        return _enforce_key_constraints(pub)
    except Exception:
        pass
    abort(400, description="Unsupported or invalid public key format")

def _enforce_key_constraints(pub):
    # Enforce RSA modulus size
    if isinstance(pub, rsa.RSAPublicKey):
        if pub.key_size < 2048:
            abort(400, description="RSA key too small; minimum 2048 bits required")
    elif isinstance(pub, ec.EllipticCurvePublicKey):
        curve = pub.curve
        allowed = (ec.SECP256R1, ec.SECP384R1)
        if not isinstance(curve, allowed):
            abort(400, description="Unsupported EC curve; only P-256 and P-384 allowed")
    elif isinstance(pub, ed25519.Ed25519PublicKey):
        pass  # acceptable
    else:
        abort(400, description="Unsupported public key type")
    return pub

def _verify_signature(pubkey_str: str, message: bytes, signature_b64: str) -> bool:
    try:
        signature = _b64decode_any(signature_b64)
    except Exception:
        abort(400, description="Invalid signature encoding")
    pub = _load_public_key(pubkey_str)
    try:
        if isinstance(pub, ed25519.Ed25519PublicKey):
            pub.verify(signature, message)
            return True
        elif isinstance(pub, rsa.RSAPublicKey):
            pub.verify(
                signature,
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        elif isinstance(pub, ec.EllipticCurvePublicKey):
            pub.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        else:
            abort(400, description="Unsupported public key type")
    except InvalidSignature:
        return False
    except Exception as e:
        logger.exception("Signature verification error: %s", e)
        abort(400, description="Signature verification failed due to key error")

def _nonce_key(agent_id: str) -> str:
    return f"nonce:{agent_id}"

def _backoff_key_ip(ip: str) -> str:
    return f"backoff:{ip}"

def _backoff_key_bootstrap_ip(ip: str) -> str:
    return f"backoff:bootstrap:{ip}"

def _store_nonce(agent_id: str, nonce: str, ttl_seconds: int = NONCE_TTL_SECONDS):
    # Only store if no current live nonce or existing expired
    existing = _get_nonce(agent_id)
    if existing:
        return  # replay protection: keep original until consumed
    expires_at = time.time() + ttl_seconds
    if redis_client:
        redis_client.setex(_nonce_key(agent_id), ttl_seconds, nonce)
    else:
        _NONCE_STORE[agent_id] = (nonce, expires_at)

def _pop_nonce(agent_id: str) -> Optional[str]:
    """Atomically consume nonce (prevents race between get and delete)."""
    if redis_client:
        pipe = redis_client.pipeline()
        key = _nonce_key(agent_id)
        pipe.get(key)
        pipe.delete(key)
        val, _ = pipe.execute()
        if val is None:
            return None
        return val
    else:
        item = _NONCE_STORE.pop(agent_id, None)
        if item is None:
            return None
        nonce, exp = item
        if time.time() > exp:
            return None
        return nonce

def _get_nonce(agent_id: str) -> Optional[str]:
    if redis_client:
        return redis_client.get(_nonce_key(agent_id))
    item = _NONCE_STORE.get(agent_id)
    if not item:
        return None
    nonce, exp = item
    if time.time() > exp:
        _NONCE_STORE.pop(agent_id, None)
        return None
    return nonce

def _set_backoff(ip: str, seconds: int):
    until = time.time() + max(0, seconds)
    key = _backoff_key_ip(ip)
    if redis_client:
        redis_client.setex(key, seconds, str(until))
    else:
        _BACKOFF_STORE[key] = until

def _get_backoff_remaining(ip: str) -> float:
    now = time.time()
    key = _backoff_key_ip(ip)
    if redis_client:
        val = redis_client.get(key)
        if val is None:
            return 0.0
        try:
            until = float(val)
        except Exception:
            return 0.0
        return max(0.0, until - now)
    else:
        until = _BACKOFF_STORE.get(key)
        if not until:
            return 0.0
        return max(0.0, until - now)

def _set_bootstrap_backoff(ip: str, seconds: int):
    until = time.time() + max(0, seconds)
    key = _backoff_key_bootstrap_ip(ip)
    if redis_client:
        redis_client.setex(key, seconds, str(until))
    else:
        _BACKOFF_STORE[key] = until

def _get_bootstrap_backoff_remaining(ip: str) -> float:
    now = time.time()
    key = _backoff_key_bootstrap_ip(ip)
    if redis_client:
        val = redis_client.get(key)
        if val is None:
            return 0.0
        try:
            until = float(val)
        except Exception:
            return 0.0
        return max(0.0, until - now)
    else:
        until = _BACKOFF_STORE.get(key)
        if not until:
            return 0.0
        return max(0.0, until - now)

def _record_login_failure(ip: str) -> int:
    ttl = 900
    if redis_client:
        count = redis_client.incr(f"loginfail:{ip}")
        if count == 1:
            redis_client.expire(f"loginfail:{ip}", ttl)
    else:
        count_key = f"loginfail:count:{ip}"
        count = int((_BACKOFF_STORE.get(count_key) or 0)) + 1
        _BACKOFF_STORE[count_key] = count
    backoff_seconds = min(60, 2 ** min(6, int(count)))
    _set_backoff(ip, backoff_seconds)
    return backoff_seconds

def _clear_login_failures(ip: str):
    if redis_client:
        redis_client.delete(f"loginfail:{ip}")
        redis_client.delete(_backoff_key_ip(ip))
    else:
        _BACKOFF_STORE.pop(f"loginfail:count:{ip}", None)
        _BACKOFF_STORE.pop(_backoff_key_ip(ip), None)

def _record_bootstrap_failure(ip: str) -> int:
    ttl = 1800
    if redis_client:
        count = redis_client.incr(f"bootstrapfail:{ip}")
        if count == 1:
            redis_client.expire(f"bootstrapfail:{ip}", ttl)
    else:
        count_key = f"bootstrapfail:count:{ip}"
        count = int((_BACKOFF_STORE.get(count_key) or 0)) + 1
        _BACKOFF_STORE[count_key] = count
    backoff_seconds = min(120, 2 ** min(7, int(count)))
    _set_bootstrap_backoff(ip, backoff_seconds)
    return backoff_seconds

def _clear_bootstrap_failures(ip: str):
    if redis_client:
        redis_client.delete(f"bootstrapfail:{ip}")
        redis_client.delete(_backoff_key_bootstrap_ip(ip))
    else:
        _BACKOFF_STORE.pop(f"bootstrapfail:count:{ip}", None)
        _BACKOFF_STORE.pop(_backoff_key_bootstrap_ip(ip), None)

def _bootstrap_used_key() -> str:
    return "bootstrap:key_used"

def _mark_bootstrap_used():
    global _BOOTSTRAP_KEY_USED_FLAG
    _BOOTSTRAP_KEY_USED_FLAG = True
    if redis_client:
        try:
            redis_client.set(_bootstrap_used_key(), "1")
        except Exception as e:
            logger.warning(f"Failed to persist bootstrap used flag in Redis: {e}")

def _clear_bootstrap_used():
    """Admin reset for disaster recovery."""
    global _BOOTSTRAP_KEY_USED_FLAG
    _BOOTSTRAP_KEY_USED_FLAG = False
    if redis_client:
        try:
            redis_client.delete(_bootstrap_used_key())
        except Exception as e:
            logger.warning(f"Failed to clear bootstrap used flag in Redis: {e}")

def _is_bootstrap_used() -> bool:
    if redis_client:
        try:
            val = redis_client.get(_bootstrap_used_key())
            if val is not None:
                return True
        except Exception as e:
            logger.warning(f"Failed to read bootstrap used flag from Redis: {e}")
    return _BOOTSTRAP_KEY_USED_FLAG

def _is_request_secure(req: request) -> bool:
    try:
        if req.is_secure:
            return True
    except Exception:
        pass
    xf_proto = req.headers.get("X-Forwarded-Proto", "").lower()
    if xf_proto in ("https", "wss"):
        return True
    xf_ssl = req.headers.get("X-Forwarded-Ssl", "").lower()
    if xf_ssl == "on":
        return True
    xf_port = req.headers.get("X-Forwarded-Port", "")
    if xf_port == "443":
        return True
    cf_visitor = req.headers.get("CF-Visitor")
    if cf_visitor and '"scheme":"https"' in cf_visitor:
        return True
    return False

# ============================================================
# Role Enforcement
# ============================================================
def require_roles(required: List[str]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            identity = get_jwt_identity()
            if not identity:
                abort(401, description="Missing identity")
            agent = Agent.query.filter_by(agent_id=identity).first()
            if not agent:
                abort(401, description="Unknown identity")
            roles = agent.roles or []
            missing = [r for r in required if r not in roles]
            if missing:
                abort(403, description=f"Insufficient roles: missing {missing}")
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return jwt_required()(wrapper)
    return decorator

# ============================================================
# JWT Revocation / Logout
# ============================================================
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload.get("jti")
    if not jti:
        return True  # treat missing jti as revoked
    if redis_client:
        try:
            val = redis_client.get(f"{JWT_REVOCATION_PREFIX}{jti}")
            return val is not None
        except Exception:
            pass
    return jti in _REVOKED_TOKENS

def revoke_jti(jti: str, exp_ts: Optional[int] = None):
    if redis_client:
        ttl = 0
        if exp_ts:
            now = int(time.time())
            ttl = max(1, exp_ts - now)
        try:
            redis_client.setex(f"{JWT_REVOCATION_PREFIX}{jti}", ttl if ttl else 3600, "1")
            return
        except Exception:
            pass
    _REVOKED_TOKENS.add(jti)

# ============================================================
# Security Headers
# ============================================================
@app.after_request
def apply_security_headers(resp):
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), payment=()"
    resp.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    resp.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    # Remove Server header if present
    if "Server" in resp.headers:
        del resp.headers["Server"]
    return resp

# ============================================================
# Root / Meta / Favicon
# ============================================================
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Graphix Registry",
        "version": APP_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "redis_backing": redis_storage_uri != "memory://",
        "endpoints": [
            "GET /",
            "GET /meta",
            "GET /health",
            "GET /jwks",
            "GET /spec",
            "POST /auth/nonce",
            "POST /auth/login",
            "POST /auth/logout",
            "POST /registry/bootstrap",
            "POST /registry/bootstrap/reset",
            "POST /registry/onboard",
            "POST /ir/propose",
            "GET /audit/logs"
        ]
    })

@app.route("/meta", methods=["GET"])
def meta():
    return jsonify({
        "service": "Graphix Registry",
        "version": APP_VERSION,
        "rate_limits": {
            "default": "200/day, 50/hour",
            "login": "3/minute + exponential backoff",
            "nonce": "5/minute (replay protection)",
            "bootstrap": "1/minute + exponential backoff",
            "onboard": "3/minute (admin)",
            "propose_ir": "10/minute",
            "audit_logs": "20/minute"
        },
        "auth": {
            "login_flow": "POST /auth/nonce -> sign '<agent_id>:<nonce>' -> POST /auth/login -> Authorization: Bearer <token>",
            "bootstrap_flow": "POST /registry/bootstrap (first agent or X-Bootstrap-Key)",
            "logout": "POST /auth/logout (revokes current token jti)"
        },
        "jwt": {
            "issuer": JWT_ISSUER,
            "audience": JWT_AUDIENCE,
            "jti_revocation": True
        },
        "cors": {
            "enabled": bool(cors_origins),
            "origins": cors_origins
        },
        "limits": {
            "max_request_bytes": app.config.get('MAX_CONTENT_LENGTH', None),
            "ir_max_bytes": IR_MAX_BYTES
        },
        "bootstrap": {
            "tls_required": ENFORCE_HTTPS_BOOTSTRAP,
            "key_used": _is_bootstrap_used()
        }
    })

@app.route("/favicon.ico", methods=["GET"])
def favicon():
    path = os.path.join(os.getcwd(), "favicon.ico")
    if os.path.exists(path):
        return send_file(path, mimetype="image/x-icon")
    abort(404, description="favicon not found")

# JWKS stub (currently symmetric signing)
@app.route("/jwks", methods=["GET"])
def jwks():
    # If changed to asymmetric (e.g., RS256), populate keys here.
    return jsonify({"keys": [], "symmetric": True})

# Minimal OpenAPI spec stub for internal tooling
@app.route("/spec", methods=["GET"])
def openapi_spec():
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Graphix Registry",
            "version": APP_VERSION
        },
        "paths": {
            "/auth/nonce": {"post": {"summary": "Get auth nonce"}},
            "/auth/login": {"post": {"summary": "Login with signed nonce"}},
            "/auth/logout": {"post": {"summary": "Logout / revoke token"}},
            "/registry/bootstrap": {"post": {"summary": "Bootstrap first agent or with key"}},
            "/registry/onboard": {"post": {"summary": "Onboard agent (admin)"}},
            "/ir/propose": {"post": {"summary": "Submit IR proposal"}},
            "/audit/logs": {"get": {"summary": "Get audit logs (paginated)"}},
            "/health": {"get": {"summary": "Health status"}},
        }
    }
    return jsonify(spec)

# ============================================================
# Authentication / Bootstrap
# ============================================================
@app.route("/auth/nonce", methods=["POST"])
@limiter.limit("5 per minute")
def auth_nonce():
    data = safe_json()
    agent_id = validate_agent_id(data.get("agent_id"))
    # Replay protection: if valid unconsumed nonce exists, return same nonce
    existing = _get_nonce(agent_id)
    if existing:
        log_audit(None, f"Nonce reissued (existing) for agent_id={agent_id}", {"replay_protection": True})
        return jsonify({"nonce": existing, "agent_id": agent_id})
    agent = Agent.query.filter_by(agent_id=agent_id).first()
    nonce = secrets.token_urlsafe(32)
    _store_nonce(agent_id, nonce)
    if not agent:
        log_audit(None, f"Nonce issued for agent_id={agent_id} (existence not disclosed)")
    else:
        log_audit(agent_id, "Nonce issued")
    return jsonify({"nonce": nonce, "agent_id": agent_id})

@app.route("/auth/login", methods=["POST"])
@limiter.limit("3 per minute")
def login():
    ip = get_remote_address()
    wait_remaining = _get_backoff_remaining(ip)
    if wait_remaining > 0:
        abort(429, description=f"Login temporarily blocked. Try again in {int(wait_remaining)}s due to repeated failed attempts.")

    data = safe_json()
    agent_id = validate_agent_id(data.get("agent_id"))
    nonce = data.get("nonce")
    signature = data.get("signature")

    if not nonce or not isinstance(nonce, str) or len(nonce) > 512:
        abort(400, description="nonce is required and must be a string")
    if not signature or not isinstance(signature, str) or len(signature) > 4096:
        abort(400, description="signature is required and must be a base64 string")

    # Atomic pop to prevent race
    stored_nonce = _pop_nonce(agent_id)
    if not stored_nonce or not secrets.compare_digest(stored_nonce, nonce):
        backoff = _record_login_failure(ip)
        log_audit(agent_id, "Login failed: nonce invalid or expired",
                  {"ip": ip, "backoff_seconds": backoff})
        # Add slight random jitter to reduce timing side-channels
        time.sleep((5 + secrets.randbelow(25)) / 1000.0)
        abort(401, description="Invalid nonce")

    agent = Agent.query.filter_by(agent_id=agent_id).first()
    if not agent:
        backoff = _record_login_failure(ip)
        log_audit(agent_id, "Login failed: agent not found",
                  {"ip": ip, "backoff_seconds": backoff})
        time.sleep((5 + secrets.randbelow(25)) / 1000.0)
        abort(401, description="Invalid credentials")

    message = f"{agent_id}:{nonce}".encode("utf-8")
    verified = _verify_signature(agent.pubkey, message, signature)

    if not verified:
        backoff = _record_login_failure(ip)
        log_audit(agent_id, "Login failed: signature invalid",
                  {"ip": ip, "backoff_seconds": backoff})
        time.sleep((5 + secrets.randbelow(25)) / 1000.0)
        abort(401, description="Invalid credentials")

    _clear_login_failures(ip)
    claims = {
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": secrets.token_hex(16)
    }
    token = create_access_token(identity=agent_id, additional_claims=claims)
    log_audit(agent_id, "Login success", {"ip": ip})
    return jsonify({"access_token": token, "agent_id": agent_id, "issuer": JWT_ISSUER, "audience": JWT_AUDIENCE})

@app.route("/auth/logout", methods=["POST"])
@jwt_required()
def logout():
    if not JWT_ENABLE_LOGOUT:
        abort(400, description="Logout not enabled")
    jti = get_jwt().get("jti")
    exp = get_jwt().get("exp")
    if jti:
        revoke_jti(jti, exp)
        log_audit(get_jwt_identity(), "Logout / token revoked", {"jti": jti})
        return jsonify({"status": "revoked", "jti": jti})
    abort(400, description="Token missing jti")

@app.route("/registry/bootstrap", methods=["POST"])
@limiter.limit("1 per minute")
def registry_bootstrap():
    """
    Create first agent OR allow creation with BOOTSTRAP_KEY.
    Guard rails:
    - Enforce HTTPS/TLS only (reverse proxy sets X-Forwarded-Proto/SSL).
    - Constant-time comparison for bootstrap key verification.
    - Rate-limit invalid attempts with exponential backoff.
    - Log all failed attempts to audit chain.
    - Auto-expire bootstrap key after successful use (disallow reuse).
    """
    ip = get_remote_address()
    if ENFORCE_HTTPS_BOOTSTRAP and not _is_request_secure(request):
        log_audit(None, f"Bootstrap attempt blocked: non-TLS access", {"ip": ip})
        abort(403, description="Bootstrap endpoint requires HTTPS/TLS")

    wait_remaining = _get_bootstrap_backoff_remaining(ip)
    if wait_remaining > 0:
        abort(429, description=f"Bootstrap temporarily blocked. Try again in {int(wait_remaining)}s seconds.")

    data = safe_json()
    existing_count = Agent.query.count()
    provided_key = request.headers.get("X-Bootstrap-Key", "")

    configured_key = (BOOTSTRAP_KEY or "")
    key_valid = secrets.compare_digest(provided_key, configured_key)
    bootstrap_used = _is_bootstrap_used()
    allow_without_key = (existing_count == 0) and (not bootstrap_used)

    if not allow_without_key:
        if not key_valid:
            backoff = _record_bootstrap_failure(ip)
            log_audit(None, "Bootstrap failed: invalid key", {"ip": ip, "backoff": backoff})
            abort(403, description="Bootstrap not allowed: invalid bootstrap key")
        if bootstrap_used:
            log_audit(None, "Bootstrap failed: key already used", {"ip": ip})
            abort(403, description="Bootstrap not allowed: bootstrap key has expired")

    agent_id = validate_agent_id(data.get("agent_id"))
    pubkey = validate_pubkey(data.get("pubkey"))
    roles = validate_roles(data.get("roles", ["admin"]))
    try:
        trust = float(data.get("trust", 0.9))
    except Exception:
        abort(400, description="trust must be a float")
    if trust < 0.0 or trust > 1.0:
        abort(400, description="trust must be between 0 and 1")

    if Agent.query.filter_by(agent_id=agent_id).first():
        abort(409, description="Agent already exists")
    # Enforce public key uniqueness
    if Agent.query.filter_by(pubkey=pubkey).first():
        abort(409, description="Public key already registered")

    _ = _load_public_key(pubkey)

    new_agent = Agent(agent_id=agent_id, pubkey=pubkey, roles=roles, trust=trust)
    db.session.add(new_agent)
    db.session.commit()
    claims = {
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": secrets.token_hex(16)
    }
    token = create_access_token(identity=agent_id, additional_claims=claims)

    _clear_bootstrap_failures(ip)
    _mark_bootstrap_used()
    log_audit(agent_id, "Bootstrap created agent", {"ip": ip})
    return jsonify({
        "status": "bootstrapped",
        "agent_id": agent_id,
        "access_token": token,
        "roles": roles
    })

@app.route("/registry/bootstrap/reset", methods=["POST"])
@require_roles(["admin"])
def bootstrap_reset():
    """
    Admin endpoint to reset bootstrap key usage (disaster recovery).
    Requires header X-Bootstrap-Key match if BOOTSTRAP_KEY is set.
    """
    if not BOOTSTRAP_KEY:
        abort(400, description="No BOOTSTRAP_KEY configured")
    provided = request.headers.get("X-Bootstrap-Key", "")
    if not secrets.compare_digest(provided, BOOTSTRAP_KEY):
        abort(403, description="Invalid bootstrap key for reset")
    _clear_bootstrap_used()
    log_audit(get_jwt_identity(), "Bootstrap key usage reset")
    return jsonify({"status": "reset", "bootstrap_used": _is_bootstrap_used()})

# ============================================================
# Health
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    db_latency_ms = None
    redis_latency_ms = None
    try:
        start_db = time.time()
        db.session.execute(select(1))
        db_latency_ms = (time.time() - start_db) * 1000.0
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error(f"DB health check failed: {e}")
    if redis_client:
        try:
            start_redis = time.time()
            redis_client.ping()
            redis_latency_ms = (time.time() - start_redis) * 1000.0
            redis_status = "ok"
        except Exception as e:
            redis_status = f"error: {e}"
    else:
        redis_status = "unavailable"
    return jsonify({
        "status": "healthy" if db_status == "ok" else "degraded",
        "db": db_status,
        "db_latency_ms": db_latency_ms,
        "redis": redis_status,
        "redis_latency_ms": redis_latency_ms,
        "time": datetime.utcnow().isoformat() + "Z"
    })

# ============================================================
# Registry Onboard (admin only)
# ============================================================
@app.route("/registry/onboard", methods=["POST"])
@require_roles(["admin"])
@limiter.limit("3 per minute")
def registry_onboard():
    actor = get_jwt_identity()
    data = safe_json()
    agent_id = validate_agent_id(data.get("agent_id"))
    pubkey = validate_pubkey(data.get("pubkey"))
    roles = validate_roles(data.get("roles", []))
    try:
        trust = float(data.get("trust", 0.5))
    except Exception:
        abort(400, description="trust must be a float")
    if trust < 0.0 or trust > 1.0:
        abort(400, description="trust must be between 0 and 1")

    if Agent.query.filter_by(agent_id=agent_id).first():
        abort(409, description="Agent already exists")
    if Agent.query.filter_by(pubkey=pubkey).first():
        abort(409, description="Public key already registered")

    _ = _load_public_key(pubkey)

    new_agent = Agent(agent_id=agent_id, pubkey=pubkey, roles=roles, trust=trust)
    db.session.add(new_agent)
    db.session.commit()

    log_audit(actor, "Onboarded agent", {"agent_id": agent_id})
    return jsonify({"status": "registered", "agent_id": agent_id})

# ============================================================
# IR Proposal (trust-aware constraints)
# ============================================================
@app.route("/ir/propose", methods=["POST"])
@jwt_required()
@limiter.limit("10 per minute")
def propose_ir():
    actor_id = get_jwt_identity()
    actor_agent = Agent.query.filter_by(agent_id=actor_id).first()
    if not actor_agent:
        abort(401, description="Unknown identity")

    data = safe_json()
    ir = data.get("ir")
    agent_id = validate_agent_id(data.get("agent_id"))

    if not isinstance(ir, dict):
        abort(400, description="ir must be a JSON object")

    serialized_ir = json.dumps(ir, separators=(",", ":"))
    if len(serialized_ir.encode("utf-8")) > IR_MAX_BYTES:
        abort(400, description=f"IR too large in bytes (> {IR_MAX_BYTES})")

    nodes = ir.get("nodes")
    edges = ir.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        abort(400, description="Invalid IR structure: nodes & edges must be lists")

    # Size limits
    if len(nodes) > 10000 or len(edges) > 20000:
        abort(400, description="IR too large: nodes or edges exceed limits")

    # Additional trust restrictions
    if actor_agent.trust < 0.4 and (len(nodes) > 2000 or len(edges) > 4000):
        abort(403, description="Low trust agent cannot submit IR exceeding 2000 nodes / 4000 edges")

    if not all(isinstance(n, dict) for n in nodes):
        abort(400, description="Each node must be an object")
    if not all(isinstance(e, dict) for e in edges):
        abort(400, description="Each edge must be an object")

    if not Agent.query.filter_by(agent_id=agent_id).first():
        abort(400, description="agent_id not registered")

    proposal = IRProposal(agent_id=agent_id, ir_json=serialized_ir)
    db.session.add(proposal)
    db.session.commit()

    log_audit(actor_id, "Proposed IR", {"target_agent_id": agent_id, "proposal_id": proposal.id})
    return jsonify({
        "status": "proposal_received",
        "agent_id": agent_id,
        "proposal_id": proposal.id
    })

# ============================================================
# Audit Logs (paginated & role-based)
# ============================================================
@app.route("/audit/logs", methods=["GET"])
@jwt_required()
@limiter.limit("20 per minute")
def get_audit_logs():
    actor = get_jwt_identity()
    actor_agent = Agent.query.filter_by(agent_id=actor).first()
    roles = actor_agent.roles if actor_agent else []
    # Only admins can filter by arbitrary agent_id; others limited to self
    agent_filter = request.args.get("agent_id")
    if agent_filter and "admin" not in roles:
        abort(403, description="Only admin can filter by agent_id")
    if agent_filter:
        agent_filter = validate_agent_id(agent_filter)

    # Pagination
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        abort(400, description="limit and offset must be integers")
    if limit <= 0 or limit > 500:
        abort(400, description="limit must be between 1 and 500")
    if offset < 0:
        abort(400, description="offset must be >= 0")

    query = AuditLog.query
    if agent_filter:
        query = query.filter(AuditLog.agent_id == agent_filter)

    logs = query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
    result = []
    for log in logs:
        try:
            parsed = json.loads(log.event)
        except Exception:
            parsed = {"raw": log.event}
        parsed["id"] = log.id
        parsed["agent_id"] = log.agent_id
        parsed["timestamp"] = log.timestamp.isoformat()
        result.append(parsed)

    log_audit(actor, "Audit log viewed", {"count": len(result), "offset": offset, "limit": limit})
    return jsonify({"count": len(result), "logs": result, "offset": offset, "limit": limit})

# ============================================================
# Logout already implemented above
# ============================================================

# ============================================================
# Error Handlers (uniform JSON shape)
# ============================================================
@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"Bad request: {error}")
    return jsonify({"error": "Bad Request", "detail": error.description}), 400

@app.errorhandler(401)
def unauthorized(error):
    logger.warning(f"Unauthorized access: {error}")
    return jsonify({"error": "Unauthorized", "detail": error.description}), 401

@app.errorhandler(403)
def forbidden(error):
    logger.warning(f"Forbidden: {error}")
    return jsonify({"error": "Forbidden", "detail": error.description}), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found", "detail": error.description}), 404

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "Request Entity Too Large", "detail": "Request payload exceeds configured maximum size"}), 413

@app.errorhandler(429)
def rate_limit_exceeded(error):
    # Flask-Limiter passes an exception-like object; unify detail
    return jsonify({"error": "Too Many Requests", "detail": str(error)}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal Server Error", "detail": "An unexpected error occurred"}), 500

# ============================================================
# Main Entrypoint
# ============================================================
if __name__ == "__main__":
    # Recommended: run under a production WSGI/ASGI server (gunicorn/uvicorn) in prod
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
