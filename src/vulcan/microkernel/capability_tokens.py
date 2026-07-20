"""In-process capability tokens bound to exact principal and operation context."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from secrets import token_hex
from threading import Lock

from .authority import AuthorityError, AuthorityGrant, Operation, require_authority, AuditSink
from .principals import Principal, digest


@dataclass(frozen=True)
class CapabilityToken:
    principal_digest: str
    release_digest: str
    operation: Operation
    episode_id: str
    resource_digest: str
    expires_at: datetime
    nonce: str = field(default_factory=lambda: token_hex(32))
    token_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if len(self.nonce) < 32:
            raise ValueError("nonce is too short")
        object.__setattr__(self, "token_digest", digest(self.to_json(include_digest=False)))

    def to_json(self, *, include_digest: bool = True) -> dict[str, str]:
        payload = {"episode_id": self.episode_id, "expires_at": self.expires_at.astimezone(timezone.utc).isoformat(), "nonce": self.nonce, "operation": self.operation.value, "principal_digest": self.principal_digest, "release_digest": self.release_digest, "resource_digest": self.resource_digest}
        if include_digest:
            payload["token_digest"] = self.token_digest
        return payload

    @classmethod
    def from_json(cls, _payload: object) -> "CapabilityToken":
        raise AuthorityError("serialized capability tokens are not accepted in-process")


class CapabilityTokenIssuer:
    def __init__(self) -> None:
        self._issued: set[str] = set()
        self._used: set[str] = set()
        self._lock = Lock()

    def issue(self, *, principal: Principal, grant: AuthorityGrant, operation: Operation, episode_id: str, resource_digest: str, expires_at: datetime, audit: AuditSink, clock) -> CapabilityToken:
        require_authority(principal=principal, grant=grant, operation=operation, episode_id=episode_id, resource_digest=resource_digest, audit=audit, clock=clock)
        token = CapabilityToken(principal.identity_digest, principal.release_digest, operation, episode_id, resource_digest, expires_at)
        with self._lock:
            self._issued.add(token.token_digest)
        return token

    def consume(self, *, token: CapabilityToken, principal: Principal, operation: Operation, episode_id: str, resource_digest: str, now: datetime) -> None:
        if not isinstance(token, CapabilityToken):
            raise AuthorityError("capability object required")
        if token.principal_digest != principal.identity_digest or token.release_digest != principal.release_digest:
            raise AuthorityError("capability principal mismatch")
        if token.operation is not operation or token.episode_id != episode_id or token.resource_digest != resource_digest:
            raise AuthorityError("capability scope mismatch")
        if now.astimezone(timezone.utc) >= token.expires_at.astimezone(timezone.utc):
            raise AuthorityError("capability expired")
        with self._lock:
            if token.token_digest not in self._issued:
                raise AuthorityError("capability was not issued by this issuer")
            if token.token_digest in self._used:
                raise AuthorityError("capability replay denied")
            self._used.add(token.token_digest)
