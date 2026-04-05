"""Flash messaging system for the Vulcan unified platform.

Provides FlashMessage and FlashMessageManager for displaying
errors, warnings, and info messages to users.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional


class FlashMessage:
    """Flash message for displaying errors/warnings."""

    def __init__(self, level: str, message: str, details: Optional[str] = None):
        self.level = level  # error, warning, info, success
        self.message = message
        self.details = details
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class FlashMessageManager:
    """Thread-safe flash message manager."""

    def __init__(self, max_messages: int = 10):
        self.messages = deque(maxlen=max_messages)
        self._lock = asyncio.Lock()

    async def add_message(
        self, level: str, message: str, details: Optional[str] = None
    ):
        """Add a new flash message."""
        async with self._lock:
            self.messages.append(FlashMessage(level, message, details))

    async def get_recent_messages(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent flash messages."""
        async with self._lock:
            return [msg.to_dict() for msg in list(self.messages)[-limit:]]

    async def clear_messages(self):
        """Clear all messages."""
        async with self._lock:
            self.messages.clear()
