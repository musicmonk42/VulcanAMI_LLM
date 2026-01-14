"""
Startup Trace Logger

Provides auditable trace logging for VULCAN startup sequence.
Records all component registrations, callbacks, and initialization status
for debugging and future-proofing modular development.

Module: vulcan.server.startup.trace_logger
Author: Vulcan AI Team
"""

import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ComponentRegistration:
    """Records a component registration event."""
    name: str
    component_type: str  # 'tool', 'agent', 'classifier', 'orchestrator', 'callback'
    status: str  # 'registered', 'failed', 'skipped'
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class StartupTraceLogger:
    """
    Records and logs all startup registrations for audit trail.
    
    Provides a comprehensive trace of:
    - Tool registrations
    - Agent pool initialization
    - Classifier setup
    - Router configuration
    - Workflow engine initialization
    - Callback chain wiring
    - Orchestrator setup
    """
    
    def __init__(self):
        self.registrations: List[ComponentRegistration] = []
        self.start_time = time.time()
        self._tool_count = 0
        self._agent_count = 0
        self._callback_count = 0
        
    def log_tool_registration(
        self,
        tool_name: str,
        tool_type: str,
        status: str = "registered",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log a tool registration event."""
        reg = ComponentRegistration(
            name=tool_name,
            component_type="tool",
            status=status,
            timestamp=time.time(),
            details=details or {"type": tool_type},
            error=error
        )
        self.registrations.append(reg)
        
        if status == "registered":
            self._tool_count += 1
            logger.info(f"🔧 Tool registered: {tool_name} ({tool_type})")
        elif status == "failed":
            logger.warning(f"✗ Tool registration failed: {tool_name} - {error}")
        else:
            logger.debug(f"⊗ Tool skipped: {tool_name}")
            
    def log_agent_registration(
        self,
        agent_id: str,
        capability: str,
        status: str = "registered",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log an agent registration event."""
        reg = ComponentRegistration(
            name=agent_id,
            component_type="agent",
            status=status,
            timestamp=time.time(),
            details=details or {"capability": capability},
            error=error
        )
        self.registrations.append(reg)
        
        if status == "registered":
            self._agent_count += 1
            logger.info(f"🤖 Agent registered: {agent_id} (capability={capability})")
        elif status == "failed":
            logger.warning(f"✗ Agent registration failed: {agent_id} - {error}")
            
    def log_classifier_init(
        self,
        classifier_name: str,
        status: str = "registered",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log classifier initialization."""
        reg = ComponentRegistration(
            name=classifier_name,
            component_type="classifier",
            status=status,
            timestamp=time.time(),
            details=details or {},
            error=error
        )
        self.registrations.append(reg)
        
        if status == "registered":
            logger.info(f"🎯 Classifier initialized: {classifier_name}")
        elif status == "failed":
            logger.warning(f"✗ Classifier init failed: {classifier_name} - {error}")
            
    def log_orchestrator_init(
        self,
        orchestrator_name: str,
        status: str = "registered",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Log orchestrator initialization."""
        reg = ComponentRegistration(
            name=orchestrator_name,
            component_type="orchestrator",
            status=status,
            timestamp=time.time(),
            details=details or {},
            error=error
        )
        self.registrations.append(reg)
        
        if status == "registered":
            logger.info(f"🎭 Orchestrator initialized: {orchestrator_name}")
        elif status == "failed":
            logger.warning(f"✗ Orchestrator init failed: {orchestrator_name} - {error}")
            
    def log_callback_registration(
        self,
        source: str,
        target: str,
        callback_type: str,
        status: str = "registered",
        error: Optional[str] = None
    ) -> None:
        """Log callback chain registration."""
        reg = ComponentRegistration(
            name=f"{source}→{target}",
            component_type="callback",
            status=status,
            timestamp=time.time(),
            details={"source": source, "target": target, "type": callback_type},
            error=error
        )
        self.registrations.append(reg)
        
        if status == "registered":
            self._callback_count += 1
            logger.info(f"🔗 Callback registered: {source} → {target} ({callback_type})")
        elif status == "failed":
            logger.warning(f"✗ Callback registration failed: {source}→{target} - {error}")
            
    def print_summary(self) -> None:
        """Print comprehensive startup summary with all registrations."""
        elapsed = time.time() - self.start_time
        
        logger.info("=" * 70)
        logger.info("VULCAN-AGI STARTUP TRACE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total startup time: {elapsed:.2f}s")
        logger.info(f"Total registrations: {len(self.registrations)}")
        logger.info("")
        
        # Count by type and status
        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        failures: List[ComponentRegistration] = []
        
        for reg in self.registrations:
            by_type[reg.component_type] = by_type.get(reg.component_type, 0) + 1
            by_status[reg.status] = by_status.get(reg.status, 0) + 1
            if reg.status == "failed":
                failures.append(reg)
                
        logger.info("📊 Registration Summary:")
        logger.info(f"  • Tools: {self._tool_count}")
        logger.info(f"  • Agents: {self._agent_count}")
        logger.info(f"  • Classifiers: {by_type.get('classifier', 0)}")
        logger.info(f"  • Orchestrators: {by_type.get('orchestrator', 0)}")
        logger.info(f"  • Callbacks: {self._callback_count}")
        logger.info("")
        
        logger.info("✓ Status Summary:")
        logger.info(f"  • Registered: {by_status.get('registered', 0)}")
        logger.info(f"  • Failed: {by_status.get('failed', 0)}")
        logger.info(f"  • Skipped: {by_status.get('skipped', 0)}")
        logger.info("")
        
        if failures:
            logger.warning("⚠️  Failures:")
            for fail in failures:
                logger.warning(f"  • {fail.component_type}/{fail.name}: {fail.error}")
            logger.info("")
            
        # List all tools
        tools = [r for r in self.registrations if r.component_type == "tool" and r.status == "registered"]
        if tools:
            logger.info("🔧 Registered Tools:")
            for tool in tools:
                tool_type = tool.details.get("type", "unknown")
                logger.info(f"  • {tool.name} ({tool_type})")
            logger.info("")
            
        # List all agents
        agents = [r for r in self.registrations if r.component_type == "agent" and r.status == "registered"]
        if agents:
            logger.info("🤖 Registered Agents:")
            for agent in agents:
                capability = agent.details.get("capability", "unknown")
                logger.info(f"  • {agent.name} ({capability})")
            logger.info("")
            
        # List all callbacks
        callbacks = [r for r in self.registrations if r.component_type == "callback" and r.status == "registered"]
        if callbacks:
            logger.info("🔗 Registered Callbacks:")
            for cb in callbacks:
                cb_type = cb.details.get("type", "unknown")
                logger.info(f"  • {cb.name} ({cb_type})")
            logger.info("")
            
        logger.info("=" * 70)
        logger.info("STARTUP TRACE COMPLETE - System ready for queries")
        logger.info("=" * 70)
        
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary as dictionary for API/status endpoints."""
        elapsed = time.time() - self.start_time
        
        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        
        for reg in self.registrations:
            by_type[reg.component_type] = by_type.get(reg.component_type, 0) + 1
            by_status[reg.status] = by_status.get(reg.status, 0) + 1
            
        return {
            "startup_time_seconds": elapsed,
            "total_registrations": len(self.registrations),
            "by_type": by_type,
            "by_status": by_status,
            "tools_count": self._tool_count,
            "agents_count": self._agent_count,
            "callbacks_count": self._callback_count,
        }


# Global singleton for startup tracing
_startup_trace: Optional[StartupTraceLogger] = None


def get_startup_trace() -> StartupTraceLogger:
    """Get or create the global startup trace logger."""
    global _startup_trace
    if _startup_trace is None:
        _startup_trace = StartupTraceLogger()
    return _startup_trace


def reset_startup_trace() -> None:
    """Reset the startup trace (for testing)."""
    global _startup_trace
    _startup_trace = None
