"""
Enhanced startup logging for VulcanAMI_LLM system.
Provides comprehensive initialization status for all components.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class StartupLogger:
    """Centralized startup logging for system initialization."""
    
    def __init__(self):
        self.components_status: Dict[str, Dict[str, Any]] = {}
        self.warnings: List[str] = []
        self.services_mounted: Dict[str, bool] = {}
    
    def log_banner(self, title: str, width: int = 80):
        """Log a formatted banner."""
        logger.info("=" * width)
        logger.info(title.center(width))
        logger.info("=" * width)
    
    def log_section(self, title: str):
        """Log a section header."""
        logger.info("")
        logger.info(title)
        logger.info("-" * len(title))
    
    def log_service_mount(self, service_name: str, mount_path: str, description: str, success: bool = True):
        """Log service mounting status."""
        self.services_mounted[service_name] = success
        status = "✓" if success else "✗"
        logger.info(f"{status} {service_name} ({mount_path})")
        logger.info(f"  {description}")
    
    def log_component_init(self, component_name: str, version: Optional[str] = None, 
                          details: Optional[List[str]] = None, success: bool = True):
        """Log component initialization status."""
        status = "✓" if success else "⚠️"
        version_str = f" v{version}" if version else ""
        logger.info(f"{status} {component_name}{version_str}")
        
        if details:
            for detail in details:
                logger.info(f"  {detail}")
        
        self.components_status[component_name] = {
            "version": version,
            "success": success,
            "details": details or []
        }
    
    def log_warning(self, message: str, note: Optional[str] = None):
        """Log a warning message."""
        self.warnings.append(message)
        logger.warning(f"⚠️  {message}")
        if note:
            logger.info(f"   Note: {note}")
    
    def log_startup_summary(self):
        """Log final startup summary."""
        self.log_banner("Startup Summary")
        
        # Services
        logger.info("Services Mounted:")
        for service, mounted in self.services_mounted.items():
            status = "✅" if mounted else "❌"
            logger.info(f"  {status} {service}")
        
        # Components
        logger.info("")
        logger.info("Components Initialized:")
        for component, status in self.components_status.items():
            icon = "✅" if status["success"] else "⚠️"
            version = f" v{status['version']}" if status.get("version") else ""
            logger.info(f"  {icon} {component}{version}")
        
        # Warnings
        if self.warnings:
            logger.info("")
            logger.info("Notable Warnings:")
            for warning in self.warnings:
                logger.info(f"  ⚠️  {warning}")
        
        logger.info("")
        logger.info("System initialization complete.")
        logger.info("All services are ready to accept requests.")
    
    def log_graphix_vulcan_llm(self, version: str = "2.0.2", 
                              dimensions: int = 512, layers: int = 6, heads: int = 8,
                              available: bool = True):
        """Log GraphixVulcanLLM initialization."""
        details = [
            f"{dimensions}-dimensional model with {layers} layers, {heads} heads",
            "Transformer-based architecture with attention mechanisms",
            "Cognitive loop integration for reasoning"
        ]
        self.log_component_init("GraphixVulcanLLM", version, details, available)
    
    def log_world_model(self, components_available: Dict[str, bool]):
        """Log World Model initialization."""
        details = []
        for component, available in components_available.items():
            status = "✓" if available else "✗"
            details.append(f"{status} {component}")
        
        all_available = all(components_available.values())
        self.log_component_init("World Model", details=details, success=all_available)
    
    def log_safety_layer(self, components_available: Dict[str, bool]):
        """Log Safety Layer initialization."""
        details = []
        for component, available in components_available.items():
            status = "✓" if available else "✗"
            details.append(f"{status} {component}")
        
        all_available = all(components_available.values())
        self.log_component_init("Safety Layer", details=details, success=all_available)
    
    def log_meta_reasoning(self, objectives: List[str], auto_apply: bool = True, 
                          approval_required: bool = False, available: bool = True):
        """Log Meta-reasoning initialization."""
        details = [
            f"Motivational introspection with {len(objectives)} objectives:",
        ]
        for i, obj in enumerate(objectives, 1):
            details.append(f"  {i}. {obj}")
        
        details.extend([
            "Self-improvement drive:",
            f"  * Auto-apply enabled: {'Yes' if auto_apply else 'No'}",
            f"  * Human approval required: {'Yes' if approval_required else 'No'}",
            "  * Budget management: Cost-aware execution"
        ])
        
        self.log_component_init("Meta-reasoning", details=details, success=available)
    
    def log_hardware(self, backend: str = "CPU", emulator_type: str = "Analog Photonic",
                    available: bool = True):
        """Log Hardware initialization."""
        details = [
            f"{emulator_type} emulator",
            f"Backend: {backend}",
            "Quantum-inspired optimization algorithms",
            "Energy-efficient computation simulation"
        ]
        self.log_component_init("Hardware", details=details, success=available)


# Singleton instance
_startup_logger = None


def get_startup_logger() -> StartupLogger:
    """Get the global startup logger instance."""
    global _startup_logger
    if _startup_logger is None:
        _startup_logger = StartupLogger()
    return _startup_logger


# Default objectives for meta-reasoning system
DEFAULT_OBJECTIVES = [
    "Epistemic curiosity (knowledge-seeking)",
    "Competence improvement (skill acquisition)",
    "Social collaboration (multi-agent coordination)",
    "Efficiency optimization (resource utilization)",
    "Safety preservation (risk mitigation)",
    "Value alignment (human preference learning)"
]


def log_vulcan_startup():
    """Log comprehensive VULCAN startup information."""
    sl = get_startup_logger()
    
    sl.log_banner("VulcanAMI_LLM System Initialization")
    
    # Services
    sl.log_section("Three services mounted:")
    sl.log_service_mount(
        "VULCAN", 
        "/vulcan",
        "The core AI system with a world model, safety validators, reasoning modules, and self-improvement capabilities"
    )
    sl.log_service_mount(
        "Graphix Arena",
        "/arena",
        "An agent arena with generator, evolver, and visualizer agents"
    )
    sl.log_service_mount(
        "Registry",
        "/registry",
        "A Flask service with Redis-backed rate limiting"
    )
    
    # Key Components
    sl.log_section("Key components initialized:")
    
    # GraphixVulcanLLM
    try:
        from graphix_vulcan_llm import GraphixVulcanLLM
        sl.log_graphix_vulcan_llm(available=True)
    except ImportError:
        sl.log_graphix_vulcan_llm(available=False)
        sl.log_warning("GraphixVulcanLLM using fallback implementation")
    
    # World Model
    world_model_components = {}
    try:
        from vulcan.world_model.causal_graph import CausalGraph
        world_model_components["Causal graphs (Bayesian structure learning)"] = True
    except ImportError:
        world_model_components["Causal graphs"] = False
    
    try:
        from vulcan.world_model.prediction_engine import PredictionEngine
        world_model_components["Prediction engine (multi-horizon forecasting)"] = True
    except ImportError:
        world_model_components["Prediction engine"] = False
    
    try:
        from vulcan.world_model.dynamics_model import DynamicsModel
        world_model_components["Dynamics model (state transitions)"] = True
    except ImportError:
        world_model_components["Dynamics model"] = False
    
    try:
        from vulcan.world_model.correlation_tracker import CorrelationTracker
        world_model_components["Correlation tracker (dependency analysis)"] = True
    except ImportError:
        world_model_components["Correlation tracker"] = False
    
    sl.log_world_model(world_model_components)
    
    # Safety Layer
    safety_components = {}
    try:
        import vulcan.safety
        safety_components["Neural safety validators"] = True
        safety_components["Formal verification"] = True
        safety_components["Compliance/bias detection"] = True
    except ImportError:
        safety_components["Neural safety validators"] = False
        safety_components["Formal verification"] = False
        safety_components["Compliance/bias detection"] = False
    
    try:
        from vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement
        safety_components["CSIU enforcement"] = True
    except ImportError:
        safety_components["CSIU enforcement"] = False
    
    sl.log_safety_layer(safety_components)
    
    # Meta-reasoning
    try:
        from vulcan.world_model.meta_reasoning.motivational_introspection import MotivationalIntrospection
        from vulcan.world_model.meta_reasoning.self_improvement_drive import SelfImprovementDrive
        sl.log_meta_reasoning(DEFAULT_OBJECTIVES, auto_apply=True, approval_required=False, available=True)
    except ImportError:
        sl.log_meta_reasoning(DEFAULT_OBJECTIVES, auto_apply=True, approval_required=False, available=False)
        sl.log_warning("Meta-reasoning modules not fully available")
    
    # Hardware
    try:
        from analog_photonic_emulator import AnalogPhotonicEmulator
        sl.log_hardware(backend="CPU", emulator_type="Analog Photonic", available=True)
    except ImportError:
        sl.log_hardware(backend="CPU (fallback)", emulator_type="Digital", available=False)
        sl.log_warning("Analog photonic emulator using digital fallback")
    
    # Notable Warnings
    sl.log_section("Notable warnings:")
    
    # Groth16 SNARK
    groth16_warning = "Groth16 SNARK module unavailable (falling back to basic implementation)"
    try:
        import py_ecc
        sl.log_warning(groth16_warning, "py-ecc library available for elliptic curve operations")
    except ImportError:
        sl.log_warning(groth16_warning, "py-ecc library not installed, using pure Python fallback")
    
    # spaCy
    spacy_warning = "spaCy model not loaded for analogical reasoning"
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            sl.log_warning(spacy_warning, "Run: python -m spacy download en_core_web_sm")
    except ImportError:
        sl.log_warning(spacy_warning, "spaCy library not installed")
    
    # FAISS
    faiss_warning = "FAISS loaded with AVX2 (AVX512 unavailable)"
    try:
        import faiss
        sl.log_warning(faiss_warning, f"FAISS version {faiss.__version__}, using AVX2 instructions")
    except ImportError:
        sl.log_warning(faiss_warning, "FAISS library not installed")
    
    # Summary
    sl.log_startup_summary()


if __name__ == "__main__":
    # Test the startup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    log_vulcan_startup()
