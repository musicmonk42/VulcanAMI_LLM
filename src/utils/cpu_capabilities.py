"""
Enhanced CPU Capability Detection Module

Provides comprehensive detection of CPU instruction sets across platforms:
- x86/x64: AVX, AVX2, AVX-512, SSE variants
- ARM: NEON, SVE
- Platform-specific optimizations

Used by FAISS and other performance-critical components to determine
optimal instruction set usage and provide detailed diagnostics.
"""

import platform
import logging
import threading
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CPUCapabilities:
    """Comprehensive CPU capability information"""

    architecture: str = ""
    platform: str = ""
    processor: str = ""

    # x86/x64 instruction sets
    has_sse: bool = False
    has_sse2: bool = False
    has_sse3: bool = False
    has_ssse3: bool = False
    has_sse4_1: bool = False
    has_sse4_2: bool = False
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512f: bool = False
    has_avx512dq: bool = False
    has_avx512bw: bool = False
    has_avx512vl: bool = False
    has_fma: bool = False

    # ARM instruction sets
    has_neon: bool = False
    has_sve: bool = False
    has_sve2: bool = False

    # Additional features
    cpu_cores: int = 0
    cpu_flags: Set[str] = field(default_factory=set)

    def get_best_vector_instruction_set(self) -> str:
        """Determine the best available vector instruction set"""
        if self.architecture.lower().startswith(
            "arm"
        ) or self.architecture.lower().startswith("aarch"):
            if self.has_sve2:
                return "ARM SVE2"
            elif self.has_sve:
                return "ARM SVE"
            elif self.has_neon:
                return "ARM NEON"
            else:
                return "None (scalar only)"
        else:
            # x86/x64 architecture
            if self.has_avx512f:
                variants = []
                if self.has_avx512dq:
                    variants.append("DQ")
                if self.has_avx512bw:
                    variants.append("BW")
                if self.has_avx512vl:
                    variants.append("VL")
                return f"AVX-512 ({', '.join(variants) if variants else 'F only'})"
            elif self.has_avx2:
                return "AVX2"
            elif self.has_avx:
                return "AVX"
            elif self.has_sse4_2:
                return "SSE4.2"
            elif self.has_sse4_1:
                return "SSE4.1"
            elif self.has_ssse3:
                return "SSSE3"
            elif self.has_sse3:
                return "SSE3"
            elif self.has_sse2:
                return "SSE2"
            elif self.has_sse:
                return "SSE"
            else:
                return "None (scalar only)"

    def get_performance_tier(self) -> str:
        """Categorize CPU performance tier based on capabilities"""
        best = self.get_best_vector_instruction_set()

        if "AVX-512" in best or "SVE2" in best:
            return "High Performance"
        elif "AVX2" in best or "SVE" in best or "NEON" in best:
            return "Medium Performance"
        elif "AVX" in best or "SSE4" in best:
            return "Standard Performance"
        else:
            return "Basic Performance"

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/display"""
        return {
            "architecture": self.architecture,
            "platform": self.platform,
            "processor": self.processor,
            "best_instruction_set": self.get_best_vector_instruction_set(),
            "performance_tier": self.get_performance_tier(),
            "cpu_cores": self.cpu_cores,
            "x86_features": {
                "SSE": self.has_sse,
                "SSE2": self.has_sse2,
                "SSE3": self.has_sse3,
                "SSSE3": self.has_ssse3,
                "SSE4.1": self.has_sse4_1,
                "SSE4.2": self.has_sse4_2,
                "AVX": self.has_avx,
                "AVX2": self.has_avx2,
                "AVX-512F": self.has_avx512f,
                "AVX-512DQ": self.has_avx512dq,
                "AVX-512BW": self.has_avx512bw,
                "AVX-512VL": self.has_avx512vl,
                "FMA": self.has_fma,
            },
            "arm_features": {
                "NEON": self.has_neon,
                "SVE": self.has_sve,
                "SVE2": self.has_sve2,
            },
        }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"CPUCapabilities(arch={self.architecture!r}, "
            f"best={self.get_best_vector_instruction_set()!r}, "
            f"tier={self.get_performance_tier()!r})"
        )


def detect_cpu_capabilities() -> CPUCapabilities:
    """
    Detect CPU capabilities across platforms.

    Returns:
        CPUCapabilities object with detected features
    """
    caps = CPUCapabilities()

    try:
        import multiprocessing

        caps.cpu_cores = multiprocessing.cpu_count()
    except (ImportError, NotImplementedError, OSError):
        caps.cpu_cores = 1

    # Detect platform and architecture
    caps.platform = platform.system()
    caps.architecture = platform.machine()
    caps.processor = platform.processor() or "Unknown"

    try:
        if caps.platform == "Linux":
            _detect_linux_capabilities(caps)
        elif caps.platform == "Darwin":
            _detect_macos_capabilities(caps)
        elif caps.platform == "Windows":
            _detect_windows_capabilities(caps)
        else:
            logger.debug(f"Unknown platform: {caps.platform}")
    except Exception as e:
        logger.warning(f"Error detecting CPU capabilities: {e}")

    return caps


def _detect_linux_capabilities(caps: CPUCapabilities) -> None:
    """Detect capabilities on Linux by reading /proc/cpuinfo"""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()

        # Extract CPU flags
        for line in cpuinfo.split("\n"):
            if line.startswith("flags") or line.startswith("features"):
                flags_str = line.split(":", 1)[1].strip()
                caps.cpu_flags = set(flags_str.split())
                break

        # Check x86/x64 features
        caps.has_sse = "sse" in caps.cpu_flags
        caps.has_sse2 = "sse2" in caps.cpu_flags
        caps.has_sse3 = "sse3" in caps.cpu_flags or "pni" in caps.cpu_flags
        caps.has_ssse3 = "ssse3" in caps.cpu_flags
        caps.has_sse4_1 = "sse4_1" in caps.cpu_flags
        caps.has_sse4_2 = "sse4_2" in caps.cpu_flags
        caps.has_avx = "avx" in caps.cpu_flags
        caps.has_avx2 = "avx2" in caps.cpu_flags
        caps.has_avx512f = "avx512f" in caps.cpu_flags
        caps.has_avx512dq = "avx512dq" in caps.cpu_flags
        caps.has_avx512bw = "avx512bw" in caps.cpu_flags
        caps.has_avx512vl = "avx512vl" in caps.cpu_flags
        caps.has_fma = "fma" in caps.cpu_flags

        # Check ARM features
        caps.has_neon = "neon" in caps.cpu_flags or "asimd" in caps.cpu_flags
        caps.has_sve = "sve" in caps.cpu_flags
        caps.has_sve2 = "sve2" in caps.cpu_flags

    except (IOError, OSError) as e:
        logger.debug(f"Could not read /proc/cpuinfo: {e}")


def _detect_macos_capabilities(caps: CPUCapabilities) -> None:
    """Detect capabilities on macOS using sysctl"""
    try:
        import subprocess

        # On macOS, we can use sysctl to get CPU features
        result = subprocess.run(
            ["sysctl", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode == 0:
            # Parse sysctl output properly to avoid including key names
            flags = set()
            for line in result.stdout.lower().split("\n"):
                if ":" in line:
                    _, _, value = line.partition(":")
                    flags.update(value.strip().split())
            caps.cpu_flags = flags

            # Check common features
            caps.has_sse = "sse" in caps.cpu_flags
            caps.has_sse2 = "sse2" in caps.cpu_flags
            caps.has_sse3 = "sse3" in caps.cpu_flags
            caps.has_ssse3 = "ssse3" in caps.cpu_flags
            caps.has_sse4_1 = "sse4.1" in caps.cpu_flags or "sse4_1" in caps.cpu_flags
            caps.has_sse4_2 = "sse4.2" in caps.cpu_flags or "sse4_2" in caps.cpu_flags
            caps.has_avx = "avx" in caps.cpu_flags
            caps.has_avx2 = "avx2" in caps.cpu_flags
            caps.has_avx512f = "avx512f" in caps.cpu_flags
            caps.has_fma = "fma" in caps.cpu_flags

            # ARM features on Apple Silicon
            if (
                "arm" in caps.architecture.lower()
                or "aarch" in caps.architecture.lower()
            ):
                # Apple Silicon always has NEON
                caps.has_neon = True
    except Exception as e:
        logger.debug(f"Could not detect macOS capabilities: {e}")

        # Apple Silicon fallback
        if "arm" in caps.architecture.lower() or "aarch" in caps.architecture.lower():
            caps.has_neon = True


def _detect_windows_capabilities(caps: CPUCapabilities) -> None:
    """Detect capabilities on Windows"""
    # Try to use py-cpuinfo if available for better detection
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        flags = set(f.lower() for f in info.get("flags", []))
        caps.cpu_flags = flags

        # Check x86/x64 features
        caps.has_sse = "sse" in flags
        caps.has_sse2 = "sse2" in flags
        caps.has_sse3 = "sse3" in flags or "pni" in flags
        caps.has_ssse3 = "ssse3" in flags
        caps.has_sse4_1 = "sse4_1" in flags
        caps.has_sse4_2 = "sse4_2" in flags
        caps.has_avx = "avx" in flags
        caps.has_avx2 = "avx2" in flags
        caps.has_avx512f = "avx512f" in flags
        caps.has_avx512dq = "avx512dq" in flags
        caps.has_avx512bw = "avx512bw" in flags
        caps.has_avx512vl = "avx512vl" in flags
        caps.has_fma = "fma" in flags

        logger.debug("Windows CPU detection: using py-cpuinfo")
        return
    except ImportError:
        # Fall back to conservative defaults
        pass

    # Conservative defaults when py-cpuinfo is not available
    if "amd64" in caps.architecture.lower() or "x86_64" in caps.architecture.lower():
        # Modern x64 CPUs typically have at least SSE2
        caps.has_sse = True
        caps.has_sse2 = True
        # Most modern x64 CPUs have SSE4.2
        caps.has_sse3 = True
        caps.has_ssse3 = True
        caps.has_sse4_1 = True
        caps.has_sse4_2 = True

    logger.debug("Windows CPU detection: using conservative defaults")


def format_capability_warning(caps: CPUCapabilities, component: str = "FAISS") -> str:
    """
    Format a detailed capability warning message.

    Args:
        caps: CPU capabilities
        component: Component name (e.g., "FAISS", "NumPy")

    Returns:
        Formatted warning message
    """
    best_instr = caps.get_best_vector_instruction_set()
    perf_tier = caps.get_performance_tier()

    if caps.architecture.lower().startswith(
        "arm"
    ) or caps.architecture.lower().startswith("aarch"):
        if not caps.has_sve and not caps.has_sve2:
            return (
                f"{component} loaded with ARM NEON "
                f"(SVE/SVE2 unavailable, performance tier: {perf_tier})"
            )
        else:
            return f"{component} using {best_instr} (performance tier: {perf_tier})"
    else:
        # x86/x64
        if not caps.has_avx512f:
            if caps.has_avx2:
                return (
                    f"{component} loaded with AVX2 "
                    f"(AVX512 unavailable, performance tier: {perf_tier})"
                )
            elif caps.has_avx:
                return (
                    f"{component} loaded with AVX "
                    f"(AVX2/AVX512 unavailable, performance tier: {perf_tier})"
                )
            else:
                return (
                    f"{component} loaded with {best_instr} "
                    f"(modern vector instructions unavailable, performance tier: {perf_tier})"
                )
        else:
            return f"{component} using {best_instr} (performance tier: {perf_tier})"


def get_capability_summary() -> str:
    """Get a human-readable summary of CPU capabilities"""
    caps = detect_cpu_capabilities()

    lines = [
        "CPU Capability Summary:",
        f"  Platform: {caps.platform}",
        f"  Architecture: {caps.architecture}",
        f"  Processor: {caps.processor}",
        f"  Cores: {caps.cpu_cores}",
        f"  Best Instruction Set: {caps.get_best_vector_instruction_set()}",
        f"  Performance Tier: {caps.get_performance_tier()}",
    ]

    return "\n".join(lines)


# Global instance for caching
_cpu_capabilities: Optional[CPUCapabilities] = None
_cpu_capabilities_lock = threading.Lock()


def get_cpu_capabilities() -> CPUCapabilities:
    """Get cached CPU capabilities (singleton pattern with thread-safety)"""
    global _cpu_capabilities
    if _cpu_capabilities is None:
        with _cpu_capabilities_lock:
            # Double-check pattern to avoid race conditions
            if _cpu_capabilities is None:
                _cpu_capabilities = detect_cpu_capabilities()
    return _cpu_capabilities
