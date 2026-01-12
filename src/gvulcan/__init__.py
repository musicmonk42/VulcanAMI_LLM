from . import merkle, zk, dqs, bloom, opa, crc32c, config
from pathlib import Path

# Export key classes for convenience
from .dqs import DQSScorer, DQSTracker, DQSComponents, DQSResult, compute_dqs
from .bloom import BloomFilter, CountingBloomFilter, ScalableBloomFilter
from .opa import OPAClient, WriteBarrierInput, WriteBarrierResult, PolicyRegistry
from .merkle import MerkleTree, MerkleLSMDAG, MerkleProof
from .config import GVulcanConfig, ConfigurationManager, get_config

__all__ = [
    "__version__",
    # Modules
    "zk", "merkle", "dqs", "bloom", "opa", "crc32c", "config",
    # DQS classes
    "DQSScorer", "DQSTracker", "DQSComponents", "DQSResult", "compute_dqs",
    # Bloom filter classes
    "BloomFilter", "CountingBloomFilter", "ScalableBloomFilter",
    # OPA classes
    "OPAClient", "WriteBarrierInput", "WriteBarrierResult", "PolicyRegistry",
    # Merkle classes
    "MerkleTree", "MerkleLSMDAG", "MerkleProof",
    # Config classes
    "GVulcanConfig", "ConfigurationManager", "get_config",
]


def _read_semver() -> str:
    try:
        p = Path(__file__).resolve().parents[2] / "configs" / "packer" / "semver.txt"
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"


__version__ = _read_semver()

# Expose ZK module for easy access
