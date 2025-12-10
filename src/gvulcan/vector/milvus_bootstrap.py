"""
Milvus Collection Bootstrap and Management

This module provides comprehensive Milvus collection initialization with support for
multi-tier configurations, index management, validation, and monitoring.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    MilvusException,
)
import logging
import time

logger = logging.getLogger(__name__)


class BootstrapError(Exception):
    """Base exception for bootstrap errors"""

    pass


class ConfigurationError(BootstrapError):
    """Raised when configuration is invalid"""

    pass


class CollectionCreationError(BootstrapError):
    """Raised when collection creation fails"""

    pass


def _field(name: str, dtype: DataType, **kwargs) -> FieldSchema:
    """
    Create a field schema.

    Args:
        name: Field name
        dtype: Data type
        **kwargs: Additional field parameters

    Returns:
        FieldSchema instance
    """
    return FieldSchema(name=name, dtype=dtype, **kwargs)


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate configuration structure.

    Args:
        cfg: Configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_keys = ["defaults", "tiers"]

    for key in required_keys:
        if key not in cfg:
            raise ConfigurationError(f"Missing required key: {key}")

    defaults = cfg["defaults"]
    required_defaults = ["pk_field", "vec_field", "metric_type", "dim"]

    for key in required_defaults:
        if key not in defaults:
            raise ConfigurationError(f"Missing required default: {key}")

    if not isinstance(cfg["tiers"], list):
        raise ConfigurationError("'tiers' must be a list")

    if not cfg["tiers"]:
        raise ConfigurationError("At least one tier must be defined")

    # Validate dimension
    dim = defaults["dim"]
    if not isinstance(dim, int) or dim <= 0:
        raise ConfigurationError(f"Invalid dimension: {dim}")

    # Validate tiers
    for tier in cfg["tiers"]:
        required_tier_keys = ["collection", "index", "description"]
        for key in required_tier_keys:
            if key not in tier:
                raise ConfigurationError(f"Tier missing required key: {key}")

        # Validate index config
        index_config = tier["index"]
        if "type" not in index_config:
            raise ConfigurationError("Index config missing 'type'")

    logger.info("Configuration validation passed")


def create_collection_if_not_exists(
    cfg: Dict[str, Any], tier: Dict[str, Any], dim: int, drop_existing: bool = False
) -> Collection:
    """
    Create collection if it doesn't exist, with comprehensive error handling.

    Args:
        cfg: Full configuration dictionary
        tier: Tier-specific configuration
        dim: Vector dimensionality
        drop_existing: Whether to drop existing collection

    Returns:
        Collection instance

    Raises:
        CollectionCreationError: If collection creation fails
    """
    coll_name = tier["collection"]

    try:
        # Check if collection exists
        if utility.has_collection(coll_name):
            if drop_existing:
                logger.warning(f"Dropping existing collection: {coll_name}")
                utility.drop_collection(coll_name)
            else:
                logger.info(f"Collection already exists: {coll_name}")
                return Collection(coll_name)

        # Get field names
        pk = cfg["defaults"]["pk_field"]
        vec = cfg["defaults"]["vec_field"]

        # Build field schemas
        fields = [
            _field(pk, DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            _field(vec, DataType.FLOAT_VECTOR, dim=dim),
        ]

        # Add extra fields
        for f in cfg["defaults"].get("extra_fields", []):
            t = f["type"]

            if t == "INT64":
                fields.append(_field(f["name"], DataType.INT64))
            elif t == "FLOAT":
                fields.append(_field(f["name"], DataType.FLOAT))
            elif t == "DOUBLE":
                fields.append(_field(f["name"], DataType.DOUBLE))
            elif t == "VARCHAR":
                max_len = f.get("max_length", 128)
                fields.append(_field(f["name"], DataType.VARCHAR, max_length=max_len))
            elif t == "BOOL":
                fields.append(_field(f["name"], DataType.BOOL))
            elif t == "JSON":
                fields.append(_field(f["name"], DataType.JSON))
            else:
                logger.warning(
                    f"Unsupported field type: {t}, skipping field {f['name']}"
                )
                continue

        # Create schema
        description = tier.get(
            "description", f"Collection for tier {tier.get('name', 'unknown')}"
        )
        schema = CollectionSchema(
            fields=fields,
            description=description,
            enable_dynamic_field=cfg["defaults"].get("enable_dynamic_fields", False),
        )

        # Create collection
        logger.info(f"Creating collection: {coll_name}")
        coll = Collection(name=coll_name, schema=schema)

        logger.info(f"Successfully created collection: {coll_name}")

        # Create index
        create_index_for_collection(cfg, tier, coll, vec)

        return coll

    except MilvusException as e:
        raise CollectionCreationError(
            f"Failed to create collection {coll_name}: {e}"
        ) from e
    except Exception as e:
        raise CollectionCreationError(
            f"Unexpected error creating collection {coll_name}: {e}"
        ) from e


def create_index_for_collection(
    cfg: Dict[str, Any], tier: Dict[str, Any], coll: Collection, vec_field: str
) -> None:
    """
    Create index for collection vector field.

    Args:
        cfg: Full configuration
        tier: Tier configuration
        coll: Collection instance
        vec_field: Name of vector field
    """
    index_config = tier["index"]
    index_type = index_config["type"]
    metric_type = cfg["defaults"]["metric_type"]
    index_params = index_config.get("params", {})

    # Validate index parameters based on type
    if index_type == "HNSW":
        if "M" not in index_params:
            index_params["M"] = 16
        if "efConstruction" not in index_params:
            index_params["efConstruction"] = 256
    elif index_type == "IVF_FLAT":
        if "nlist" not in index_params:
            index_params["nlist"] = 1024
    elif index_type == "IVF_PQ":
        if "nlist" not in index_params:
            index_params["nlist"] = 1024
        if "m" not in index_params:
            index_params["m"] = 8
        if "nbits" not in index_params:
            index_params["nbits"] = 8

    index_params_full = {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": index_params,
    }

    try:
        logger.info(f"Creating index on {vec_field}: {index_params_full}")
        coll.create_index(field_name=vec_field, index_params=index_params_full)
        logger.info(f"Successfully created index on {vec_field}")
    except MilvusException as e:
        logger.error(f"Failed to create index: {e}")
        raise


def load_collection(coll_name: str, timeout: float = 30.0) -> Collection:
    """
    Load collection into memory.

    Args:
        coll_name: Collection name
        timeout: Load timeout in seconds

    Returns:
        Loaded collection
    """
    try:
        coll = Collection(coll_name)

        logger.info(f"Loading collection: {coll_name}")
        coll.load()

        # Wait for load to complete
        start_time = time.time()
        while time.time() - start_time < timeout:
            if (
                coll.num_entities > 0
                or utility.get_loading_progress(coll_name).get("progress", 0) == 100
            ):
                logger.info(
                    f"Collection loaded: {coll_name} ({coll.num_entities} entities)"
                )
                return coll
            time.sleep(0.5)

        logger.warning(f"Collection load may not be complete: {coll_name}")
        return coll

    except Exception as e:
        logger.error(f"Failed to load collection {coll_name}: {e}")
        raise


def get_collection_stats(coll_name: str) -> Dict[str, Any]:
    """
    Get statistics for a collection.

    Args:
        coll_name: Collection name

    Returns:
        Dictionary with collection statistics
    """
    try:
        coll = Collection(coll_name)

        stats = {
            "name": coll_name,
            "num_entities": coll.num_entities,
            "schema": {
                "description": coll.description,
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.dtype),
                        "params": field.params,
                    }
                    for field in coll.schema.fields
                ],
            },
        }

        # Get index info
        indexes = coll.indexes
        if indexes:
            stats["indexes"] = [
                {"field": idx.field_name, "params": idx.params} for idx in indexes
            ]

        return stats

    except Exception as e:
        logger.error(f"Failed to get stats for {coll_name}: {e}")
        return {"name": coll_name, "error": str(e)}


def bootstrap_all_collections(
    cfg: Dict[str, Any], drop_existing: bool = False, load_after_creation: bool = True
) -> Dict[str, Collection]:
    """
    Bootstrap all configured collections.

    Args:
        cfg: Configuration dictionary
        drop_existing: Whether to drop existing collections
        load_after_creation: Whether to load collections after creation

    Returns:
        Dictionary mapping collection names to Collection instances
    """
    validate_config(cfg)

    dim = cfg["defaults"]["dim"]
    collections = {}

    for tier in cfg["tiers"]:
        coll_name = tier["collection"]

        try:
            coll = create_collection_if_not_exists(cfg, tier, dim, drop_existing)

            if load_after_creation:
                coll = load_collection(coll_name)

            collections[coll_name] = coll

            logger.info(f"Successfully bootstrapped collection: {coll_name}")

        except Exception as e:
            logger.error(f"Failed to bootstrap collection {coll_name}: {e}")
            if not cfg["defaults"].get("continue_on_error", True):
                raise

    return collections


def main(
    config_path: Optional[Path] = None,
    host: str = "milvus",
    port: int = 19530,
    drop_existing: bool = False,
) -> None:
    """
    Main bootstrap function.

    Args:
        config_path: Path to configuration file
        host: Milvus host
        port: Milvus port
        drop_existing: Whether to drop existing collections
    """
    logger.info("Starting Milvus bootstrap")

    # Determine config path
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "vector"
            / "milvus"
            / "collections.yaml"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    # Connect to Milvus
    logger.info(f"Connecting to Milvus: {host}:{port}")
    connections.connect(alias="default", host=host, port=str(port))

    try:
        # Bootstrap collections
        collections = bootstrap_all_collections(cfg, drop_existing=drop_existing)

        # Print summary
        logger.info("=" * 60)
        logger.info("Bootstrap Summary")
        logger.info("=" * 60)

        for coll_name, coll in collections.items():
            stats = get_collection_stats(coll_name)
            logger.info(f"\nCollection: {coll_name}")
            logger.info(f"  Entities: {stats.get('num_entities', 0)}")
            logger.info(f"  Fields: {len(stats.get('schema', {}).get('fields', []))}")

            if "indexes" in stats:
                logger.info(f"  Indexes: {len(stats['indexes'])}")

        logger.info("=" * 60)
        logger.info("Bootstrap complete!")

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}", exc_info=True)
        raise

    finally:
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Bootstrap Milvus collections")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--host", default="milvus", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument(
        "--drop-existing", action="store_true", help="Drop existing collections"
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        host=args.host,
        port=args.port,
        drop_existing=args.drop_existing,
    )
