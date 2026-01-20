#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN-AGI Package Setup Configuration.

This module configures VULCAN as a proper Python package that can be installed
with 'pip install -e .' to enable proper module discovery and imports.

Key Features:
    - Enables 'import src' to work in any environment
    - Enables 'from src.vulcan.orchestrator.agent_pool import AgentPool'
    - Supports GitHub Actions stress tests with real VULCAN components
    - Provides console script entry points for CLI usage

Installation:
    Development mode (editable):
        pip install -e .

    With test dependencies:
        pip install -e ".[test]"

    With dev dependencies:
        pip install -e ".[dev]"

Example:
    >>> import src
    >>> from src.vulcan.orchestrator.agent_pool import AgentPoolManager
    >>> from src.vulcan.reasoning.selection.tool_selector import ToolSelector

Author: VULCAN-AGI Team
License: MIT
Version: 2.0.0
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from setuptools import find_packages, setup

# Configure logging for setup process
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MIN_PYTHON_VERSION = (3, 11)
DEFAULT_VERSION = "2.0.0"
PACKAGE_NAME = "vulcan-agi"


def check_python_version() -> None:
    """
    Verify Python version meets minimum requirements.

    Raises:
        SystemExit: If Python version is below minimum required.
    """
    if sys.version_info < MIN_PYTHON_VERSION:
        logger.error(
            f"VULCAN requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+, "
            f"but you're using Python {sys.version_info.major}.{sys.version_info.minor}"
        )
        sys.exit(1)


def get_version() -> str:
    """
    Get package version from src/vulcan/__version__.py or use default.

    Returns:
        str: The version string (e.g., "2.0.0").

    Note:
        Falls back to DEFAULT_VERSION if version file doesn't exist.
    """
    version_file = Path(__file__).parent / "src" / "vulcan" / "__version__.py"

    if version_file.exists():
        version_dict: dict = {}
        try:
            with open(version_file, encoding="utf-8") as f:
                exec(f.read(), version_dict)  # noqa: S102
            return version_dict.get("__version__", DEFAULT_VERSION)
        except (OSError, SyntaxError) as e:
            logger.warning(f"Failed to read version file: {e}")

    return DEFAULT_VERSION


def get_requirements() -> List[str]:
    """
    Read requirements from requirements.txt or provide fallback list.

    Returns:
        List[str]: List of package requirements with version specifiers.

    Note:
        Falls back to minimal dependency list if requirements.txt is missing
        or cannot be read.
    """
    requirements_file = Path(__file__).parent / "requirements.txt"

    if requirements_file.exists():
        try:
            with open(requirements_file, encoding="utf-8") as f:
                requirements: List[str] = []
                for line in f:
                    line = line.strip()
                    # Skip empty lines, comments, and pip options
                    if (
                        line
                        and not line.startswith("#")
                        and not line.startswith("-")
                        and not line.startswith("git+")
                    ):
                        # Remove inline comments and whitespace
                        requirement = line.split("#")[0].strip()
                        if requirement:
                            requirements.append(requirement)
                return requirements
        except OSError as e:
            logger.warning(f"Failed to read requirements.txt: {e}")

    # Fallback minimal dependencies
    logger.info("Using fallback dependency list")
    return [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "openai>=1.0.0",
        "redis>=4.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "llvmlite",
        "faker",
    ]


def get_long_description() -> str:
    """
    Read long description from README.md.

    Returns:
        str: Contents of README.md or fallback description.
    """
    readme_file = Path(__file__).parent / "README.md"

    if readme_file.exists():
        try:
            with open(readme_file, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to read README.md: {e}")

    return (
        "VULCAN-AGI: Volumetric Unified Learning Cognitive Architecture Network\n\n"
        "A production-grade AGI system integrating multiple cognitive architectures."
    )


def get_package_data() -> dict:
    """
    Get package data configuration for non-Python files.

    Returns:
        dict: Package data configuration mapping.
    """
    return {
        "src": [
            "vulcan/configs/*.json",
            "vulcan/configs/*.yaml",
            "vulcan/configs/*.yml",
            "vulcan/data/*.json",
            "vulcan/gvulcan/prompts/*.txt",
            "vulcan/gvulcan/prompts/*.md",
            "vulcan/gvulcan/skills/*.json",
        ],
    }


def get_extras_require() -> dict:
    """
    Get optional dependency groups.

    Returns:
        dict: Mapping of extra names to dependency lists.
    """
    return {
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-timeout>=2.2.0",
            "pytest-benchmark>=4.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
            "bandit>=1.7.0",
            "locust>=2.38.1",
            "pre-commit>=3.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-timeout>=2.2.0",
            "pytest-benchmark>=4.0.0",
            "pytest-cov>=4.0.0",
            "requests>=2.28.0",
            "aiohttp>=3.8.0",
            "psutil>=5.9.0",
            "httpx>=0.24.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    }


def get_entry_points() -> dict:
    """
    Get console script entry points.

    Returns:
        dict: Entry points configuration.
    """
    return {
        "console_scripts": [
            "vulcan=src.vulcan.main:main",
            "vulcan-verify=scripts.verify_installation:main",
        ],
    }


def get_classifiers() -> List[str]:
    """
    Get PyPI classifiers.

    Returns:
        List[str]: List of classifier strings.
    """
    return [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ]


# Verify Python version before setup
check_python_version()

# Run setup
setup(
    # Package metadata
    name=PACKAGE_NAME,
    version=get_version(),
    author="VULCAN-AGI Team",
    author_email="team@vulcan-agi.dev",
    maintainer="VULCAN-AGI Team",
    maintainer_email="team@vulcan-agi.dev",
    description="VULCAN-AGI: Volumetric Unified Learning Cognitive Architecture Network",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/musicmonk42/VulcanAMI_LLM",
    project_urls={
        "Bug Reports": "https://github.com/musicmonk42/VulcanAMI_LLM/issues",
        "Source": "https://github.com/musicmonk42/VulcanAMI_LLM",
        "Documentation": "https://github.com/musicmonk42/VulcanAMI_LLM#readme",
    },
    license="MIT",
    # Package discovery - CRITICAL for 'import src' to work
    packages=find_packages(include=["src", "src.*", "scripts"]),
    package_dir={"": "."},  # Map packages to project root
    # Include root-level modules
    py_modules=["graph_compiler", "load_test", "app", "graphix_vulcan_llm"],
    # Package data
    package_data=get_package_data(),
    include_package_data=True,
    # Dependencies
    install_requires=get_requirements(),
    extras_require=get_extras_require(),
    # Entry points
    entry_points=get_entry_points(),
    # Python version
    python_requires=f">={MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}",
    # Classifiers
    classifiers=get_classifiers(),
    # Keywords
    keywords=[
        "agi",
        "artificial-intelligence",
        "cognitive-architecture",
        "vulcan",
        "machine-learning",
        "deep-learning",
        "reasoning",
        "llm",
    ],
    # Additional options
    zip_safe=False,
    platforms=["any"],
)
