#!/usr/bin/env python3
"""
Bootstrap Knowledge - Seed the Curiosity Engine with Baseline Knowledge
========================================================================

This script initializes the KnowledgeCrystallizer and seeds the system with
baseline knowledge from a text file to address the "0 knowledge gaps" issue
where the knowledge graph is empty.

Part of the VULCAN-AGI system - Phase 4: Knowledge Seeding

Usage:
    # Basic usage with text file
    python scripts/bootstrap_knowledge.py --file path/to/knowledge.txt

    # With custom chunk size
    python scripts/bootstrap_knowledge.py --file corpus.txt --chunk-size 1000

    # Specify knowledge domain
    python scripts/bootstrap_knowledge.py --file science.txt --domain science

    # Dry run to validate file without crystallizing
    python scripts/bootstrap_knowledge.py --file corpus.txt --dry-run

    # With custom tokenizer vocabulary
    python scripts/bootstrap_knowledge.py --file corpus.txt --vocab-path checkpoints/vocab.json

Cron Example:
    # Seed knowledge daily at 3 AM
    0 3 * * * cd /path/to/VulcanAMI_LLM && python scripts/bootstrap_knowledge.py --file data/knowledge_corpus.txt

Features:
    1. Initializes the KnowledgeCrystallizer component
    2. Accepts a text file path as an argument
    3. Reads the text file and feeds it into crystallizer.crystallize_experience()
    4. Supports optional tokenizer integration for enhanced text processing
    5. Provides comprehensive statistics and error handling
    6. Thread-safe execution with proper resource cleanup

Integrations:
    - KnowledgeCrystallizer: Core knowledge extraction and storage
    - SimpleTokenizer: Optional vocabulary-based text tokenization
    - ExecutionTrace: Standard trace format for knowledge crystallization

Output:
    - Console output with progress and statistics
    - Exit code 0 on success, 1 on failure
    - Detailed error logging for troubleshooting
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# =============================================================================
# PATH CONFIGURATION - Must be first before other imports
# =============================================================================

# Get project root and add to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_src_root = _project_root / "src"

# Add paths for imports
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bootstrap_knowledge.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CHECKS WITH GRACEFUL DEGRADATION
# =============================================================================

# Knowledge Crystallizer availability
CRYSTALLIZER_AVAILABLE = False
try:
    from vulcan.knowledge_crystallizer.knowledge_crystallizer_core import (
        KnowledgeCrystallizer,
        ExecutionTrace,
        CrystallizationResult,
    )
    CRYSTALLIZER_AVAILABLE = True
    logger.debug("KnowledgeCrystallizer module loaded successfully")
except ImportError as e:
    logger.warning(f"KnowledgeCrystallizer not available: {e}")
    KnowledgeCrystallizer = None
    ExecutionTrace = None
    CrystallizationResult = None

# SimpleTokenizer availability
TOKENIZER_AVAILABLE = False
try:
    from local_llm.tokenizer.simple_tokenizer import SimpleTokenizer
    TOKENIZER_AVAILABLE = True
    logger.debug("SimpleTokenizer module loaded successfully")
except ImportError as e:
    logger.debug(f"SimpleTokenizer not available: {e}")
    SimpleTokenizer = None

# =============================================================================
# CONSTANTS
# =============================================================================

# Default configuration
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_DOMAIN = "general"
MAX_CHUNK_SIZE = 10000
MIN_CHUNK_SIZE = 50

# Content truncation for action logs
ACTION_CONTENT_TRUNCATE_LENGTH = 500

# Sentence delimiters for intelligent chunking
SENTENCE_DELIMITERS = (". ", ".\n", "! ", "? ", "\n\n", ";\n")

# Signal handling
SHUTDOWN_REQUESTED = threading.Event()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BootstrapConfig:
    """Configuration for knowledge bootstrap process."""

    file_path: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    domain: str = DEFAULT_DOMAIN
    vocab_path: Optional[Path] = None
    dry_run: bool = False
    verbose: bool = False
    max_chunks: Optional[int] = None  # Limit for testing

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate chunk size
        if self.chunk_size < MIN_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be >= {MIN_CHUNK_SIZE}, got {self.chunk_size}"
            )
        if self.chunk_size > MAX_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be <= {MAX_CHUNK_SIZE}, got {self.chunk_size}"
            )

        # Ensure file_path is a Path object
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        # Ensure vocab_path is a Path object if provided
        if self.vocab_path and isinstance(self.vocab_path, str):
            self.vocab_path = Path(self.vocab_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            "file_path": str(self.file_path),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "domain": self.domain,
            "vocab_path": str(self.vocab_path) if self.vocab_path else None,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
            "max_chunks": self.max_chunks,
        }


@dataclass
class BootstrapStats:
    """Statistics from knowledge bootstrap process."""

    file_path: str = ""
    file_size_bytes: int = 0
    total_words: int = 0
    total_chunks: int = 0
    chunks_processed: int = 0
    principles_extracted: int = 0
    validation_successes: int = 0
    validation_failures: int = 0
    avg_confidence: float = 0.0
    execution_time_seconds: float = 0.0
    tokenizer_used: Optional[str] = None
    vocab_size: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "total_words": self.total_words,
            "total_chunks": self.total_chunks,
            "chunks_processed": self.chunks_processed,
            "principles_extracted": self.principles_extracted,
            "validation_successes": self.validation_successes,
            "validation_failures": self.validation_failures,
            "avg_confidence": self.avg_confidence,
            "execution_time_seconds": self.execution_time_seconds,
            "tokenizer_used": self.tokenizer_used,
            "vocab_size": self.vocab_size,
            "errors": self.errors,
            "success": self.success,
        }


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    signal_name = signal.Signals(signum).name
    logger.warning(f"Received {signal_name} signal - initiating graceful shutdown...")
    SHUTDOWN_REQUESTED.set()


def _setup_signal_handlers():
    """Configure signal handlers for graceful shutdown."""
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        logger.debug("Signal handlers configured")
    except Exception as e:
        logger.warning(f"Could not setup signal handlers: {e}")


# =============================================================================
# TOKENIZER LOADER
# =============================================================================


class TokenizerLoader:
    """Handles tokenizer loading with caching and error handling."""

    _instance: Optional["TokenizerLoader"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._tokenizer = None
        self._vocab_path: Optional[Path] = None

    @classmethod
    def get_instance(cls) -> "TokenizerLoader":
        """Get singleton instance of TokenizerLoader."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load(self, vocab_path: Optional[Path] = None) -> Optional[Any]:
        """
        Load a tokenizer if explicitly requested.

        Args:
            vocab_path: Path to vocabulary file (required for loading)

        Returns:
            Tokenizer instance or None if not available/not requested
        """
        # Only load if vocab_path is explicitly provided
        if vocab_path is None:
            logger.debug("No vocab_path provided - skipping tokenizer loading")
            return None

        # Return cached tokenizer if same path
        if self._tokenizer is not None and self._vocab_path == vocab_path:
            return self._tokenizer

        if not TOKENIZER_AVAILABLE or SimpleTokenizer is None:
            logger.warning("SimpleTokenizer not available in environment")
            return None

        if not vocab_path.exists():
            logger.warning(f"Vocabulary file not found: {vocab_path}")
            return None

        try:
            logger.info(f"Loading tokenizer from: {vocab_path}")
            start_time = time.time()
            self._tokenizer = SimpleTokenizer(str(vocab_path))
            self._vocab_path = vocab_path
            load_time = time.time() - start_time
            logger.info(
                f"✓ Tokenizer loaded in {load_time:.2f}s "
                f"(vocab_size={len(self._tokenizer.vocab)})"
            )
            return self._tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return None

    def get_vocab_size(self) -> Optional[int]:
        """Get vocabulary size if tokenizer is loaded."""
        if self._tokenizer and hasattr(self._tokenizer, "vocab"):
            return len(self._tokenizer.vocab)
        return None


# =============================================================================
# TEXT CHUNKER
# =============================================================================


class TextChunker:
    """Intelligent text chunking with sentence boundary detection."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks with intelligent boundary detection.

        The chunker attempts to break at sentence boundaries to preserve
        semantic coherence. Falls back to character boundaries if no
        suitable sentence end is found.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = []
        text_len = len(text)
        start = 0

        while start < text_len:
            # Check for shutdown request
            if SHUTDOWN_REQUESTED.is_set():
                logger.warning("Shutdown requested - stopping chunking")
                break

            # Calculate end position
            end = min(start + self.chunk_size, text_len)

            # Try to find sentence boundary if not at end of text
            if end < text_len:
                end = self._find_sentence_boundary(text, start, end)

            # Extract and clean chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # If we've processed the entire text, stop
            if end >= text_len:
                break

            # Move to next position with overlap
            # Ensure we always make forward progress
            next_start = end - self.overlap
            if next_start <= start:
                next_start = end  # No overlap if chunk is too small
            start = next_start

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the best sentence boundary within the chunk range.

        Searches for sentence-ending punctuation from the end of the range
        back to the midpoint, preferring longer chunks.

        Args:
            text: Full text
            start: Start position of current chunk
            end: Initial end position

        Returns:
            Adjusted end position at sentence boundary, or original end
        """
        chunk_text = text[start:end]
        min_position = self.chunk_size // 2

        # Search for sentence delimiters from end toward middle
        for delimiter in SENTENCE_DELIMITERS:
            last_pos = chunk_text.rfind(delimiter)
            if last_pos > min_position:
                return start + last_pos + len(delimiter)

        # No good boundary found - use original end
        return end


# =============================================================================
# KNOWLEDGE BOOTSTRAPPER
# =============================================================================


class KnowledgeBootstrapper:
    """
    Main class for bootstrapping knowledge into the VULCAN system.

    This class orchestrates the process of:
    1. Reading text from input files
    2. Chunking text into processable segments
    3. Creating execution traces for each chunk
    4. Crystallizing knowledge via KnowledgeCrystallizer
    5. Tracking statistics and errors
    """

    def __init__(self, config: BootstrapConfig):
        """
        Initialize knowledge bootstrapper.

        Args:
            config: Bootstrap configuration
        """
        self.config = config
        self.stats = BootstrapStats()
        self.crystallizer: Optional[KnowledgeCrystallizer] = None
        self.tokenizer_loader = TokenizerLoader.get_instance()
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        self._confidences: List[float] = []

    def initialize(self) -> bool:
        """
        Initialize components required for bootstrapping.

        Returns:
            True if initialization successful, False otherwise
        """
        if self.config.dry_run:
            logger.info("Dry run mode - skipping crystallizer initialization")
            return True

        if not CRYSTALLIZER_AVAILABLE:
            logger.error(
                "KnowledgeCrystallizer not available - cannot proceed with bootstrap. "
                "Ensure vulcan.knowledge_crystallizer module is installed."
            )
            self.stats.errors.append("KnowledgeCrystallizer not available")
            return False

        try:
            logger.info("Initializing KnowledgeCrystallizer...")
            self.crystallizer = KnowledgeCrystallizer()
            logger.info("✓ KnowledgeCrystallizer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeCrystallizer: {e}")
            self.stats.errors.append(f"Crystallizer init failed: {e}")
            return False

    def run(self) -> BootstrapStats:
        """
        Execute the knowledge bootstrap process.

        Returns:
            BootstrapStats with results and statistics
        """
        start_time = time.time()
        self.stats = BootstrapStats()
        self.stats.file_path = str(self.config.file_path)

        try:
            # Validate input file
            if not self._validate_input_file():
                return self.stats

            # Read and analyze content
            content = self._read_file()
            if content is None:
                return self.stats

            # Load tokenizer if requested
            self._load_tokenizer()

            # Chunk the content
            chunks = self._chunk_content(content)
            if not chunks:
                return self.stats

            # Process chunks
            if not self.config.dry_run:
                self._process_chunks(chunks)

            # Calculate final statistics
            self._finalize_stats(start_time)

            return self.stats

        except Exception as e:
            logger.error(f"Bootstrap failed with exception: {e}", exc_info=True)
            self.stats.errors.append(f"Fatal error: {e}")
            self.stats.execution_time_seconds = time.time() - start_time
            return self.stats

    def _validate_input_file(self) -> bool:
        """Validate the input file exists and is readable."""
        file_path = self.config.file_path

        if not file_path.exists():
            logger.error(f"Input file not found: {file_path}")
            self.stats.errors.append(f"File not found: {file_path}")
            return False

        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            self.stats.errors.append(f"Not a file: {file_path}")
            return False

        if not os.access(file_path, os.R_OK):
            logger.error(f"File not readable: {file_path}")
            self.stats.errors.append(f"File not readable: {file_path}")
            return False

        return True

    def _read_file(self) -> Optional[str]:
        """Read content from input file."""
        try:
            logger.info(f"Reading knowledge file: {self.config.file_path}")
            with open(self.config.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.stats.file_size_bytes = len(content.encode("utf-8"))
            self.stats.total_words = len(content.split())

            logger.info(
                f"✓ Loaded {self.stats.total_words:,} words "
                f"({self.stats.file_size_bytes:,} bytes)"
            )
            return content

        except UnicodeDecodeError as e:
            logger.error(f"File encoding error (expected UTF-8): {e}")
            self.stats.errors.append(f"Encoding error: {e}")
            return None
        except IOError as e:
            logger.error(f"Failed to read file: {e}")
            self.stats.errors.append(f"Read error: {e}")
            return None

    def _load_tokenizer(self) -> None:
        """Load tokenizer if vocabulary path is provided."""
        tokenizer = self.tokenizer_loader.load(self.config.vocab_path)
        if tokenizer:
            self.stats.tokenizer_used = type(tokenizer).__name__
            self.stats.vocab_size = self.tokenizer_loader.get_vocab_size()
        else:
            self.stats.tokenizer_used = None
            if self.config.vocab_path:
                logger.warning(
                    "Tokenizer requested but could not be loaded - "
                    "proceeding with raw text processing"
                )

    def _chunk_content(self, content: str) -> List[str]:
        """Chunk content into processable segments."""
        logger.info(
            f"Chunking content (chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap})..."
        )

        chunks = self.chunker.chunk(content)
        self.stats.total_chunks = len(chunks)

        # Apply max_chunks limit if specified
        if self.config.max_chunks and len(chunks) > self.config.max_chunks:
            logger.info(
                f"Limiting to {self.config.max_chunks} chunks "
                f"(total available: {len(chunks)})"
            )
            chunks = chunks[: self.config.max_chunks]
            self.stats.total_chunks = len(chunks)

        logger.info(f"✓ Created {len(chunks)} chunks for processing")
        return chunks

    def _process_chunks(self, chunks: List[str]) -> None:
        """Process all chunks through the crystallizer."""
        if not self.crystallizer:
            logger.error("Crystallizer not initialized - cannot process chunks")
            self.stats.errors.append("Crystallizer not initialized")
            return

        logger.info(f"Processing {len(chunks)} chunks...")
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Check for shutdown request
            if SHUTDOWN_REQUESTED.is_set():
                logger.warning(
                    f"Shutdown requested - processed {i}/{total_chunks} chunks"
                )
                break

            self._process_single_chunk(chunk, i, total_chunks)

    def _process_single_chunk(
        self,
        chunk: str,
        index: int,
        total: int,
    ) -> None:
        """
        Process a single chunk through the crystallizer.

        Args:
            chunk: Text chunk to process
            index: Index of this chunk (0-based)
            total: Total number of chunks
        """
        try:
            # Generate trace ID using SHA256 for better practice
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:8]
            trace_id = f"bootstrap_{index}_{chunk_hash}"

            # Create execution trace
            trace = self._create_execution_trace(chunk, trace_id, index)

            # Crystallize the experience
            result = self.crystallizer.crystallize_experience(trace)

            # Update statistics
            self.stats.chunks_processed += 1
            principles_count = len(result.principles)
            self.stats.principles_extracted += principles_count

            if result.validation_results:
                for vr in result.validation_results:
                    if vr.is_valid:
                        self.stats.validation_successes += 1
                    else:
                        self.stats.validation_failures += 1

            if result.confidence > 0:
                self._confidences.append(result.confidence)

            # Log progress
            if self.config.verbose or principles_count > 0:
                logger.info(
                    f"Chunk {index + 1}/{total}: "
                    f"extracted {principles_count} principles "
                    f"(confidence: {result.confidence:.2f})"
                )
            elif (index + 1) % 10 == 0:
                # Progress update every 10 chunks
                logger.info(
                    f"Progress: {index + 1}/{total} chunks processed "
                    f"({self.stats.principles_extracted} principles extracted)"
                )

        except Exception as e:
            error_msg = f"Chunk {index}: {e}"
            logger.warning(f"Error processing chunk: {error_msg}")
            self.stats.errors.append(error_msg)

    def _create_execution_trace(
        self,
        text: str,
        trace_id: str,
        chunk_index: int,
    ) -> ExecutionTrace:
        """
        Create an ExecutionTrace from text content.

        Args:
            text: Source text content
            trace_id: Unique trace identifier
            chunk_index: Index of this chunk

        Returns:
            ExecutionTrace instance
        """
        return ExecutionTrace(
            trace_id=trace_id,
            actions=[
                {
                    "type": "text_ingestion",
                    "content": text[:ACTION_CONTENT_TRUNCATE_LENGTH],
                    "full_length": len(text),
                }
            ],
            outcomes={
                "text_processed": True,
                "content_type": "knowledge_base",
                "word_count": len(text.split()),
            },
            context={
                "source": "bootstrap_knowledge",
                "domain": self.config.domain,
                "file_path": str(self.config.file_path),
            },
            success=True,
            metadata={
                "chunk_index": chunk_index,
                "chunk_size": len(text),
                "source_file": str(self.config.file_path),
                "domain": self.config.domain,
            },
            domain=self.config.domain,
        )

    def _finalize_stats(self, start_time: float) -> None:
        """Calculate final statistics."""
        self.stats.execution_time_seconds = time.time() - start_time

        # Calculate average confidence
        if self._confidences:
            self.stats.avg_confidence = sum(self._confidences) / len(self._confidences)

        # Determine success
        self.stats.success = (
            self.stats.chunks_processed > 0
            and len(self.stats.errors) == 0
        ) or self.config.dry_run

        # Log summary
        self._log_summary()

    def _log_summary(self) -> None:
        """Log final bootstrap summary."""
        logger.info("=" * 70)
        logger.info("KNOWLEDGE BOOTSTRAP SUMMARY")
        logger.info("=" * 70)
        logger.info(f"File: {self.stats.file_path}")
        logger.info(f"Size: {self.stats.file_size_bytes:,} bytes")
        logger.info(f"Words: {self.stats.total_words:,}")
        logger.info("-" * 70)
        logger.info(
            f"Chunks: {self.stats.chunks_processed}/{self.stats.total_chunks} processed"
        )
        logger.info(f"Principles extracted: {self.stats.principles_extracted}")
        logger.info(
            f"Validations: {self.stats.validation_successes} success, "
            f"{self.stats.validation_failures} failed"
        )
        logger.info(f"Average confidence: {self.stats.avg_confidence:.2f}")
        logger.info("-" * 70)
        logger.info(f"Tokenizer: {self.stats.tokenizer_used or 'None'}")
        if self.stats.vocab_size:
            logger.info(f"Vocab size: {self.stats.vocab_size:,}")
        logger.info(f"Execution time: {self.stats.execution_time_seconds:.2f}s")
        logger.info(f"Errors: {len(self.stats.errors)}")
        logger.info(f"Status: {'SUCCESS' if self.stats.success else 'FAILED'}")
        logger.info("=" * 70)

        if self.stats.errors:
            logger.warning("Errors encountered:")
            for error in self.stats.errors[:10]:  # Show first 10 errors
                logger.warning(f"  - {error}")
            if len(self.stats.errors) > 10:
                logger.warning(f"  ... and {len(self.stats.errors) - 10} more errors")


# =============================================================================
# CLI INTERFACE
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed the Curiosity Engine with baseline knowledge from a text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/bootstrap_knowledge.py --file data/corpus.txt

    # With custom chunk size and domain
    python scripts/bootstrap_knowledge.py --file science.txt --chunk-size 1000 --domain science

    # Dry run to validate file
    python scripts/bootstrap_knowledge.py --file corpus.txt --dry-run

    # With tokenizer vocabulary
    python scripts/bootstrap_knowledge.py --file corpus.txt --vocab-path checkpoints/vocab.json
        """,
    )

    parser.add_argument(
        "--file", "-f",
        type=Path,
        required=True,
        help="Path to the text file containing knowledge to seed",
    )

    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Size of text chunks in characters (default: {DEFAULT_CHUNK_SIZE})",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks in characters (default: {DEFAULT_CHUNK_OVERLAP})",
    )

    parser.add_argument(
        "--domain", "-d",
        default=DEFAULT_DOMAIN,
        help=f"Knowledge domain for the content (default: {DEFAULT_DOMAIN})",
    )

    parser.add_argument(
        "--vocab-path", "-v",
        type=Path,
        default=None,
        help="Path to tokenizer vocabulary file (optional)",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process (for testing)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk file without crystallizing",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for each chunk",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for bootstrap_knowledge script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup signal handlers
    _setup_signal_handlers()

    # Parse arguments
    args = parse_arguments()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    logger.info("=" * 70)
    logger.info("VULCAN-AGI Knowledge Bootstrap")
    logger.info("=" * 70)

    try:
        # Create configuration
        config = BootstrapConfig(
            file_path=args.file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            domain=args.domain,
            vocab_path=args.vocab_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
            max_chunks=args.max_chunks,
        )

        logger.info(f"Configuration: {config.to_dict()}")

        # Create and initialize bootstrapper
        bootstrapper = KnowledgeBootstrapper(config)

        if not bootstrapper.initialize():
            logger.error("✗ Initialization failed")
            return 1

        # Run bootstrap
        stats = bootstrapper.run()

        # Return appropriate exit code
        if stats.success:
            logger.info("✓ Knowledge bootstrap completed successfully")
            return 0
        else:
            logger.error("✗ Knowledge bootstrap failed")
            return 1

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Bootstrap interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Bootstrap failed with unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

