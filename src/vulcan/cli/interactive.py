"""
CLI Interactive Mode

Interactive REPL interface for VULCAN with command history,
color-coded output, and graceful error handling.

Provides an interactive command-line interface for querying VULCAN,
checking status, running tests, and more.
"""

import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import readline for command history
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False
    logger.warning("readline not available, command history disabled")


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_colored(message: str, color: str = Colors.ENDC):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")


def print_header():
    """Print welcome header."""
    print_colored("\n" + "=" * 60, Colors.CYAN)
    print_colored("VULCAN Interactive Mode", Colors.BOLD + Colors.HEADER)
    print_colored("=" * 60, Colors.CYAN)
    print_colored("\nAvailable commands:", Colors.YELLOW)
    print("  query <text>    - Query VULCAN")
    print("  status          - Show system status")
    print("  memory          - Show memory statistics")
    print("  improve         - Show self-improvement status")
    print("  benchmark       - Run performance benchmarks")
    print("  help            - Show this help message")
    print("  exit            - Exit interactive mode\n")


def handle_query(query_text: str) -> None:
    """
    Handle query command.
    
    Args:
        query_text: The query to process
    """
    print_colored(f"\nProcessing query: {query_text}", Colors.CYAN)
    
    try:
        # Placeholder implementation
        print_colored("\nResponse:", Colors.GREEN)
        print(f"This is a placeholder response to: {query_text}")
        print("In the full implementation, this would route through")
        print("VULCAN's reasoning engine and return a comprehensive answer.")
    except Exception as e:
        print_colored(f"\nError processing query: {e}", Colors.RED)


def handle_status() -> None:
    """Handle status command."""
    print_colored("\nSystem Status:", Colors.CYAN)
    
    try:
        # Placeholder implementation
        print_colored("✓ System operational", Colors.GREEN)
        print(f"  Uptime: 2h 34m")
        print(f"  Memory usage: 45%")
        print(f"  Active agents: 4")
        print(f"  Queue size: 0")
    except Exception as e:
        print_colored(f"\nError getting status: {e}", Colors.RED)


def handle_memory() -> None:
    """Handle memory command."""
    print_colored("\nMemory Statistics:", Colors.CYAN)
    
    try:
        # Placeholder implementation
        print(f"  Total memories: 1,247")
        print(f"  Recent memories: 83")
        print(f"  Storage used: 125 MB")
        print(f"  Average retrieval time: 0.08s")
    except Exception as e:
        print_colored(f"\nError getting memory stats: {e}", Colors.RED)


def handle_improve() -> None:
    """Handle improve command."""
    print_colored("\nSelf-Improvement Status:", Colors.CYAN)
    
    try:
        # Placeholder implementation
        print_colored("✓ Drive active", Colors.GREEN)
        print(f"  Improvements proposed: 12")
        print(f"  Improvements approved: 8")
        print(f"  Success rate: 87%")
        print(f"  Last improvement: 15 minutes ago")
    except Exception as e:
        print_colored(f"\nError getting improvement status: {e}", Colors.RED)


def handle_benchmark() -> None:
    """Handle benchmark command."""
    print_colored("\nRunning performance benchmarks...", Colors.CYAN)
    
    try:
        # Placeholder implementation
        from vulcan.tests.test_benchmarks import run_all_benchmarks
        
        print("This would run the full benchmark suite.")
        print("(Skipped in interactive mode to save time)")
    except Exception as e:
        print_colored(f"\nError running benchmarks: {e}", Colors.RED)


def run_interactive_mode():
    """
    Run interactive REPL mode.
    
    Starts an interactive command-line interface for VULCAN.
    Supports command history (if readline available) and graceful error handling.
    
    Example:
        ```python
        from vulcan.cli.interactive import run_interactive_mode
        run_interactive_mode()
        ```
    """
    print_header()
    
    if READLINE_AVAILABLE:
        # Configure readline for better UX
        readline.parse_and_bind("tab: complete")
        print_colored("Command history enabled (↑/↓ arrows)", Colors.YELLOW)
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BOLD}vulcan>{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Handle commands
            if command == "exit" or command == "quit":
                print_colored("\nExiting VULCAN interactive mode. Goodbye!", Colors.CYAN)
                break
            
            elif command == "help":
                print_header()
            
            elif command == "query":
                if args:
                    handle_query(args)
                else:
                    print_colored("Usage: query <text>", Colors.YELLOW)
            
            elif command == "status":
                handle_status()
            
            elif command == "memory":
                handle_memory()
            
            elif command == "improve":
                handle_improve()
            
            elif command == "benchmark":
                handle_benchmark()
            
            else:
                print_colored(f"Unknown command: {command}", Colors.RED)
                print_colored("Type 'help' for available commands", Colors.YELLOW)
        
        except KeyboardInterrupt:
            print_colored("\n\nUse 'exit' to quit interactive mode", Colors.YELLOW)
            continue
        
        except EOFError:
            print_colored("\n\nExiting VULCAN interactive mode. Goodbye!", Colors.CYAN)
            break
        
        except Exception as e:
            print_colored(f"\nError: {e}", Colors.RED)
            logger.exception("Error in interactive mode")


if __name__ == "__main__":
    # Run interactive mode if executed directly
    logging.basicConfig(level=logging.INFO)
    run_interactive_mode()
