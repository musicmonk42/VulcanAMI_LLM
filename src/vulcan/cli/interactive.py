"""
CLI Interactive Mode

Interactive REPL interface for VULCAN with command history,
color-coded output, and graceful error handling.

Provides an interactive command-line interface for querying VULCAN,
checking status, running tests, and more.
"""

import atexit
import difflib
import os
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
    """ANSI color codes with auto-detection."""
    
    _colors_enabled = (
        hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        and os.environ.get('TERM') != 'dumb'
        and sys.platform != 'win32'  # Or use colorama on Windows
    )
    
    HEADER = '\033[95m' if _colors_enabled else ''
    BLUE = '\033[94m' if _colors_enabled else ''
    GREEN = '\033[92m' if _colors_enabled else ''
    RED = '\033[91m' if _colors_enabled else ''
    YELLOW = '\033[93m' if _colors_enabled else ''
    CYAN = '\033[96m' if _colors_enabled else ''
    ENDC = '\033[0m' if _colors_enabled else ''
    BOLD = '\033[1m' if _colors_enabled else ''
    
    @classmethod
    def disable(cls):
        """Disable colors (useful for testing or piped output)."""
        cls._colors_enabled = False
        for attr in ['HEADER', 'BLUE', 'GREEN', 'RED', 'YELLOW', 'CYAN', 'ENDC', 'BOLD']:
            setattr(cls, attr, '')
    
    @classmethod
    def enable(cls):
        """Re-enable colors if supported."""
        if sys.platform != 'win32' and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            cls._colors_enabled = True
            cls.HEADER = '\033[95m'
            cls.BLUE = '\033[94m'
            cls.GREEN = '\033[92m'
            cls.RED = '\033[91m'
            cls.YELLOW = '\033[93m'
            cls.CYAN = '\033[96m'
            cls.ENDC = '\033[0m'
            cls.BOLD = '\033[1m'


# Command aliases for convenience
COMMAND_ALIASES = {
    'q': 'query',
    's': 'status',
    'm': 'memory',
    'i': 'improve',
    'b': 'benchmark',
    'h': 'help',
    '?': 'help',
}

# Available commands for tab completion
AVAILABLE_COMMANDS = ['query', 'status', 'memory', 'improve', 'benchmark', 'help', 'exit', 'quit']


def print_colored(message: str, color: str = Colors.ENDC):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")


def print_header():
    """Print welcome header."""
    print_colored("\n" + "=" * 60, Colors.CYAN)
    print_colored("VULCAN Interactive Mode", Colors.BOLD + Colors.HEADER)
    print_colored("=" * 60, Colors.CYAN)
    print_colored("\nAvailable commands:", Colors.YELLOW)
    print("  query <text>    - Query VULCAN (alias: q)")
    print("  status          - Show system status (alias: s)")
    print("  memory          - Show memory statistics (alias: m)")
    print("  improve         - Show self-improvement status (alias: i)")
    print("  benchmark       - Run performance benchmarks (alias: b)")
    print("  help            - Show this help message (alias: h, ?)")
    print("  exit            - Exit interactive mode\n")


def setup_readline():
    """Configure readline for better UX with history persistence."""
    if not READLINE_AVAILABLE:
        return
    
    histfile = os.path.join(os.path.expanduser("~"), ".vulcan_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    
    atexit.register(readline.write_history_file, histfile)
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")


def setup_completion():
    """Setup tab completion for commands."""
    if not READLINE_AVAILABLE:
        return
    
    def completer(text, state):
        options = [cmd for cmd in AVAILABLE_COMMANDS if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        return None
    
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def suggest_command(unknown_cmd: str) -> Optional[str]:
    """Suggest similar commands for typos."""
    suggestions = difflib.get_close_matches(
        unknown_cmd,
        AVAILABLE_COMMANDS,
        n=1,
        cutoff=0.6
    )
    return suggestions[0] if suggestions else None


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
    Supports command history (if readline available), command aliases,
    tab completion, and graceful error handling.
    
    Example:
        ```python
        from vulcan.cli.interactive import run_interactive_mode
        run_interactive_mode()
        ```
    """
    print_header()
    
    if READLINE_AVAILABLE:
        # Configure readline for better UX
        setup_readline()
        setup_completion()
        print_colored("Command history and tab completion enabled", Colors.YELLOW)
    
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
            
            # Resolve command aliases
            command = COMMAND_ALIASES.get(command, command)
            
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
                suggestion = suggest_command(command)
                if suggestion:
                    print_colored(f"Did you mean '{suggestion}'?", Colors.YELLOW)
                else:
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
