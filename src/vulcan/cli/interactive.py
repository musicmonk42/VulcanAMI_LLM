"""
CLI Interactive Mode

Modern interactive REPL interface for VULCAN with cross-platform support,
command history, color-coded output, and real API integration.

Features:
- Cross-platform colors and history (Windows, Linux, macOS)
- Tab completion for commands
- Command history with up/down arrows
- Real-time API integration
- Authentication support
- Graceful error handling

Powered by prompt_toolkit for superior cross-platform UX.
"""

import difflib
import logging
import os
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from vulcan.cli.client import VulcanAPIError, VulcanClient

logger = logging.getLogger(__name__)

# Cross-platform style (works on Windows, Linux, macOS!)
VULCAN_STYLE = Style.from_dict({
    'prompt': '#00aa00 bold',
    'command': '#ffffff',
    'error': '#ff0000',
    'success': '#00ff00',
    'info': '#00aaaa',
    'warning': '#ffaa00',
    'header': '#ff00ff bold',
    'cyan': '#00ffff',
})


def print_colored(message: str, style: str = ''):
    """
    Print colored message using prompt_toolkit.
    
    Args:
        message: Message to print
        style: Style class (error, success, info, warning, header, cyan)
    """
    from prompt_toolkit import print_formatted_text
    from prompt_toolkit.formatted_text import FormattedText
    
    if style:
        print_formatted_text(FormattedText([(f'class:{style}', message)]), style=VULCAN_STYLE)
    else:
        print(message)


class Colors:
    """Legacy Colors class for backward compatibility with tests."""
    
    _colors_enabled = True
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @classmethod
    def disable(cls):
        """Disable colors (useful for testing or piped output)."""
        cls._colors_enabled = False
        for attr in ['HEADER', 'BLUE', 'GREEN', 'RED', 'YELLOW', 'CYAN', 'ENDC', 'BOLD']:
            setattr(cls, attr, '')
    
    @classmethod
    def enable(cls):
        """Re-enable colors if supported."""
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
AVAILABLE_COMMANDS = ['query', 'status', 'memory', 'improve', 'benchmark', 'config', 'help', 'exit', 'quit']


def print_header():
    """Print welcome header."""
    print_colored("\n" + "=" * 60, 'cyan')
    print_colored("VULCAN Interactive Mode", 'header')
    print_colored("=" * 60, 'cyan')
    print_colored("\nAvailable commands:", 'warning')
    print("  query <text>    - Query VULCAN (alias: q)")
    print("  status          - Show system status (alias: s)")
    print("  memory <query>  - Search memory (alias: m)")
    print("  improve         - Show self-improvement status (alias: i)")
    print("  benchmark       - Run performance benchmarks (alias: b)")
    print("  config          - Show current configuration")
    print("  help            - Show this help message (alias: h, ?)")
    print("  exit            - Exit interactive mode\n")


def suggest_command(unknown_cmd: str) -> Optional[str]:
    """Suggest similar commands for typos."""
    suggestions = difflib.get_close_matches(
        unknown_cmd,
        AVAILABLE_COMMANDS,
        n=1,
        cutoff=0.6
    )
    return suggestions[0] if suggestions else None


def handle_query(client: VulcanClient, query_text: str) -> None:
    """
    Handle query command with real API call.
    
    Args:
        client: VulcanClient instance
        query_text: The query to process
    """
    print_colored(f"\nProcessing query: {query_text}", 'info')
    
    try:
        # Make real API call
        result = client.chat(query_text)
        
        print_colored("\nResponse:", 'success')
        print(result.get("response", "No response received"))
        
        # Show metadata if available
        if "metadata" in result:
            meta = result["metadata"]
            print_colored("\nMetadata:", 'info')
            if "reasoning_type" in meta:
                print(f"  Reasoning: {meta['reasoning_type']}")
            if "processing_time" in meta:
                print(f"  Processing time: {meta['processing_time']:.2f}s")
    
    except VulcanAPIError as e:
        print_colored(f"\nAPI Error: {e.message}", 'error')
    except Exception as e:
        print_colored(f"\nError processing query: {e}", 'error')
        logger.exception("Query error")


def handle_status(client: VulcanClient) -> None:
    """Handle status command with real API call."""
    print_colored("\nSystem Status:", 'info')
    
    try:
        # Make real API call
        result = client.health()
        
        status = result.get("status", "unknown")
        if status == "healthy":
            print_colored(f"✓ System operational", 'success')
        else:
            print_colored(f"⚠ System status: {status}", 'warning')
        
        # Show additional health info
        if "uptime" in result:
            print(f"  Uptime: {result['uptime']}")
        if "version" in result:
            print(f"  Version: {result['version']}")
        if "components" in result:
            print_colored("\n  Components:", 'info')
            for comp, status in result["components"].items():
                icon = "✓" if status == "healthy" else "✗"
                color = 'success' if status == "healthy" else 'error'
                print_colored(f"    {icon} {comp}: {status}", color)
    
    except VulcanAPIError as e:
        print_colored(f"\nAPI Error: {e.message}", 'error')
        if e.status_code == 0:
            print_colored("\nTroubleshooting:", 'warning')
            print("  1. Check if the server is running")
            print("  2. Verify the server URL is correct")
            print("  3. Check network connectivity")
    except Exception as e:
        print_colored(f"\nError getting status: {e}", 'error')
        logger.exception("Status error")


def handle_memory(client: VulcanClient, query: str = "") -> None:
    """Handle memory command with real API call."""
    print_colored("\nMemory Search:", 'info')
    
    if not query:
        print_colored("Usage: memory <search query>", 'warning')
        print("Example: memory recent improvements")
        return
    
    try:
        # Make real API call
        result = client.search_memory(query, k=10)
        
        memories = result.get("results", [])
        if memories:
            print_colored(f"\nFound {len(memories)} memories:", 'success')
            for i, memory in enumerate(memories, 1):
                print(f"\n{i}. {memory.get('content', 'No content')[:200]}...")
                if "score" in memory:
                    print(f"   Relevance: {memory['score']:.3f}")
        else:
            print_colored("No memories found matching your query.", 'warning')
    
    except VulcanAPIError as e:
        print_colored(f"\nAPI Error: {e.message}", 'error')
    except Exception as e:
        print_colored(f"\nError searching memory: {e}", 'error')
        logger.exception("Memory search error")


def handle_improve() -> None:
    """Handle improve command (placeholder)."""
    print_colored("\nSelf-Improvement Status:", 'info')
    print_colored("⚠ This feature requires server-side implementation", 'warning')
    print("Future versions will show:")
    print("  - Improvements proposed and approved")
    print("  - Success rate")
    print("  - Recent improvements")


def handle_benchmark() -> None:
    """Handle benchmark command (placeholder)."""
    print_colored("\nPerformance Benchmarks:", 'info')
    print_colored("⚠ This feature requires server-side implementation", 'warning')
    print("Future versions will:")
    print("  - Run performance benchmarks")
    print("  - Show timing metrics")
    print("  - Compare against baselines")


def handle_config(client: VulcanClient) -> None:
    """Handle config command - show current configuration."""
    print_colored("\nCurrent Configuration:", 'info')
    print(f"  Server URL: {client.base_url}")
    
    if client.client.headers.get("X-API-Key"):
        print_colored("  API Key: *** (configured)", 'success')
    else:
        print_colored("  API Key: Not set", 'warning')
        print("\n  To set API key:")
        print("    export VULCAN_API_KEY=your-key-here")
    
    print("\n  Environment Variables:")
    print("    VULCAN_SERVER_URL - Server URL")
    print("    VULCAN_API_KEY    - API authentication key")



def run_interactive_mode():
    """
    Run interactive REPL mode with prompt_toolkit.
    
    Modern interactive command-line interface for VULCAN with:
    - Cross-platform colors and history (Windows, Linux, macOS)
    - Tab completion for commands
    - Command history with up/down arrows
    - Real-time API integration
    - Authentication support
    
    Example:
        ```python
        from vulcan.cli.interactive import run_interactive_mode
        run_interactive_mode()
        ```
    """
    print_header()
    
    # Initialize API client using CLIConfig for consistent configuration
    try:
        from vulcan.cli.config import CLIConfig
        config = CLIConfig()
        client = VulcanClient(
            base_url=config.get_server_url(),
            api_key=config.get_api_key()
        )
        print_colored(f"Connected to: {client.base_url}", 'info')
        
        if config.get_api_key():
            print_colored("Authentication: Enabled ✓", 'success')
        else:
            print_colored("Authentication: Not configured", 'warning')
            print("Set VULCAN_API_KEY environment variable to enable authentication\n")
    except Exception as e:
        print_colored(f"Warning: Failed to initialize client: {e}", 'warning')
        print("Some features may not be available.\n")
        client = None
    
    # Setup prompt_toolkit session
    history_file = os.path.expanduser("~/.vulcan_history")
    command_completer = WordCompleter(AVAILABLE_COMMANDS, ignore_case=True)
    
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=command_completer,
        style=VULCAN_STYLE,
    )
    
    print_colored("Command history and tab completion enabled ✓", 'success')
    print()
    
    while True:
        try:
            # Get user input with styled prompt
            user_input = session.prompt(
                HTML('<prompt>vulcan&gt; </prompt>'),
                style=VULCAN_STYLE
            ).strip()
            
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
                print_colored("\nExiting VULCAN interactive mode. Goodbye!", 'info')
                break
            
            elif command == "help":
                print_header()
            
            elif command == "query":
                if args and client:
                    handle_query(client, args)
                elif not args:
                    print_colored("Usage: query <text>", 'warning')
                else:
                    print_colored("Error: Client not initialized", 'error')
            
            elif command == "status":
                if client:
                    handle_status(client)
                else:
                    print_colored("Error: Client not initialized", 'error')
            
            elif command == "memory":
                if client:
                    handle_memory(client, args)
                else:
                    print_colored("Error: Client not initialized", 'error')
            
            elif command == "improve":
                handle_improve()
            
            elif command == "benchmark":
                handle_benchmark()
            
            elif command == "config":
                if client:
                    handle_config(client)
                else:
                    print_colored("Error: Client not initialized", 'error')
            
            else:
                print_colored(f"Unknown command: {command}", 'error')
                suggestion = suggest_command(command)
                if suggestion:
                    print_colored(f"Did you mean '{suggestion}'?", 'warning')
                else:
                    print_colored("Type 'help' for available commands", 'warning')
        
        except KeyboardInterrupt:
            print_colored("\n\nUse 'exit' to quit interactive mode", 'warning')
            continue
        
        except EOFError:
            print_colored("\n\nExiting VULCAN interactive mode. Goodbye!", 'info')
            break
        
        except Exception as e:
            print_colored(f"\nError: {e}", 'error')
            logger.exception("Error in interactive mode")
    
    # Cleanup
    if client:
        client.close()


# Legacy compatibility stubs for tests
READLINE_AVAILABLE = True  # For test compatibility


def setup_readline():
    """Legacy compatibility - no longer needed with prompt_toolkit."""
    pass


def setup_completion():
    """Legacy compatibility - no longer needed with prompt_toolkit."""
    pass


if __name__ == "__main__":
    # Run interactive mode if executed directly
    run_interactive_mode()
