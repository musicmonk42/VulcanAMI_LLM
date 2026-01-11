"""Test suite for cli/interactive.py - CLI interactive mode features"""

import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# COLOR TESTS
# ============================================================


class TestColors:
    """Test color code handling."""

    def test_colors_disable(self):
        """Test color codes can be disabled."""
        from vulcan.cli.interactive import Colors
        
        # Save original state
        original_enabled = Colors._colors_enabled
        original_red = Colors.RED
        
        try:
            # Disable colors
            Colors.disable()
            
            # All color codes should be empty strings
            assert Colors.RED == ''
            assert Colors.GREEN == ''
            assert Colors.BLUE == ''
            assert Colors.YELLOW == ''
            assert Colors.CYAN == ''
            assert Colors.ENDC == ''
            assert Colors.BOLD == ''
            assert Colors.HEADER == ''
            
            assert not Colors._colors_enabled
        finally:
            # Restore original state
            Colors._colors_enabled = original_enabled
            if original_enabled:
                Colors.enable()

    def test_colors_enable(self):
        """Test colors can be re-enabled."""
        from vulcan.cli.interactive import Colors
        
        # Save original state
        original_enabled = Colors._colors_enabled
        
        try:
            # Disable first
            Colors.disable()
            assert Colors.RED == ''
            
            # Re-enable (only works on non-Windows with TTY)
            Colors.enable()
            
            # If platform supports colors and stdout is TTY, codes should be set
            if sys.platform != 'win32' and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                assert Colors._colors_enabled
                assert Colors.RED != ''
            else:
                # Otherwise still disabled
                assert Colors.RED == ''
        finally:
            # Restore original state
            Colors._colors_enabled = original_enabled

    def test_colors_windows_detection(self):
        """Test Windows auto-detection."""
        from vulcan.cli.interactive import Colors
        
        # The Colors class should have _colors_enabled set based on platform
        # On Windows, it should be False by default
        if sys.platform == 'win32':
            # Colors should be disabled on Windows
            assert not Colors._colors_enabled
        else:
            # On other platforms, depends on TTY
            pass  # Can vary based on environment


# ============================================================
# COMMAND ALIAS TESTS
# ============================================================


class TestCommandAliases:
    """Test command alias resolution."""

    def test_command_aliases(self):
        """Test alias resolution."""
        from vulcan.cli.interactive import COMMAND_ALIASES
        
        # Verify key aliases exist
        assert COMMAND_ALIASES['q'] == 'query'
        assert COMMAND_ALIASES['s'] == 'status'
        assert COMMAND_ALIASES['m'] == 'memory'
        assert COMMAND_ALIASES['i'] == 'improve'
        assert COMMAND_ALIASES['b'] == 'benchmark'
        assert COMMAND_ALIASES['h'] == 'help'
        assert COMMAND_ALIASES['?'] == 'help'

    @patch('vulcan.cli.interactive.handle_status')
    @patch('builtins.input', side_effect=['s', 'exit'])
    def test_alias_execution(self, mock_input, mock_handle):
        """Test that aliases are properly executed."""
        from vulcan.cli.interactive import run_interactive_mode
        
        # Run interactive mode with aliased command
        run_interactive_mode()
        
        # handle_status should have been called (s -> status)
        mock_handle.assert_called_once()


# ============================================================
# COMMAND SUGGESTION TESTS
# ============================================================


class TestCommandSuggestions:
    """Test command suggestion for typos."""

    def test_command_suggestions(self):
        """Test typo suggestions."""
        from vulcan.cli.interactive import suggest_command
        
        # Close typos should get suggestions
        assert suggest_command('quer') == 'query'
        assert suggest_command('statu') == 'status'
        assert suggest_command('memor') == 'memory'
        
        # Very different strings should return None
        assert suggest_command('xyz123') is None
        assert suggest_command('asdfghjkl') is None

    def test_suggestion_cutoff(self):
        """Test suggestion cutoff threshold."""
        from vulcan.cli.interactive import suggest_command
        
        # Partially similar should still suggest (qu matches quit better)
        suggestion = suggest_command('qu')
        assert suggestion in ['query', 'quit']  # Either is acceptable
        
        # But very dissimilar should not
        assert suggest_command('a') is None


# ============================================================
# READLINE HISTORY TESTS
# ============================================================


class TestReadlineHistory:
    """Test readline history setup."""

    @patch('vulcan.cli.interactive.READLINE_AVAILABLE', True)
    @patch('vulcan.cli.interactive.readline')
    def test_readline_history_setup(self, mock_readline):
        """Test history file configuration."""
        from vulcan.cli.interactive import setup_readline
        
        # Mock the readline module
        mock_readline.read_history_file = MagicMock()
        mock_readline.set_history_length = MagicMock()
        mock_readline.write_history_file = MagicMock()
        mock_readline.parse_and_bind = MagicMock()
        
        # Setup readline
        setup_readline()
        
        # Should set history length
        mock_readline.set_history_length.assert_called_once_with(1000)
        
        # Should configure key bindings
        assert mock_readline.parse_and_bind.call_count >= 2

    @patch('vulcan.cli.interactive.READLINE_AVAILABLE', False)
    def test_readline_not_available(self):
        """Test graceful handling when readline not available."""
        from vulcan.cli.interactive import setup_readline
        
        # Should not raise exception
        setup_readline()

    @patch('vulcan.cli.interactive.READLINE_AVAILABLE', True)
    @patch('vulcan.cli.interactive.readline')
    @patch('vulcan.cli.interactive.atexit')
    def test_history_persistence(self, mock_atexit, mock_readline):
        """Test history file is saved on exit."""
        from vulcan.cli.interactive import setup_readline
        
        mock_readline.read_history_file = MagicMock()
        mock_readline.set_history_length = MagicMock()
        mock_readline.write_history_file = MagicMock()
        mock_readline.parse_and_bind = MagicMock()
        
        setup_readline()
        
        # Should register atexit handler
        mock_atexit.register.assert_called()
        
        # The registered function should be write_history_file
        args, kwargs = mock_atexit.register.call_args
        assert mock_readline.write_history_file in args


# ============================================================
# TAB COMPLETION TESTS
# ============================================================


class TestTabCompletion:
    """Test command tab completion."""

    @patch('vulcan.cli.interactive.READLINE_AVAILABLE', True)
    @patch('vulcan.cli.interactive.readline')
    def test_tab_completion(self, mock_readline):
        """Test command completion."""
        from vulcan.cli.interactive import setup_completion
        
        mock_readline.set_completer = MagicMock()
        mock_readline.parse_and_bind = MagicMock()
        
        setup_completion()
        
        # Should set completer
        mock_readline.set_completer.assert_called_once()
        
        # Get the completer function
        completer = mock_readline.set_completer.call_args[0][0]
        
        # Test completion
        assert completer('que', 0) == 'query'
        assert completer('que', 1) is None  # No more matches
        
        assert completer('st', 0) == 'status'
        assert completer('st', 1) is None

    @patch('vulcan.cli.interactive.READLINE_AVAILABLE', True)
    @patch('vulcan.cli.interactive.readline')
    def test_completion_multiple_matches(self, mock_readline):
        """Test completion with multiple matches."""
        from vulcan.cli.interactive import setup_completion
        
        mock_readline.set_completer = MagicMock()
        mock_readline.parse_and_bind = MagicMock()
        
        setup_completion()
        
        # Get the completer function
        completer = mock_readline.set_completer.call_args[0][0]
        
        # 'q' matches both 'query' and 'quit'
        matches = []
        state = 0
        while True:
            result = completer('q', state)
            if result is None:
                break
            matches.append(result)
            state += 1
        
        assert 'query' in matches
        assert 'quit' in matches
        assert len(matches) == 2

    @patch('vulcan.cli.interactive.READLINE_AVAILABLE', False)
    def test_completion_not_available(self):
        """Test graceful handling when readline not available."""
        from vulcan.cli.interactive import setup_completion
        
        # Should not raise exception
        setup_completion()


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestInteractiveMode:
    """Test interactive mode integration."""

    @patch('builtins.input', side_effect=['help', 'exit'])
    def test_help_command(self, mock_input):
        """Test help command displays commands."""
        from vulcan.cli.interactive import run_interactive_mode
        
        # Capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            run_interactive_mode()
            output = fake_out.getvalue()
            
            # Should display available commands
            assert 'query' in output.lower()
            assert 'status' in output.lower()
            assert 'memory' in output.lower()

    @patch('builtins.input', side_effect=['unknown_command', 'exit'])
    def test_unknown_command_suggestion(self, mock_input):
        """Test unknown command shows suggestion."""
        from vulcan.cli.interactive import run_interactive_mode
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            run_interactive_mode()
            output = fake_out.getvalue()
            
            # Should show unknown command message
            assert 'Unknown command' in output or 'unknown' in output.lower()

    @patch('builtins.input', side_effect=KeyboardInterrupt())
    def test_keyboard_interrupt_handling(self, mock_input):
        """Test graceful handling of Ctrl+C."""
        from vulcan.cli.interactive import run_interactive_mode
        
        # Should handle KeyboardInterrupt gracefully
        with patch('sys.stdout', new=StringIO()):
            with patch('builtins.input', side_effect=[KeyboardInterrupt(), 'exit']):
                run_interactive_mode()
