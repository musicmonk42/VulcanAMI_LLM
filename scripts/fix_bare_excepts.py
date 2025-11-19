#!/usr/bin/env python3
"""
Automated script to fix bare except clauses across the codebase.
This script identifies bare except clauses and suggests proper fixes.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_bare_excepts(file_path: Path) -> List[Tuple[int, str]]:
    """Find all bare except clauses in a file."""
    bare_excepts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines, 1):
            # Match "except:" but not "except Exception" or "except SomeError"
            if re.search(r'^\s*except\s*:\s*(?:#.*)?$', line):
                bare_excepts.append((i, line.rstrip()))
                
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        
    return bare_excepts

def suggest_fix(file_path: Path, line_num: int, context_lines: int = 5) -> str:
    """Suggest a fix for a bare except clause based on context."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Get context
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        context = lines[start:end]
        
        # Analyze context to suggest appropriate exception types
        context_str = ''.join(context).lower()
        
        suggestions = []
        
        # Common patterns
        if 'json' in context_str or 'loads' in context_str or 'dumps' in context_str:
            suggestions.append('json.JSONDecodeError')
        if 'open' in context_str or 'file' in context_str or 'read' in context_str:
            suggestions.append('(FileNotFoundError, IOError)')
        if 'pickle' in context_str:
            suggestions.append('(pickle.PicklingError, pickle.UnpicklingError)')
        if 'int(' in context_str or 'float(' in context_str:
            suggestions.append('ValueError')
        if 'dict' in context_str or 'key' in context_str:
            suggestions.append('KeyError')
        if 'index' in context_str or 'list' in context_str:
            suggestions.append('IndexError')
        if 'import' in context_str:
            suggestions.append('ImportError')
        if 'attribute' in context_str or '.' in context_str:
            suggestions.append('AttributeError')
        if 'type' in context_str:
            suggestions.append('TypeError')
        if '__del__' in context_str or 'destructor' in context_str:
            suggestions.append('Exception')
        if 'pass' in context_str and len(context_str) < 100:
            suggestions.append('Exception')
            
        if not suggestions:
            suggestions.append('Exception')
            
        return ' or '.join(set(suggestions))
        
    except Exception as e:
        logger.error(f"Error suggesting fix for {file_path}:{line_num}: {e}")
        return 'Exception'

def fix_bare_except(file_path: Path, line_num: int, suggested_exception: str) -> bool:
    """Fix a bare except clause in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if line_num < 1 or line_num > len(lines):
            logger.error(f"Invalid line number {line_num} for {file_path}")
            return False
            
        old_line = lines[line_num - 1]
        
        # Extract indentation
        indent = len(old_line) - len(old_line.lstrip())
        indent_str = old_line[:indent]
        
        # Create new line with proper exception handling
        if 'Exception' in suggested_exception:
            new_line = f"{indent_str}except Exception as e:\n"
        else:
            new_line = f"{indent_str}except {suggested_exception} as e:\n"
        
        # Check if next line is just 'pass' - if so, add logging
        if line_num < len(lines):
            next_line = lines[line_num].strip()
            if next_line == 'pass':
                # Replace pass with proper logging
                lines[line_num] = f"{indent_str}    logger.error(f\"Operation failed: {{e}}\")\n"
        
        lines[line_num - 1] = new_line
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        logger.info(f"Fixed {file_path}:{line_num}")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing {file_path}:{line_num}: {e}")
        return False

def main():
    """Main function to scan and fix all bare except clauses."""
    src_dir = Path('src')
    
    if not src_dir.exists():
        logger.error("src directory not found")
        return
        
    all_fixes = []
    
    # Find all Python files
    for py_file in src_dir.rglob('*.py'):
        bare_excepts = find_bare_excepts(py_file)
        
        if bare_excepts:
            logger.info(f"\nFile: {py_file}")
            logger.info(f"Found {len(bare_excepts)} bare except clause(s)")
            
            for line_num, line_content in bare_excepts:
                suggestion = suggest_fix(py_file, line_num)
                all_fixes.append((py_file, line_num, suggestion))
                logger.info(f"  Line {line_num}: {line_content.strip()}")
                logger.info(f"  Suggested fix: except {suggestion} as e:")
    
    # Report summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Total files with bare except: {len(set(f[0] for f in all_fixes))}")
    logger.info(f"Total bare except clauses: {len(all_fixes)}")
    logger.info(f"{'='*70}")
    
    # Ask for confirmation before fixing
    response = input("\nDo you want to automatically fix these? (yes/no): ")
    
    if response.lower() == 'yes':
        fixed_count = 0
        for file_path, line_num, suggestion in all_fixes:
            if fix_bare_except(file_path, line_num, suggestion):
                fixed_count += 1
        
        logger.info(f"\nFixed {fixed_count} out of {len(all_fixes)} bare except clauses")
    else:
        logger.info("No changes made. Run with 'yes' to apply fixes.")

if __name__ == '__main__':
    main()
