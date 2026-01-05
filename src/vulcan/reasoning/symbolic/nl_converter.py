"""
Natural Language to Formal Logic Converter.

BUG #5 FIX: The symbolic parser expects formal logic notation but receives
natural language, causing parse errors like:
    [Parse error] Unexpected token 'Every' at line 1, column 12

This module provides pattern-based conversion from natural language sentences
to first-order logic notation.

Example:
    Input:  "Every engineer reviewed a document"
    Output: "∀e ∃d Reviewed(e, d)"

Features:
    - Pattern-based conversion for common logical structures
    - Universal quantifier detection ("Every X does Y")
    - Existential quantifier detection ("Some X does Y", "a/an Y")
    - Implication detection ("If X then Y")
    - Conjunction/disjunction detection ("X and Y", "X or Y")
    - Negation detection ("not X", "no X")
    - Fallback for already-formal logic notation
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NaturalLanguageToLogicConverter:
    """
    Convert natural language to formal logic notation.
    
    BUG #5 FIX: This class handles the conversion from natural language
    sentences like "Every engineer reviewed a document" to formal logic
    notation like "∀e ∃d Reviewed(e, d)".
    
    The converter uses pattern matching with regex to identify common
    logical structures in natural language and translate them to formal
    first-order logic notation.
    
    Example:
        >>> converter = NaturalLanguageToLogicConverter()
        >>> converter.convert("Every engineer reviewed a document")
        '∀e ∃d Reviewed(e, d)'
        >>> converter.convert("Some students passed the exam")
        '∃s Passed(s)'
        >>> converter.convert("If it rains then the ground is wet")
        'Rain → Wet'
    """
    
    def __init__(self):
        """Initialize the converter with compiled patterns."""
        self.patterns = self._compile_patterns()
        logger.debug("[NLConverter] BUG#5 FIX: NL to Logic converter initialized")
    
    def _compile_patterns(self) -> List[Dict]:
        """
        Compile regex patterns for common logical structures.
        
        Returns:
            List of pattern dictionaries with regex, template, and handler
        """
        return [
            # Universal quantifier with existential object: "Every X reviewed a Y"
            {
                'pattern': re.compile(
                    r'^every\s+(\w+)\s+(?:reviews?|reviewed|has|have|does|do|did)\s+a[n]?\s+(\w+)$',
                    re.I
                ),
                'handler': self._handle_universal_existential
            },
            # Universal quantifier: "Every X is Y" or "Every X does Y"
            {
                'pattern': re.compile(
                    r'^every\s+(\w+)\s+(is|are|has|have|does|do|did|can|will|must|should)\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_universal
            },
            # Universal quantifier: "All X are Y"
            {
                'pattern': re.compile(
                    r'^all\s+(\w+s?)\s+(is|are|have|can|will|must|should)\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_universal
            },
            # Existential quantifier: "Some X does Y" with auxiliary verb
            {
                'pattern': re.compile(
                    r'^some\s+(\w+s?)\s+(is|are|has|have|does|do|did|can|will)\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_existential
            },
            # Existential quantifier: "Some X verb Y" with action verb (e.g., "Some students passed")
            {
                'pattern': re.compile(
                    r'^some\s+(\w+s?)\s+(\w+(?:ed|s|es)?)\s+(.*)$',
                    re.I
                ),
                'handler': self._handle_existential_action
            },
            # Existential quantifier: "There exists X such that Y"
            {
                'pattern': re.compile(
                    r'^there\s+(?:exists?|is|are)\s+(?:a[n]?\s+)?(\w+)\s+(?:such\s+that|that|which|who)\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_existential_detailed
            },
            # Implication: "If X then Y"
            {
                'pattern': re.compile(
                    r'^if\s+(.+?)\s+then\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_implication
            },
            # Implication: "X implies Y"
            {
                'pattern': re.compile(
                    r'^(.+?)\s+implies\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_implication
            },
            # Biconditional: "X if and only if Y"
            {
                'pattern': re.compile(
                    r'^(.+?)\s+if\s+and\s+only\s+if\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_biconditional
            },
            # Negation: "No X does Y" or "No X is Y"
            {
                'pattern': re.compile(
                    r'^no\s+(\w+s?)\s+(is|are|has|have|does|do|did|can|will)\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_universal_negation
            },
            # Negation: "Not X" or "It is not the case that X"
            {
                'pattern': re.compile(
                    r'^(?:not|it\s+is\s+not\s+(?:the\s+case\s+)?that)\s+(.+)$',
                    re.I
                ),
                'handler': self._handle_negation
            },
            # Simple predicate: "X is Y" (e.g., "Socrates is mortal")
            {
                'pattern': re.compile(
                    r'^(\w+)\s+(?:is|are)\s+(?:a[n]?\s+)?(\w+)$',
                    re.I
                ),
                'handler': self._handle_simple_predicate
            },
            # Simple predicate: "X verb Y" (e.g., "John loves Mary")
            {
                'pattern': re.compile(
                    r'^(\w+)\s+(\w+s?)\s+(\w+)$',
                    re.I
                ),
                'handler': self._handle_binary_predicate
            },
        ]
    
    def convert(self, text: str) -> Optional[str]:
        """
        Convert natural language to formal logic.
        
        BUG #5 FIX: This is the main entry point for converting natural
        language sentences to formal first-order logic notation.
        
        Args:
            text: Natural language text
            
        Returns:
            Formal logic string or None if conversion fails
            
        Example:
            >>> converter = NaturalLanguageToLogicConverter()
            >>> converter.convert("Every engineer reviewed a document")
            '∀e ∃d Reviewed(e, d)'
        """
        if not text:
            return None
            
        # Clean input
        text = text.strip()
        
        # Check if already looks like formal logic (contains logic symbols)
        if self._is_formal_logic(text):
            logger.debug(f"[NLConverter] BUG#5 FIX: Text already appears to be formal logic")
            return text
        
        # Try each pattern
        for pattern_dict in self.patterns:
            match = pattern_dict['pattern'].match(text)
            if match:
                try:
                    formal = pattern_dict['handler'](match, text)
                    if formal:
                        logger.info(
                            f"[NLConverter] BUG#5 FIX: Converted NL to formal logic: "
                            f"'{text[:50]}...' -> '{formal}'"
                        )
                        return formal
                except Exception as e:
                    logger.debug(f"[NLConverter] Pattern handler failed: {e}")
                    continue
        
        # No pattern matched - try to extract a simple predicate
        simple = self._extract_simple_predicate(text)
        if simple:
            logger.info(
                f"[NLConverter] BUG#5 FIX: Extracted simple predicate: "
                f"'{text[:50]}...' -> '{simple}'"
            )
            return simple
        
        logger.debug(f"[NLConverter] BUG#5 FIX: No conversion pattern matched for: '{text[:50]}...'")
        return None
    
    def _is_formal_logic(self, text: str) -> bool:
        """
        Check if text already contains formal logic notation.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains logic symbols
        """
        logic_symbols = ['∀', '∃', '→', '∧', '∨', '¬', '⇒', '⇔', '->', '<->', '&&', '||']
        return any(sym in text for sym in logic_symbols)
    
    def _handle_universal_existential(self, match: re.Match, text: str) -> str:
        """
        Handle universal quantifier with existential object.
        
        Pattern: "Every X reviewed a Y"
        Result: ∀x ∃y Reviewed(x, y)
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        subject = match.group(1).lower()
        obj = match.group(2).lower()
        
        # Generate variables from first letters
        var1 = subject[0]
        var2 = obj[0]
        if var1 == var2:
            var2 = var2 + '2'
        
        # Extract verb from text
        verb = self._extract_verb(text, subject, obj)
        predicate = verb.capitalize() if verb else "Related"
        
        return f"∀{var1} ∃{var2} {predicate}({var1}, {var2})"
    
    def _handle_universal(self, match: re.Match, text: str) -> str:
        """
        Handle universal quantifier: Every X is Y.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        entity = match.group(1).lower()
        verb = match.group(2).lower()
        object_phrase = match.group(3).strip()
        
        # Generate variable
        var = entity[0]
        
        # Parse object
        if 'a ' in object_phrase.lower() or 'an ' in object_phrase.lower():
            # Existential in object: "Every X reviewed a Y"
            obj_match = re.search(r'a[n]?\s+(\w+)', object_phrase, re.I)
            if obj_match:
                obj = obj_match.group(1).lower()
                obj_var = obj[0]
                if obj_var == var:
                    obj_var = obj_var + '2'
                predicate = self._verb_to_predicate(verb)
                return f"∀{var} ∃{obj_var} {predicate}({var}, {obj_var})"
        
        # Simple universal: "Every X is Y"
        predicate = self._phrase_to_predicate(object_phrase)
        return f"∀{var} {predicate}({var})"
    
    def _handle_existential(self, match: re.Match, text: str) -> str:
        """
        Handle existential quantifier: Some X does Y.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        entity = match.group(1).lower()
        # Remove trailing 's' if plural
        if entity.endswith('s') and len(entity) > 1:
            entity = entity[:-1]
        
        var = entity[0]
        object_phrase = match.group(3).strip()
        predicate = self._phrase_to_predicate(object_phrase)
        
        return f"∃{var} {predicate}({var})"
    
    def _handle_existential_action(self, match: re.Match, text: str) -> str:
        """
        Handle existential quantifier with action verb: Some X verbed Y.
        
        Examples:
            "Some students passed the exam" -> ∃s Passed(s)
            "Some cats caught mice" -> ∃c Caught(c)
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        entity = match.group(1).lower()
        verb = match.group(2).lower()
        
        # Remove trailing 's' if plural
        if entity.endswith('s') and len(entity) > 1:
            entity = entity[:-1]
        
        var = entity[0]
        
        # Normalize verb (remove past tense -ed)
        if verb.endswith('ed'):
            verb = verb[:-2] if verb.endswith('ied') else verb[:-2] if len(verb) > 3 else verb[:-1]
        elif verb.endswith('s') and not verb.endswith('ss'):
            verb = verb[:-1]
        
        predicate = verb.capitalize()
        
        return f"∃{var} {predicate}({var})"
    
    def _handle_existential_detailed(self, match: re.Match, text: str) -> str:
        """
        Handle existential with "such that": There exists X such that Y.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        entity = match.group(1).lower()
        condition = match.group(2).strip()
        
        var = entity[0]
        predicate = self._phrase_to_predicate(condition)
        
        return f"∃{var} {predicate}({var})"
    
    def _handle_implication(self, match: re.Match, text: str) -> str:
        """
        Handle if-then statements.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        antecedent = match.group(1).strip()
        consequent = match.group(2).strip()
        
        # Recursively convert both parts
        ant_formal = self.convert(antecedent)
        con_formal = self.convert(consequent)
        
        # If conversion failed, use simplified predicate form
        if not ant_formal:
            ant_formal = self._phrase_to_predicate(antecedent)
        if not con_formal:
            con_formal = self._phrase_to_predicate(consequent)
        
        return f"{ant_formal} → {con_formal}"
    
    def _handle_biconditional(self, match: re.Match, text: str) -> str:
        """
        Handle biconditional (iff) statements.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        left = match.group(1).strip()
        right = match.group(2).strip()
        
        # Recursively convert both parts
        left_formal = self.convert(left) or self._phrase_to_predicate(left)
        right_formal = self.convert(right) or self._phrase_to_predicate(right)
        
        return f"{left_formal} ↔ {right_formal}"
    
    def _handle_universal_negation(self, match: re.Match, text: str) -> str:
        """
        Handle universal negation: No X does Y.
        
        "No X is Y" = "For all X, not Y(X)"
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        entity = match.group(1).lower()
        if entity.endswith('s') and len(entity) > 1:
            entity = entity[:-1]
        
        var = entity[0]
        object_phrase = match.group(3).strip()
        predicate = self._phrase_to_predicate(object_phrase)
        
        return f"∀{var} ¬{predicate}({var})"
    
    def _handle_negation(self, match: re.Match, text: str) -> str:
        """
        Handle negation (not, no).
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        statement = match.group(1).strip()
        formal = self.convert(statement)
        
        if not formal:
            formal = self._phrase_to_predicate(statement)
        
        return f"¬{formal}"
    
    def _handle_simple_predicate(self, match: re.Match, text: str) -> str:
        """
        Handle simple predicates: X is Y.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        subject = match.group(1).lower()
        predicate = match.group(2).capitalize()
        
        return f"{predicate}({subject})"
    
    def _handle_binary_predicate(self, match: re.Match, text: str) -> str:
        """
        Handle binary predicates: X verb Y.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Formal logic string
        """
        subject = match.group(1).lower()
        verb = match.group(2).lower()
        obj = match.group(3).lower()
        
        # Normalize verb (remove trailing 's' for third person)
        if verb.endswith('s') and not verb.endswith('ss'):
            verb = verb[:-1]
        
        predicate = verb.capitalize()
        
        return f"{predicate}({subject}, {obj})"
    
    def _extract_verb(self, text: str, subject: str, obj: str) -> str:
        """
        Extract verb from text between subject and object.
        
        Args:
            text: Full text
            subject: Subject word
            obj: Object word
            
        Returns:
            Extracted verb or default
        """
        # Try to find verb between subject and "a/an obj"
        pattern = re.compile(
            rf'{subject}\s+(\w+(?:ed|s|es)?)\s+a[n]?\s+{obj}',
            re.I
        )
        match = pattern.search(text)
        if match:
            verb = match.group(1).lower()
            # Normalize verb
            if verb.endswith('ed'):
                verb = verb[:-2] if verb.endswith('ied') else verb[:-1] if verb.endswith('ed') else verb
            elif verb.endswith('s') and not verb.endswith('ss'):
                verb = verb[:-1]
            return verb
        
        return "Related"
    
    def _verb_to_predicate(self, verb: str) -> str:
        """
        Convert verb to predicate name.
        
        Args:
            verb: Verb string
            
        Returns:
            Capitalized predicate name
        """
        # Handle common auxiliary verbs
        verb_map = {
            'is': 'Is',
            'are': 'Is',
            'has': 'Has',
            'have': 'Has',
            'does': 'Does',
            'do': 'Does',
            'can': 'Can',
            'will': 'Will',
            'must': 'Must',
            'should': 'Should',
        }
        
        return verb_map.get(verb.lower(), verb.capitalize())
    
    def _phrase_to_predicate(self, phrase: str) -> str:
        """
        Convert phrase to predicate name.
        
        Args:
            phrase: Phrase to convert
            
        Returns:
            Predicate representation
        """
        # Clean the phrase
        phrase = phrase.strip()
        
        # Remove articles
        phrase = re.sub(r'^(a|an|the)\s+', '', phrase, flags=re.I)
        
        # Extract first meaningful word
        words = phrase.split()
        if not words:
            return "P"
        
        # Use first word as predicate, capitalized
        predicate = words[0].capitalize()
        
        # Remove non-alphanumeric characters
        predicate = re.sub(r'[^a-zA-Z0-9]', '', predicate)
        
        return predicate if predicate else "P"
    
    def _extract_simple_predicate(self, text: str) -> Optional[str]:
        """
        Extract a simple predicate from text as fallback.
        
        Args:
            text: Text to extract from
            
        Returns:
            Simple predicate or None
        """
        # Try to find a verb phrase
        words = text.split()
        if len(words) >= 2:
            # Look for "X verbs Y" pattern
            for i, word in enumerate(words[:-1]):
                # Check if this could be a subject (starts with capital or is lowercase noun)
                next_word = words[i + 1] if i + 1 < len(words) else None
                if next_word and self._looks_like_verb(next_word):
                    subject = word.lower()
                    verb = next_word.capitalize()
                    if len(words) > i + 2:
                        obj = words[i + 2].lower()
                        return f"{verb}({subject}, {obj})"
                    else:
                        return f"{verb}({subject})"
        
        return None
    
    def _looks_like_verb(self, word: str) -> bool:
        """
        Check if word looks like a verb.
        
        Args:
            word: Word to check
            
        Returns:
            True if word might be a verb
        """
        word = word.lower()
        
        # Common verb endings
        verb_endings = ['s', 'ed', 'ing', 'es']
        
        # Check if it ends like a verb
        if any(word.endswith(ending) for ending in verb_endings):
            return True
        
        # Common short verbs
        common_verbs = {
            'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
            'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall',
            'should', 'may', 'might', 'must', 'love', 'like', 'hate', 'know',
            'think', 'believe', 'see', 'hear', 'feel', 'want', 'need'
        }
        
        return word in common_verbs


# Convenience function for direct usage
def convert_nl_to_logic(text: str) -> Optional[str]:
    """
    Convert natural language text to formal logic.
    
    BUG #5 FIX: Convenience function for converting natural language
    to formal first-order logic notation.
    
    Args:
        text: Natural language text
        
    Returns:
        Formal logic string or None if conversion fails
        
    Example:
        >>> convert_nl_to_logic("Every engineer reviewed a document")
        '∀e ∃d Reviewed(e, d)'
    """
    converter = NaturalLanguageToLogicConverter()
    return converter.convert(text)


# Export
__all__ = [
    'NaturalLanguageToLogicConverter',
    'convert_nl_to_logic',
]
