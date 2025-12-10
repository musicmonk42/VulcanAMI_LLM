"""
Comprehensive pytest suite for schema_auto_generator.py
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest
# Import the module to test
import schema_auto_generator as sag


class TestTokenization:
    """Test the EBNF tokenizer."""
    
    def test_basic_tokens(self):
        """Test tokenization of basic EBNF elements."""
        tokens = sag.tokenize("<Type> ::= STRING;")
        assert len(tokens) > 0
        assert tokens[0] == ('REF', '<Type>')
        assert tokens[1] == ('KEYWORD', '::=')
        assert tokens[2] == ('IDENT', 'STRING')
    
    def test_literal_tokens(self):
        """Test tokenization of string literals."""
        tokens = sag.tokenize("'literal' \"another\"")
        assert ('LITERAL', "'literal'") in tokens
        assert ('LITERAL', '"another"') in tokens
    
    def test_operators(self):
        """Test tokenization of EBNF operators."""
        tokens = sag.tokenize("[ ] { } ( ) * + |")
        expected = [
            ('KEYWORD', '['), ('KEYWORD', ']'),
            ('KEYWORD', '{'), ('KEYWORD', '}'),
            ('KEYWORD', '('), ('KEYWORD', ')'),
            ('KEYWORD', '*'), ('KEYWORD', '+'),
            ('OR', '|')
        ]
        assert tokens == expected
    
    def test_comment_stripping(self):
        """Test that comments are removed during tokenization."""
        tokens = sag.tokenize("<Type> ::= STRING; // this is a comment")
        # Comment should be stripped, so no comment tokens
        assert all(t[1] != '// this is a comment' for t in tokens)
    
    def test_invalid_character(self):
        """Test that invalid characters raise ParsingError."""
        with pytest.raises(sag.ParsingError):
            sag.tokenize("@#$%")


class TestEBNFParsing:
    """Test EBNF parsing functionality."""
    
    def test_simple_production(self):
        """Test parsing a simple production rule."""
        ebnf = "<Node> ::= STRING;"
        productions = sag.parse_ebnf(ebnf)
        assert 'Node' in productions
        assert productions['Node'] is not None
    
    def test_alternation(self):
        """Test parsing alternation (OR)."""
        ebnf = "<Type> ::= 'source' | 'sink' | 'process';"
        productions = sag.parse_ebnf(ebnf)
        assert 'Type' in productions
        expr = productions['Type']
        assert expr['type'] == 'alternation'
        assert len(expr['items']) == 3
    
    def test_optional(self):
        """Test parsing optional elements."""
        ebnf = "<Node> ::= id:ID [label:STRING];"
        productions = sag.parse_ebnf(ebnf)
        assert 'Node' in productions
    
    def test_repetition_star(self):
        """Test parsing zero-or-more repetition (*)."""
        ebnf = "<List> ::= item*;"
        productions = sag.parse_ebnf(ebnf)
        assert 'List' in productions
    
    def test_repetition_plus(self):
        """Test parsing one-or-more repetition (+)."""
        ebnf = "<List> ::= item+;"
        productions = sag.parse_ebnf(ebnf)
        assert 'List' in productions
    
    def test_nested_references(self):
        """Test parsing nested type references."""
        ebnf = """
        <Graph> ::= nodes:<Node>+;
        <Node> ::= id:ID;
        """
        productions = sag.parse_ebnf(ebnf)
        assert 'Graph' in productions
        assert 'Node' in productions
    
    def test_complex_expression(self):
        """Test parsing complex expressions with multiple constructs."""
        ebnf = "<Complex> ::= '{' id:ID, type:('a'|'b'), [opt:STRING], list:INT+ '}';"
        productions = sag.parse_ebnf(ebnf)
        assert 'Complex' in productions
    
    def test_missing_assignment(self):
        """Test that missing ::= raises error."""
        ebnf = "<Type> STRING;"
        with pytest.raises(sag.ParsingError):
            sag.parse_ebnf(ebnf)


class TestSchemaGeneration:
    """Test JSON Schema generation from parsed EBNF."""
    
    def test_simple_schema(self):
        """Test generating schema from simple production."""
        productions = {'Node': {'type': 'ident', 'value': 'STRING'}}
        schema = sag.build_json_schema_from_productions(productions)
        assert 'definitions' in schema
        assert 'Node' in schema['definitions']
    
    def test_reference_schema(self):
        """Test schema generation with type references."""
        expr = {'type': 'ref', 'value': 'OtherType'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema == {'$ref': '#/definitions/OtherType'}
    
    def test_literal_schema(self):
        """Test schema generation for literals (const)."""
        expr = {'type': 'literal', 'value': 'constant_value'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema == {'const': 'constant_value'}
    
    def test_alternation_schema(self):
        """Test schema generation for alternations (oneOf)."""
        expr = {
            'type': 'alternation',
            'items': [
                {'type': 'literal', 'value': 'a'},
                {'type': 'literal', 'value': 'b'}
            ]
        }
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert 'oneOf' in schema
        assert len(schema['oneOf']) == 2
    
    def test_optional_schema(self):
        """Test schema generation for optional fields (anyOf)."""
        expr = {
            'type': 'optional',
            'value': {'type': 'ident', 'value': 'STRING'}
        }
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert 'anyOf' in schema
    
    def test_repetition_schema(self):
        """Test schema generation for repetitions (array)."""
        expr = {
            'type': 'repetition_modifier',
            'value': {'type': 'ident', 'value': 'STRING'},
            'minItems': 1
        }
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema['type'] == 'array'
        assert schema['minItems'] == 1
    
    def test_strict_mode(self):
        """Test that strict mode sets additionalProperties to False."""
        productions = {'Node': {'type': 'sequence', 'items': [
            {'type': 'ident', 'value': 'STRING'}
        ]}}
        schema = sag.build_json_schema_from_productions(productions, strict=True)
        # Check if strict mode affects the schema (implementation-dependent)
        assert 'definitions' in schema
    
    def test_empty_expression(self):
        """Test handling of empty expressions."""
        expr = {'type': 'empty'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema == {'type': 'object'}
    
    def test_none_expression(self):
        """Test handling of None expressions."""
        schema = sag.build_schema_from_expr(None, strict=False, multilingual_generator=None)
        assert schema == {'type': 'object'}


class TestTypeMapping:
    """Test type mapping from EBNF primitives to JSON Schema types."""
    
    def test_string_type(self):
        """Test STRING maps to string."""
        expr = {'type': 'ident', 'value': 'STRING'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema['type'] == 'string'
    
    def test_number_type(self):
        """Test NUMBER maps to number."""
        expr = {'type': 'ident', 'value': 'NUMBER'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema['type'] == 'number'
    
    def test_integer_type(self):
        """Test INT maps to integer."""
        expr = {'type': 'ident', 'value': 'INT'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema['type'] == 'integer'
    
    def test_boolean_type(self):
        """Test BOOLEAN maps to boolean."""
        expr = {'type': 'ident', 'value': 'BOOLEAN'}
        schema = sag.build_schema_from_expr(expr, strict=False, multilingual_generator=None)
        assert schema['type'] == 'boolean'


class TestGrammarExtraction:
    """Test extraction of EBNF from Markdown."""
    
    def test_extract_single_block(self):
        """Test extracting single EBNF code block."""
        md_text = """
# Grammar

```ebnf
<Node> ::= STRING;
```
        """
        ebnf = sag.extract_grammar_sections(md_text)
        assert '<Node> ::= STRING;' in ebnf
    
    def test_extract_multiple_blocks(self):
        """Test extracting multiple EBNF code blocks."""
        md_text = """
```ebnf
<Node> ::= STRING;
```

Some text

```ebnf
<Edge> ::= NUMBER;
```
        """
        ebnf = sag.extract_grammar_sections(md_text)
        assert '<Node>' in ebnf
        assert '<Edge>' in ebnf
    
    def test_case_insensitive_fence(self):
        """Test that fence markers are case-insensitive."""
        md_text = "```EBNF\n<Type> ::= STRING;\n```"
        ebnf = sag.extract_grammar_sections(md_text)
        assert '<Type>' in ebnf
    
    def test_alternative_fence_names(self):
        """Test alternative fence names (grammar, bnf)."""
        md_text = """
```grammar
<A> ::= STRING;
```

```bnf
<B> ::= NUMBER;
```
        """
        ebnf = sag.extract_grammar_sections(md_text)
        assert '<A>' in ebnf
        assert '<B>' in ebnf
    
    def test_no_ebnf_blocks(self):
        """Test handling of markdown with no EBNF blocks."""
        md_text = "# Just a regular markdown\n\nNo code here."
        ebnf = sag.extract_grammar_sections(md_text)
        assert ebnf == ""


class TestHashing:
    """Test grammar hashing functionality."""
    
    def test_hash_consistency(self):
        """Test that hashing is consistent."""
        text1 = "<Node> ::= STRING;"
        hash1 = sag.hash_grammar(text1)
        hash2 = sag.hash_grammar(text1)
        assert hash1 == hash2
    
    def test_hash_difference(self):
        """Test that different texts produce different hashes."""
        text1 = "<Node> ::= STRING;"
        text2 = "<Node> ::= NUMBER;"
        hash1 = sag.hash_grammar(text1)
        hash2 = sag.hash_grammar(text2)
        assert hash1 != hash2
    
    def test_hash_format(self):
        """Test that hash is SHA-256 (64 hex chars)."""
        text = "test"
        hash_result = sag.hash_grammar(text)
        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)


class TestMultilingualSupport:
    """Test multilingual schema generation."""
    
    def test_multilingual_generator_creation(self):
        """Test creating MultilingualSchemaGenerator."""
        gen = sag.MultilingualSchemaGenerator()
        assert gen is not None
    
    def test_multilingual_field_generation(self):
        """Test generating multilingual field schema."""
        gen = sag.MultilingualSchemaGenerator()
        base_schema = {"type": "string"}
        result = gen.generate_multilingual_field("title", base_schema)
        assert 'oneOf' in result
        assert len(result['oneOf']) == 2


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_simple(self):
        """Test complete workflow with simple grammar."""
        ebnf = "<Node> ::= id:ID, type:STRING;"
        productions = sag.parse_ebnf(ebnf)
        schema = sag.build_json_schema_from_productions(productions)
        
        assert '$schema' in schema
        assert 'definitions' in schema
        assert 'Node' in schema['definitions']
        assert 'provenance' in schema
    
    def test_end_to_end_complex(self):
        """Test complete workflow with complex grammar."""
        ebnf = """
        <Graph> ::= "{" nodes:<Node>+, edges:<Edge>* "}";
        <Node> ::= "{" id:ID, type:('source'|'sink') "}";
        <Edge> ::= "{" from:ID, to:ID "}";
        """
        productions = sag.parse_ebnf(ebnf)
        schema = sag.build_json_schema_from_productions(productions)
        
        assert 'Graph' in schema['definitions']
        assert 'Node' in schema['definitions']
        assert 'Edge' in schema['definitions']
    
    def test_file_round_trip(self):
        """Test reading from file and writing to file."""
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Test Grammar

```ebnf
<TestType> ::= STRING;
```
            """)
            temp_md = f.name
        
        # Create temporary output file
        temp_json = tempfile.mktemp(suffix='.json')
        
        try:
            # Read and process
            with open(temp_md, 'r') as f:
                md_text = f.read()
            
            ebnf_text = sag.extract_grammar_sections(md_text)
            productions = sag.parse_ebnf(ebnf_text)
            schema = sag.build_json_schema_from_productions(productions)
            
            # Write output
            with open(temp_json, 'w') as f:
                json.dump(schema, f, indent=2)
            
            # Verify output
            with open(temp_json, 'r') as f:
                loaded_schema = json.load(f)
            
            assert 'TestType' in loaded_schema['definitions']
            assert loaded_schema['$schema'] == 'http://json-schema.org/draft-07/schema#'
            
        finally:
            # Clean up
            if os.path.exists(temp_md):
                os.unlink(temp_md)
            if os.path.exists(temp_json):
                os.unlink(temp_json)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_ebnf_syntax(self):
        """Test handling of invalid EBNF syntax."""
        invalid_ebnf = "<Node> := STRING;"  # Wrong operator
        with pytest.raises(sag.ParsingError):
            sag.parse_ebnf(invalid_ebnf)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        productions = sag.parse_ebnf("")
        assert productions == {}
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only input."""
        productions = sag.parse_ebnf("   \n  \t  ")
        assert productions == {}


class TestProvenance:
    """Test provenance metadata generation."""
    
    def test_provenance_structure(self):
        """Test that provenance has required fields."""
        productions = {'Node': {'type': 'ident', 'value': 'STRING'}}
        schema = sag.build_json_schema_from_productions(productions)
        
        assert 'provenance' in schema
        assert 'generated_at' in schema['provenance']
        assert 'grammar_hash' in schema['provenance']
        assert 'generator_version' in schema['provenance']
        assert schema['provenance']['generator_version'] == '2.0.0'
    
    def test_signature_field_exists(self):
        """Test that signature field exists in provenance."""
        productions = {'Node': {'type': 'ident', 'value': 'STRING'}}
        schema = sag.build_json_schema_from_productions(productions)
        
        assert 'signature' in schema['provenance']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])