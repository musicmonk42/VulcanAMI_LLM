"""
Comprehensive test suite for parsing.py

Tests all components:
- Lexer (tokenization)
- Parser (AST construction)
- CNF conversion
- Skolemization
- Formula utilities
- End-to-end parsing

Run with: pytest src/vulcan/tests/test_symbolic_parsing.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from vulcan.reasoning.symbolic.core import (Clause, Constant, Function,
                                            Literal, Variable)
from vulcan.reasoning.symbolic.parsing import (ASTConverter, ASTNode,
                                               ClauseExtractor, CNFConverter,
                                               FormulaBuilder, FormulaParser,
                                               FormulaUtils, Lexer, NodeType,
                                               Parser, PrenexConverter,
                                               Skolemizer, Token, TokenType,
                                               VariableRenamer)

# ============================================================================
# LEXER TESTS
# ============================================================================


class TestLexer:
    """Test the Lexer component."""

    def test_lexer_simple_predicate(self):
        """Test tokenizing simple predicate."""
        lexer = Lexer("P(x)")
        tokens = lexer.tokenize()

        assert len(tokens) == 5  # P, (, x, ), EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "P"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "x"
        assert tokens[3].type == TokenType.RPAREN
        assert tokens[4].type == TokenType.EOF

    def test_lexer_logical_operators(self):
        """Test tokenizing logical operators."""
        # Test NOT
        lexer = Lexer("~P")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NOT

        lexer = Lexer("¬P")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NOT

        # Test AND
        lexer = Lexer("P & Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.AND

        lexer = Lexer("P ∧ Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.AND

        # Test OR
        lexer = Lexer("P | Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.OR

        lexer = Lexer("P ∨ Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.OR

        # Test IMPLIES
        lexer = Lexer("P -> Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.IMPLIES

        lexer = Lexer("P => Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.IMPLIES

        lexer = Lexer("P → Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.IMPLIES

        # Test IFF
        lexer = Lexer("P <=> Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.IFF

        lexer = Lexer("P ↔ Q")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.IFF

    def test_lexer_quantifiers(self):
        """Test tokenizing quantifiers."""
        # Test FORALL
        lexer = Lexer("forall X P(X)")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.FORALL

        lexer = Lexer("∀X P(X)")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.FORALL

        # Test EXISTS
        lexer = Lexer("exists X P(X)")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.EXISTS

        lexer = Lexer("∃X P(X)")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.EXISTS

    def test_lexer_complex_formula(self):
        """Test tokenizing complex formula."""
        lexer = Lexer("∀X (human(X) -> mortal(X))")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.FORALL
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "X"
        assert tokens[2].type == TokenType.LPAREN
        assert tokens[3].type == TokenType.IDENTIFIER
        assert tokens[3].value == "human"
        assert tokens[7].type == TokenType.IMPLIES

    def test_lexer_whitespace_handling(self):
        """Test that whitespace is properly ignored."""
        lexer1 = Lexer("P(x)")
        lexer2 = Lexer("P  (  x  )")
        lexer3 = Lexer("P\n(\n x\n )")

        tokens1 = lexer1.tokenize()
        tokens2 = lexer2.tokenize()
        tokens3 = lexer3.tokenize()

        # Should all produce same tokens (ignoring position info)
        assert len(tokens1) == len(tokens2) == len(tokens3)
        for t1, t2, t3 in zip(tokens1, tokens2, tokens3):
            assert t1.type == t2.type == t3.type

    def test_lexer_error_handling(self):
        """Test lexer error handling for invalid input."""
        # Invalid characters should raise SyntaxError
        lexer = Lexer("P(x) @ Q")
        with pytest.raises(SyntaxError):
            lexer.tokenize()


# ============================================================================
# PARSER TESTS
# ============================================================================


class TestParser:
    """Test the Parser component."""

    def test_parser_simple_predicate(self):
        """Test parsing simple predicate."""
        lexer = Lexer("P(x)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.PREDICATE
        assert ast.value == "P"
        assert len(ast.children) == 1
        assert ast.children[0].node_type == NodeType.VARIABLE
        assert ast.children[0].value == "x"

    def test_parser_propositional_variable(self):
        """Test parsing propositional variable (predicate with no args)."""
        lexer = Lexer("P")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.PREDICATE
        assert ast.value == "P"
        assert len(ast.children) == 0

    def test_parser_multiple_arguments(self):
        """Test parsing predicate with multiple arguments."""
        lexer = Lexer("loves(john, mary)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.PREDICATE
        assert ast.value == "loves"
        assert len(ast.children) == 2
        assert ast.children[0].value == "john"
        assert ast.children[1].value == "mary"

    def test_parser_function_term(self):
        """Test parsing function in term position."""
        lexer = Lexer("P(f(x))")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.PREDICATE
        assert len(ast.children) == 1
        assert ast.children[0].node_type == NodeType.FUNCTION
        assert ast.children[0].value == "f"
        assert ast.children[0].children[0].value == "x"

    def test_parser_nested_functions(self):
        """Test parsing nested functions."""
        lexer = Lexer("P(f(g(x)))")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.PREDICATE
        func1 = ast.children[0]
        assert func1.node_type == NodeType.FUNCTION
        assert func1.value == "f"

        func2 = func1.children[0]
        assert func2.node_type == NodeType.FUNCTION
        assert func2.value == "g"

        var = func2.children[0]
        assert var.node_type == NodeType.VARIABLE
        assert var.value == "x"

    def test_parser_negation(self):
        """Test parsing negation."""
        lexer = Lexer("~P(x)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.NOT
        assert ast.children[0].node_type == NodeType.PREDICATE

    def test_parser_double_negation(self):
        """Test parsing double negation."""
        lexer = Lexer("~~P(x)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.NOT
        assert ast.children[0].node_type == NodeType.NOT
        assert ast.children[0].children[0].node_type == NodeType.PREDICATE

    def test_parser_conjunction(self):
        """Test parsing conjunction."""
        lexer = Lexer("P(x) & Q(y)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.AND
        assert ast.children[0].node_type == NodeType.PREDICATE
        assert ast.children[0].value == "P"
        assert ast.children[1].node_type == NodeType.PREDICATE
        assert ast.children[1].value == "Q"

    def test_parser_disjunction(self):
        """Test parsing disjunction."""
        lexer = Lexer("P(x) | Q(y)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.OR
        assert ast.children[0].value == "P"
        assert ast.children[1].value == "Q"

    def test_parser_implication(self):
        """Test parsing implication."""
        lexer = Lexer("P(x) -> Q(y)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.IMPLIES
        assert ast.children[0].value == "P"
        assert ast.children[1].value == "Q"

    def test_parser_biconditional(self):
        """Test parsing biconditional."""
        lexer = Lexer("P(x) <=> Q(y)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.IFF
        assert ast.children[0].value == "P"
        assert ast.children[1].value == "Q"

    def test_parser_operator_precedence(self):
        """Test operator precedence: NOT > AND > OR > IMPLIES > IFF."""
        # ~P & Q should parse as (~P) & Q
        lexer = Lexer("~P & Q")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.AND
        assert ast.children[0].node_type == NodeType.NOT
        assert ast.children[1].node_type == NodeType.PREDICATE

        # P & Q | R should parse as (P & Q) | R
        lexer = Lexer("P & Q | R")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.OR
        assert ast.children[0].node_type == NodeType.AND
        assert ast.children[1].node_type == NodeType.PREDICATE

        # P | Q -> R should parse as (P | Q) -> R
        lexer = Lexer("P | Q -> R")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.IMPLIES
        assert ast.children[0].node_type == NodeType.OR

    def test_parser_parentheses(self):
        """Test parsing with parentheses to override precedence."""
        # P & (Q | R)
        lexer = Lexer("P & (Q | R)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.AND
        assert ast.children[0].node_type == NodeType.PREDICATE
        assert ast.children[1].node_type == NodeType.OR

    def test_parser_universal_quantifier(self):
        """Test parsing universal quantifier."""
        lexer = Lexer("∀X P(X)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.FORALL
        assert len(ast.metadata["variables"]) == 1
        assert ast.metadata["variables"][0].value == "X"
        assert ast.children[0].node_type == NodeType.PREDICATE

    def test_parser_existential_quantifier(self):
        """Test parsing existential quantifier."""
        lexer = Lexer("∃Y Q(Y)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.EXISTS
        assert len(ast.metadata["variables"]) == 1
        assert ast.metadata["variables"][0].value == "Y"
        assert ast.children[0].node_type == NodeType.PREDICATE

    def test_parser_multiple_quantified_variables(self):
        """Test parsing quantifier with multiple variables."""
        lexer = Lexer("∀X Y Z P(X, Y, Z)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.FORALL
        assert len(ast.metadata["variables"]) == 3
        assert ast.metadata["variables"][0].value == "X"
        assert ast.metadata["variables"][1].value == "Y"
        assert ast.metadata["variables"][2].value == "Z"

    def test_parser_nested_quantifiers(self):
        """Test parsing nested quantifiers."""
        lexer = Lexer("∀X ∃Y loves(X, Y)")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.FORALL
        assert ast.metadata["variables"][0].value == "X"

        inner = ast.children[0]
        assert inner.node_type == NodeType.EXISTS
        assert inner.metadata["variables"][0].value == "Y"

        predicate = inner.children[0]
        assert predicate.node_type == NodeType.PREDICATE
        assert predicate.value == "loves"

    def test_parser_complex_formula(self):
        """Test parsing complex formula with multiple constructs."""
        lexer = Lexer("∀X (human(X) -> mortal(X))")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.FORALL
        implication = ast.children[0]
        assert implication.node_type == NodeType.IMPLIES
        assert implication.children[0].value == "human"
        assert implication.children[1].value == "mortal"

    def test_parser_implication_associativity(self):
        """Test that implication is right-associative."""
        # A -> B -> C should parse as A -> (B -> C)
        lexer = Lexer("P -> Q -> R")
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        assert ast.node_type == NodeType.IMPLIES
        assert ast.children[0].value == "P"
        assert ast.children[1].node_type == NodeType.IMPLIES
        assert ast.children[1].children[0].value == "Q"
        assert ast.children[1].children[1].value == "R"

    def test_parser_error_missing_rparen(self):
        """Test parser error on missing right parenthesis."""
        lexer = Lexer("P(x")
        tokens = lexer.tokenize()
        parser = Parser(tokens)

        with pytest.raises(SyntaxError):
            parser.parse()

    def test_parser_error_unexpected_token(self):
        """Test parser error on unexpected token."""
        lexer = Lexer("P(x) Q(y)")  # Missing operator
        tokens = lexer.tokenize()
        parser = Parser(tokens)

        with pytest.raises(SyntaxError):
            parser.parse()


# ============================================================================
# FORMULA UTILS TESTS
# ============================================================================


class TestFormulaUtils:
    """Test FormulaUtils helper functions."""

    def test_get_free_variables_simple(self):
        """Test getting free variables from simple formula."""
        # P(X, y) has X as free variable (y is constant)
        pred = ASTNode(
            NodeType.PREDICATE,
            value="P",
            children=[
                ASTNode(NodeType.VARIABLE, value="X"),
                ASTNode(NodeType.CONSTANT, value="y"),
            ],
        )

        free_vars = FormulaUtils.get_free_variables(pred)
        assert free_vars == {"X"}

    def test_get_free_variables_quantified(self):
        """Test that quantified variables are not free."""
        # ∀X P(X, Y) has Y as free variable, not X
        body = ASTNode(
            NodeType.PREDICATE,
            value="P",
            children=[
                ASTNode(NodeType.VARIABLE, value="X"),
                ASTNode(NodeType.VARIABLE, value="Y"),
            ],
        )
        formula = ASTNode(
            NodeType.FORALL,
            children=[body],
            metadata={"variables": [ASTNode(NodeType.VARIABLE, value="X")]},
        )

        free_vars = FormulaUtils.get_free_variables(formula)
        assert free_vars == {"Y"}
        assert "X" not in free_vars

    def test_get_all_variables(self):
        """Test getting all variables (free and bound)."""
        # ∀X P(X, Y)
        body = ASTNode(
            NodeType.PREDICATE,
            value="P",
            children=[
                ASTNode(NodeType.VARIABLE, value="X"),
                ASTNode(NodeType.VARIABLE, value="Y"),
            ],
        )
        formula = ASTNode(
            NodeType.FORALL,
            children=[body],
            metadata={"variables": [ASTNode(NodeType.VARIABLE, value="X")]},
        )

        all_vars = FormulaUtils.get_all_variables(formula)
        assert all_vars == {"X", "Y"}

    def test_substitute_variable(self):
        """Test variable substitution."""
        # P(X) with X -> a becomes P(a)
        original = ASTNode(
            NodeType.PREDICATE,
            value="P",
            children=[ASTNode(NodeType.VARIABLE, value="X")],
        )

        new_term = ASTNode(NodeType.CONSTANT, value="a")
        result = FormulaUtils.substitute_variable(original, "X", new_term)

        assert result.children[0].node_type == NodeType.CONSTANT
        assert result.children[0].value == "a"

    def test_substitute_respects_binding(self):
        """Test that substitution respects variable binding."""
        # ∀X P(X) with X -> a should NOT substitute (X is bound)
        body = ASTNode(
            NodeType.PREDICATE,
            value="P",
            children=[ASTNode(NodeType.VARIABLE, value="X")],
        )
        formula = ASTNode(
            NodeType.FORALL,
            children=[body],
            metadata={"variables": [ASTNode(NodeType.VARIABLE, value="X")]},
        )

        new_term = ASTNode(NodeType.CONSTANT, value="a")
        result = FormulaUtils.substitute_variable(formula, "X", new_term)

        # X should still be a variable
        assert result.children[0].children[0].node_type == NodeType.VARIABLE
        assert result.children[0].children[0].value == "X"

    def test_is_literal(self):
        """Test literal detection."""
        # P(x) is a literal
        pred = ASTNode(
            NodeType.PREDICATE,
            value="P",
            children=[ASTNode(NodeType.VARIABLE, value="x")],
        )
        assert FormulaUtils.is_literal(pred)

        # ~P(x) is a literal
        neg_pred = ASTNode(NodeType.NOT, children=[pred])
        assert FormulaUtils.is_literal(neg_pred)

        # P(x) & Q(y) is not a literal
        conj = ASTNode(NodeType.AND, children=[pred, pred.copy()])
        assert not FormulaUtils.is_literal(conj)

    def test_is_clause(self):
        """Test clause detection."""
        # P(x) is a clause
        lit1 = ASTNode(NodeType.PREDICATE, value="P", children=[])
        assert FormulaUtils.is_clause(lit1)

        # P | Q is a clause
        lit2 = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        clause = ASTNode(NodeType.OR, children=[lit1, lit2])
        assert FormulaUtils.is_clause(clause)

        # P & Q is not a clause (uses AND)
        not_clause = ASTNode(NodeType.AND, children=[lit1, lit2])
        assert not FormulaUtils.is_clause(not_clause)


# ============================================================================
# CNF CONVERSION TESTS
# ============================================================================


class TestCNFConverter:
    """Test CNF conversion."""

    def test_eliminate_implications_simple(self):
        """Test eliminating simple implication."""
        converter = CNFConverter()

        # P -> Q becomes ~P | Q
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        impl = ASTNode(NodeType.IMPLIES, children=[p, q])

        result = converter._eliminate_implications(impl)

        assert result.node_type == NodeType.OR
        assert result.children[0].node_type == NodeType.NOT
        assert result.children[0].children[0].value == "P"
        assert result.children[1].value == "Q"

    def test_eliminate_biconditional(self):
        """Test eliminating biconditional."""
        converter = CNFConverter()

        # P <-> Q becomes (~P | Q) & (~Q | P)
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        iff = ASTNode(NodeType.IFF, children=[p, q])

        result = converter._eliminate_implications(iff)

        assert result.node_type == NodeType.AND
        # Both children should be OR nodes
        assert result.children[0].node_type == NodeType.OR
        assert result.children[1].node_type == NodeType.OR

    def test_move_negations_inward_double_neg(self):
        """Test moving negations inward with double negation."""
        converter = CNFConverter()

        # ~~P becomes P
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        double_neg = ASTNode(
            NodeType.NOT, children=[ASTNode(NodeType.NOT, children=[p])]
        )

        result = converter._move_negations_inward(double_neg)

        assert result.node_type == NodeType.PREDICATE
        assert result.value == "P"

    def test_move_negations_inward_de_morgan_and(self):
        """Test De Morgan's law for AND."""
        converter = CNFConverter()

        # ~(P & Q) becomes ~P | ~Q
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        conj = ASTNode(NodeType.AND, children=[p, q])
        neg_conj = ASTNode(NodeType.NOT, children=[conj])

        result = converter._move_negations_inward(neg_conj)

        assert result.node_type == NodeType.OR
        assert result.children[0].node_type == NodeType.NOT
        assert result.children[1].node_type == NodeType.NOT

    def test_move_negations_inward_de_morgan_or(self):
        """Test De Morgan's law for OR."""
        converter = CNFConverter()

        # ~(P | Q) becomes ~P & ~Q
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        disj = ASTNode(NodeType.OR, children=[p, q])
        neg_disj = ASTNode(NodeType.NOT, children=[disj])

        result = converter._move_negations_inward(neg_disj)

        assert result.node_type == NodeType.AND
        assert result.children[0].node_type == NodeType.NOT
        assert result.children[1].node_type == NodeType.NOT

    def test_distribute_or_over_and_left(self):
        """Test distributing OR over AND on left side."""
        converter = CNFConverter()

        # (P & Q) | R becomes (P | R) & (Q | R)
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        r = ASTNode(NodeType.PREDICATE, value="R", children=[])

        pq = ASTNode(NodeType.AND, children=[p, q])
        formula = ASTNode(NodeType.OR, children=[pq, r])

        result = converter._distribute_or_over_and(formula)

        assert result.node_type == NodeType.AND
        assert result.children[0].node_type == NodeType.OR
        assert result.children[1].node_type == NodeType.OR

    def test_distribute_or_over_and_right(self):
        """Test distributing OR over AND on right side."""
        converter = CNFConverter()

        # P | (Q & R) becomes (P | Q) & (P | R)
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        r = ASTNode(NodeType.PREDICATE, value="R", children=[])

        qr = ASTNode(NodeType.AND, children=[q, r])
        formula = ASTNode(NodeType.OR, children=[p, qr])

        result = converter._distribute_or_over_and(formula)

        assert result.node_type == NodeType.AND
        assert result.children[0].node_type == NodeType.OR
        assert result.children[1].node_type == NodeType.OR

    def test_cnf_conversion_full(self):
        """Test full CNF conversion pipeline."""
        converter = CNFConverter()

        # P -> Q
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        formula = ASTNode(NodeType.IMPLIES, children=[p, q])

        result = converter.to_cnf(formula)

        # Should become ~P | Q (which is already in CNF)
        assert result.node_type == NodeType.OR
        assert result.children[0].node_type == NodeType.NOT
        assert result.children[1].node_type == NodeType.PREDICATE

    def test_tautology_detection(self):
        """Test tautology detection in simplification."""
        converter = CNFConverter()

        # P | ~P is a tautology
        p = ASTNode(NodeType.PREDICATE, value="P", children=[])
        not_p = ASTNode(NodeType.NOT, children=[p.copy()])

        literals = [p, not_p]
        assert converter._is_tautology(literals)

        # P | Q is not a tautology
        q = ASTNode(NodeType.PREDICATE, value="Q", children=[])
        literals = [p, q]
        assert not converter._is_tautology(literals)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test end-to-end parsing integration."""

    def test_parse_simple_formula(self):
        """Test parsing simple formula end-to-end."""
        parser = FormulaParser()
        ast = parser.from_string("P(a)")

        assert ast.node_type == NodeType.PREDICATE
        assert ast.value == "P"
        assert ast.children[0].value == "a"

    def test_parse_complex_formula(self):
        """Test parsing complex formula."""
        parser = FormulaParser()
        ast = parser.from_string("∀X (human(X) -> mortal(X))")

        assert ast.node_type == NodeType.FORALL
        assert ast.children[0].node_type == NodeType.IMPLIES

    def test_parse_to_cnf(self):
        """Test parsing and converting to CNF."""
        parser = FormulaParser()
        ast = parser.from_string("P -> Q")
        cnf = parser.parse_to_cnf(ast)

        # P -> Q should become ~P | Q
        assert cnf.node_type == NodeType.OR

    def test_parse_to_clauses(self):
        """Test parsing and extracting clauses."""
        parser = FormulaParser()
        ast = parser.from_string("P(a) | Q(b)")
        clauses = parser.parse_to_clauses(ast)

        assert len(clauses) == 1
        assert len(clauses[0].literals) == 2

    def test_socrates_syllogism(self):
        """Test classic Socrates syllogism."""
        parser = FormulaParser()

        # All humans are mortal
        rule1 = parser.from_string("∀X (human(X) -> mortal(X))")
        assert rule1.node_type == NodeType.FORALL

        # Socrates is human
        rule2 = parser.from_string("human(socrates)")
        assert rule2.node_type == NodeType.PREDICATE

        # Therefore Socrates is mortal
        goal = parser.from_string("mortal(socrates)")
        assert goal.node_type == NodeType.PREDICATE


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_deeply_nested_formula(self):
        """Test parsing deeply nested formula."""
        parser = FormulaParser()
        # ((((P))))
        ast = parser.from_string("((((P))))")
        assert ast.node_type == NodeType.PREDICATE
        assert ast.value == "P"

    def test_many_arguments(self):
        """Test predicate with many arguments."""
        parser = FormulaParser()
        ast = parser.from_string("pred(a, b, c, d, e, f, g)")

        assert ast.node_type == NodeType.PREDICATE
        assert len(ast.children) == 7

    def test_variable_constant_distinction(self):
        """Test that variables and constants are distinguished correctly."""
        parser = FormulaParser()
        ast = parser.from_string("P(X, y)")

        # X (uppercase) should be variable
        assert ast.children[0].node_type == NodeType.VARIABLE
        assert ast.children[0].value == "X"

        # y (lowercase) should be constant
        assert ast.children[1].node_type == NodeType.CONSTANT
        assert ast.children[1].value == "y"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
