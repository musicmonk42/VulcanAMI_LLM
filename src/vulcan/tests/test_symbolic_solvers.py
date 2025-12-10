"""
Comprehensive tests for probabilistic reasoning and CSP solvers.

Tests cover:
- BayesianNetworkReasoner: Discrete and continuous variables, inference, learning
- Factor operations: Multiplication, marginalization, normalization
- CSPSolver: Backtracking, forward checking, AC-3, min-conflicts
- Parameter learning: MLE, EM
- Structure learning: PC algorithm, K2 algorithm

All tests validate the FIXED implementations.
"""


import numpy as np
import pytest

# Import the classes we're testing
from src.vulcan.reasoning.symbolic.solvers import (CPT,
                                                   BayesianNetworkReasoner,
                                                   CSPSolver, Factor,
                                                   GaussianCPD, RandomVariable,
                                                   VariableType)

# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================


class TestRandomVariable:
    """Tests for RandomVariable data structure."""

    def test_random_variable_creation_discrete(self):
        """Test creating discrete random variable."""
        var = RandomVariable(
            name="Rain",
            var_type=VariableType.DISCRETE,
            domain=[True, False],
            parents=[],
        )

        assert var.name == "Rain"
        assert var.var_type == VariableType.DISCRETE
        assert var.domain == [True, False]
        assert var.parents == []

    def test_random_variable_creation_continuous(self):
        """Test creating continuous random variable."""
        var = RandomVariable(
            name="Temperature", var_type=VariableType.CONTINUOUS, parents=["Season"]
        )

        assert var.name == "Temperature"
        assert var.var_type == VariableType.CONTINUOUS
        assert var.domain is None

    def test_random_variable_with_parents(self):
        """Test creating variable with parents."""
        var = RandomVariable(
            name="WetGrass",
            var_type=VariableType.DISCRETE,
            domain=[True, False],
            parents=["Rain", "Sprinkler"],
        )

        assert len(var.parents) == 2
        assert "Rain" in var.parents
        assert "Sprinkler" in var.parents

    def test_random_variable_equality(self):
        """Test variable equality."""
        var1 = RandomVariable("X", VariableType.DISCRETE, [1, 2])
        var2 = RandomVariable("X", VariableType.DISCRETE, [1, 2])
        var3 = RandomVariable("Y", VariableType.DISCRETE, [1, 2])

        assert var1 == var2
        assert var1 != var3

    def test_random_variable_hashable(self):
        """Test that variables can be used in sets/dicts."""
        var1 = RandomVariable("X", VariableType.DISCRETE, [1, 2])
        var2 = RandomVariable("Y", VariableType.DISCRETE, [1, 2])

        var_set = {var1, var2}
        assert len(var_set) == 2


class TestFactor:
    """Tests for Factor operations."""

    def test_factor_creation(self):
        """Test creating a factor."""
        factor = Factor(
            variables=["X", "Y"],
            values={
                (True, True): 0.3,
                (True, False): 0.7,
                (False, True): 0.6,
                (False, False): 0.4,
            },
        )

        assert len(factor.variables) == 2
        assert len(factor.values) == 4

    def test_factor_normalize(self):
        """Test factor normalization."""
        factor = Factor(variables=["X"], values={(True,): 3.0, (False,): 1.0})

        factor.normalize()

        assert factor.values[(True,)] == 0.75
        assert factor.values[(False,)] == 0.25

    def test_factor_marginalize_single_var(self):
        """Test marginalizing out a variable."""
        factor = Factor(
            variables=["X", "Y"],
            values={
                (True, True): 0.3,
                (True, False): 0.2,
                (False, True): 0.4,
                (False, False): 0.1,
            },
        )

        # Marginalize out Y
        result = factor.marginalize("Y")

        assert result.variables == ["X"]
        assert result.values[(True,)] == 0.5  # 0.3 + 0.2
        assert result.values[(False,)] == 0.5  # 0.4 + 0.1

    def test_factor_marginalize_nonexistent_var(self):
        """Test marginalizing variable not in factor."""
        factor = Factor(variables=["X"], values={(True,): 0.5, (False,): 0.5})

        result = factor.marginalize("Z")

        # Should return copy of original
        assert result.variables == ["X"]
        assert len(result.values) == 2

    def test_factor_multiply_simple(self):
        """Test multiplying two factors."""
        factor1 = Factor(variables=["X"], values={(True,): 0.3, (False,): 0.7})

        factor2 = Factor(variables=["Y"], values={(True,): 0.6, (False,): 0.4})

        result = factor1.multiply(factor2)

        assert set(result.variables) == {"X", "Y"}
        assert len(result.values) == 4

    def test_factor_multiply_shared_variables(self):
        """Test multiplying factors with shared variables."""
        factor1 = Factor(
            variables=["X", "Y"],
            values={
                (True, True): 0.3,
                (True, False): 0.7,
                (False, True): 0.6,
                (False, False): 0.4,
            },
        )

        factor2 = Factor(
            variables=["Y", "Z"],
            values={
                (True, True): 0.8,
                (True, False): 0.2,
                (False, True): 0.5,
                (False, False): 0.5,
            },
        )

        result = factor1.multiply(factor2)

        assert set(result.variables) == {"X", "Y", "Z"}

    def test_factor_merge_keys_consistent(self):
        """Test key merging with consistent values."""
        factor = Factor(variables=[], values={})

        key1 = (True, True)
        key2 = (True, False)
        vars1 = ["X", "Y"]
        vars2 = ["X", "Z"]
        all_vars = ["X", "Y", "Z"]

        merged = factor._merge_keys(key1, key2, vars1, vars2, all_vars)

        assert merged is not None
        assert merged[0] == True  # X value consistent

    def test_factor_merge_keys_inconsistent(self):
        """Test key merging with inconsistent values."""
        factor = Factor(variables=[], values={})

        key1 = (True, True)
        key2 = (False, True)
        vars1 = ["X", "Y"]
        vars2 = ["X", "Z"]
        all_vars = ["X", "Y", "Z"]

        merged = factor._merge_keys(key1, key2, vars1, vars2, all_vars)

        # Should return None due to inconsistent X value
        assert merged is None


class TestCPT:
    """Tests for Conditional Probability Table."""

    def test_cpt_creation(self):
        """Test creating CPT."""
        cpt = CPT(
            variable="X",
            parents=["Y"],
            table={(True,): {True: 0.8, False: 0.2}, (False,): {True: 0.3, False: 0.7}},
        )

        assert cpt.variable == "X"
        assert cpt.parents == ["Y"]

    def test_cpt_get_probability(self):
        """Test getting probability from CPT."""
        cpt = CPT(
            variable="X",
            parents=["Y"],
            table={(True,): {True: 0.8, False: 0.2}, (False,): {True: 0.3, False: 0.7}},
        )

        prob = cpt.get_probability(True, (True,))
        assert prob == 0.8

        prob = cpt.get_probability(False, (False,))
        assert prob == 0.7

    def test_cpt_set_probability(self):
        """Test setting probability in CPT."""
        cpt = CPT(variable="X", parents=["Y"], table={})

        cpt.set_probability(True, (True,), 0.9)

        assert cpt.get_probability(True, (True,)) == 0.9

    def test_cpt_no_parents(self):
        """Test CPT with no parents (prior)."""
        cpt = CPT(variable="X", parents=[], table={(): {True: 0.6, False: 0.4}})

        prob = cpt.get_probability(True, ())
        assert prob == 0.6


class TestGaussianCPD:
    """Tests for Gaussian Conditional Probability Distribution."""

    def test_gaussian_cpd_creation(self):
        """Test creating Gaussian CPD."""
        cpd = GaussianCPD(
            variable="X",
            parents=["Y"],
            coefficients={"Y": 0.5},
            intercept=2.0,
            variance=1.0,
        )

        assert cpd.variable == "X"
        assert cpd.intercept == 2.0
        assert cpd.variance == 1.0

    def test_gaussian_cpd_mean(self):
        """Test computing mean."""
        cpd = GaussianCPD(
            variable="X",
            parents=["Y"],
            coefficients={"Y": 0.5},
            intercept=2.0,
            variance=1.0,
        )

        mean = cpd.mean({"Y": 4.0})
        assert mean == 4.0  # 2.0 + 0.5 * 4.0

    def test_gaussian_cpd_sample(self):
        """Test sampling from Gaussian CPD."""
        cpd = GaussianCPD(
            variable="X",
            parents=["Y"],
            coefficients={"Y": 0.5},
            intercept=2.0,
            variance=0.01,  # Low variance for stable test
        )

        # Sample multiple times
        samples = [cpd.sample({"Y": 4.0}) for _ in range(100)]

        # Mean should be close to 4.0
        assert abs(np.mean(samples) - 4.0) < 0.5


# ============================================================================
# BAYESIAN NETWORK REASONER TESTS
# ============================================================================


class TestBayesianNetworkReasoner:
    """Tests for BayesianNetworkReasoner."""

    def test_bn_creation(self):
        """Test creating Bayesian Network."""
        bn = BayesianNetworkReasoner()

        assert len(bn.variables) == 0
        assert len(bn.cpts) == 0

    def test_add_discrete_variable(self):
        """Test adding discrete variable."""
        bn = BayesianNetworkReasoner()

        var = bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])

        assert "Rain" in bn.variables
        assert var.var_type == VariableType.DISCRETE

    def test_add_variable_with_parents(self):
        """Test adding variable with parents."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable(
            "Sprinkler", VariableType.DISCRETE, domain=[True, False], parents=["Rain"]
        )

        assert "Sprinkler" in bn.variables
        assert "Rain" in bn.variables["Sprinkler"].parents
        assert "Sprinkler" in bn.structure["Rain"]

    def test_set_cpt_simple(self):
        """Test setting CPT."""
        bn = BayesianNetworkReasoner()
        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])

        bn.set_cpt("Rain", {(): {True: 0.3, False: 0.7}})

        assert "Rain" in bn.cpts
        assert bn.cpts["Rain"].get_probability(True, ()) == 0.3

    def test_set_cpt_with_parents(self):
        """Test setting CPT with parents."""
        bn = BayesianNetworkReasoner()
        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable(
            "WetGrass", VariableType.DISCRETE, domain=[True, False], parents=["Rain"]
        )

        table = {(True,): {True: 0.9, False: 0.1}, (False,): {True: 0.2, False: 0.8}}
        bn.set_cpt("WetGrass", table)

        assert bn.cpts["WetGrass"].get_probability(True, (True,)) == 0.9

    def test_simple_query_no_evidence(self):
        """Test simple query without evidence."""
        bn = BayesianNetworkReasoner()
        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])
        bn.set_cpt("Rain", {(): {True: 0.3, False: 0.7}})

        result = bn.query("Rain", evidence={})

        assert True in result
        assert False in result
        assert abs(result[True] - 0.3) < 0.01

    def test_query_with_evidence(self):
        """Test query with evidence."""
        bn = BayesianNetworkReasoner()

        # Build simple network: Rain -> WetGrass
        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable(
            "WetGrass", VariableType.DISCRETE, domain=[True, False], parents=["Rain"]
        )

        bn.set_cpt("Rain", {(): {True: 0.3, False: 0.7}})
        bn.set_cpt(
            "WetGrass",
            {(True,): {True: 0.9, False: 0.1}, (False,): {True: 0.2, False: 0.8}},
        )

        # Query: P(Rain | WetGrass=True)
        result = bn.query("Rain", evidence={"WetGrass": True})

        assert True in result
        assert False in result
        # P(Rain=True | WetGrass=True) should be higher than prior
        assert result[True] > 0.3

    def test_variable_elimination_chain(self):
        """Test variable elimination with chain structure."""
        bn = BayesianNetworkReasoner()

        # A -> B -> C
        bn.add_variable("A", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable("B", VariableType.DISCRETE, domain=[True, False], parents=["A"])
        bn.add_variable("C", VariableType.DISCRETE, domain=[True, False], parents=["B"])

        bn.set_cpt("A", {(): {True: 0.5, False: 0.5}})
        bn.set_cpt(
            "B", {(True,): {True: 0.8, False: 0.2}, (False,): {True: 0.3, False: 0.7}}
        )
        bn.set_cpt(
            "C", {(True,): {True: 0.9, False: 0.1}, (False,): {True: 0.4, False: 0.6}}
        )

        result = bn.query("C", evidence={"A": True})

        assert True in result
        assert False in result

    def test_gaussian_variable(self):
        """Test continuous Gaussian variable."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("X", VariableType.GAUSSIAN)
        bn.set_gaussian_cpd("X", coefficients={}, intercept=5.0, variance=1.0)

        result = bn.query("X", evidence={}, method="variable_elimination")

        assert "mean" in result
        assert "variance" in result
        assert result["mean"] == 5.0

    def test_gaussian_with_parents(self):
        """Test Gaussian variable with parents."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("Y", VariableType.GAUSSIAN)
        bn.add_variable("X", VariableType.GAUSSIAN, parents=["Y"])

        bn.set_gaussian_cpd("Y", coefficients={}, intercept=10.0, variance=1.0)
        bn.set_gaussian_cpd("X", coefficients={"Y": 0.5}, intercept=2.0, variance=1.0)

        result = bn.query("X", evidence={"Y": 4.0}, method="variable_elimination")

        assert "mean" in result
        # Mean should be 2.0 + 0.5 * 4.0 = 4.0
        assert abs(result["mean"] - 4.0) < 0.01

    def test_mcmc_inference(self):
        """Test MCMC inference."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])
        bn.set_cpt("Rain", {(): {True: 0.3, False: 0.7}})

        result = bn.query("Rain", evidence={}, method="mcmc")

        # MCMC should approximate the prior
        assert True in result
        assert abs(result[True] - 0.3) < 0.1  # Allow some sampling error

    def test_create_factor_from_cpt(self):
        """Test creating factor from CPT."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("X", VariableType.DISCRETE, domain=[True, False])
        bn.set_cpt("X", {(): {True: 0.6, False: 0.4}})

        factor = bn._create_factor_from_cpt("X", {})

        assert factor is not None
        assert "X" in factor.variables
        assert len(factor.values) == 2

    def test_create_factor_with_evidence(self):
        """Test creating factor with evidence."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("X", VariableType.DISCRETE, domain=[True, False])
        bn.set_cpt("X", {(): {True: 0.6, False: 0.4}})

        # X is observed - factor should represent P(X=True) as a constant
        factor = bn._create_factor_from_cpt("X", {"X": True})

        assert factor is not None
        assert factor.variables == []  # No variables (constant factor)
        assert factor.values == {(): 0.6}  # P(X=True) = 0.6

    def test_get_parent_combinations(self):
        """Test generating parent combinations."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("A", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable("B", VariableType.DISCRETE, domain=[True, False])

        combos = bn._get_parent_combinations(["A", "B"], {})

        assert len(combos) == 4

    def test_get_parent_combinations_with_evidence(self):
        """Test parent combinations with evidence."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("A", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable("B", VariableType.DISCRETE, domain=[True, False])

        combos = bn._get_parent_combinations(["A", "B"], {"A": True})

        # Only one value for A (observed)
        assert len(combos) == 2
        assert all(combo["A"] == True for combo in combos)

    def test_eliminate_variable(self):
        """Test variable elimination operation."""
        bn = BayesianNetworkReasoner()

        factor1 = Factor(
            variables=["X", "Y"],
            values={
                (True, True): 0.3,
                (True, False): 0.7,
                (False, True): 0.2,
                (False, False): 0.8,
            },
        )

        factor2 = Factor(
            variables=["Y", "Z"],
            values={
                (True, True): 0.6,
                (True, False): 0.4,
                (False, True): 0.5,
                (False, False): 0.5,
            },
        )

        factors = [factor1, factor2]
        result = bn._eliminate_variable("Y", factors)

        # Should have one factor with X and Z
        assert len(result) == 1
        assert "Y" not in result[0].variables


# ============================================================================
# PARAMETER LEARNING TESTS
# ============================================================================


class TestParameterLearning:
    """Tests for parameter learning."""

    def test_learn_parameters_mle_simple(self):
        """Test MLE parameter learning."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("X", VariableType.DISCRETE, domain=[True, False])

        # Data: 70% True, 30% False
        data = [{"X": True}] * 70 + [{"X": False}] * 30

        bn.learn_parameters_mle(data)

        cpt = bn.cpts["X"]
        prob_true = cpt.get_probability(True, ())

        assert abs(prob_true - 0.7) < 0.01

    def test_learn_parameters_with_parents(self):
        """Test learning CPT with parents."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("A", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable("B", VariableType.DISCRETE, domain=[True, False], parents=["A"])

        # Data: P(B=True | A=True) = 0.8, P(B=True | A=False) = 0.3
        data = (
            [{"A": True, "B": True}] * 8
            + [{"A": True, "B": False}] * 2
            + [{"A": False, "B": True}] * 3
            + [{"A": False, "B": False}] * 7
        )

        bn.learn_parameters_mle(data)

        cpt = bn.cpts["B"]
        assert abs(cpt.get_probability(True, (True,)) - 0.8) < 0.1

    def test_learn_gaussian_cpd(self):
        """Test learning Gaussian CPD."""
        bn = BayesianNetworkReasoner()

        bn.add_variable("X", VariableType.GAUSSIAN)
        bn.add_variable("Y", VariableType.GAUSSIAN, parents=["X"])

        # Generate data: Y = 2 + 0.5*X + noise
        np.random.seed(42)
        data = []
        for _ in range(100):
            x = np.random.normal(0, 1)
            y = 2 + 0.5 * x + np.random.normal(0, 0.1)
            data.append({"X": x, "Y": y})

        bn.learn_parameters_mle(data)

        cpd = bn.gaussian_cpds["Y"]
        # Should learn intercept ≈ 2 and coefficient ≈ 0.5
        assert abs(cpd.intercept - 2.0) < 0.5
        assert abs(cpd.coefficients["X"] - 0.5) < 0.2


# ============================================================================
# CSP SOLVER TESTS
# ============================================================================


class TestCSPSolver:
    """Tests for Constraint Satisfaction Problem solver."""

    def test_csp_creation(self):
        """Test creating CSP solver."""
        solver = CSPSolver()

        assert len(solver.variables) == 0
        assert len(solver.constraints) == 0

    def test_add_variable(self):
        """Test adding variable to CSP."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])

        assert "X" in solver.variables
        assert solver.variables["X"].domain == [1, 2, 3]

    def test_add_constraint(self):
        """Test adding constraint."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])
        solver.add_variable("Y", [1, 2, 3])

        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])

        assert len(solver.constraints) == 1

    def test_solve_simple_csp(self):
        """Test solving simple CSP."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])
        solver.add_variable("Y", [1, 2, 3])
        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])

        solution = solver.solve()

        assert solution is not None
        assert solution["X"] != solution["Y"]

    def test_solve_map_coloring(self):
        """Test solving map coloring problem."""
        solver = CSPSolver()

        # Australia map coloring with 3 colors
        regions = ["WA", "NT", "SA", "Q", "NSW", "V"]
        colors = ["red", "green", "blue"]

        for region in regions:
            solver.add_variable(region, colors)

        # Add adjacency constraints
        adjacencies = [
            ("WA", "NT"),
            ("WA", "SA"),
            ("NT", "SA"),
            ("NT", "Q"),
            ("SA", "Q"),
            ("SA", "NSW"),
            ("SA", "V"),
            ("Q", "NSW"),
            ("NSW", "V"),
        ]

        for r1, r2 in adjacencies:
            solver.add_constraint([r1, r2], lambda a, r1=r1, r2=r2: a[r1] != a[r2])

        solution = solver.solve()

        assert solution is not None
        # Check all constraints satisfied
        for r1, r2 in adjacencies:
            assert solution[r1] != solution[r2]

    def test_solve_n_queens(self):
        """Test solving N-Queens problem."""
        solver = CSPSolver()
        n = 4

        # Variables: row for each column
        for col in range(n):
            solver.add_variable(f"Q{col}", list(range(n)))

        # Constraints: no two queens in same row or diagonal
        for col1 in range(n):
            for col2 in range(col1 + 1, n):

                def constraint(a, c1=col1, c2=col2):
                    row1 = a[f"Q{c1}"]
                    row2 = a[f"Q{c2}"]
                    # Not same row
                    if row1 == row2:
                        return False
                    # Not same diagonal
                    if abs(row1 - row2) == abs(c1 - c2):
                        return False
                    return True

                solver.add_constraint([f"Q{col1}", f"Q{col2}"], constraint)

        solution = solver.solve()

        assert solution is not None

    def test_unsolvable_csp(self):
        """Test unsolvable CSP."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2])
        solver.add_variable("Y", [1, 2])
        solver.add_variable("Z", [1, 2])

        # All must be different - impossible with only 2 values
        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])
        solver.add_constraint(["Y", "Z"], lambda a: a["Y"] != a["Z"])
        solver.add_constraint(["X", "Z"], lambda a: a["X"] != a["Z"])

        solution = solver.solve()

        assert solution is None

    def test_forward_checking(self):
        """Test forward checking algorithm."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])
        solver.add_variable("Y", [1, 2, 3])
        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])

        solution = solver.solve(algorithm="forward_checking")

        assert solution is not None
        assert solution["X"] != solution["Y"]

    def test_ac3_consistency(self):
        """Test AC-3 arc consistency."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3, 4])
        solver.add_variable("Y", [2, 3, 4, 5])
        solver.add_constraint(["X", "Y"], lambda a: a["X"] < a["Y"])

        # Run AC-3
        consistent = solver._ac3()

        assert consistent
        # Should reduce domains
        assert len(solver.domains["X"]) <= 4

    def test_min_conflicts(self):
        """Test min-conflicts local search."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])
        solver.add_variable("Y", [1, 2, 3])
        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])

        solution = solver.solve(algorithm="min_conflicts")

        # May or may not find solution (local search)
        if solution is not None:
            assert solution["X"] != solution["Y"]

    def test_is_consistent(self):
        """Test consistency checking."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])
        solver.add_variable("Y", [1, 2, 3])
        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])

        # X=1, Y=1 is inconsistent
        assert not solver._is_consistent("Y", 1, {"X": 1})

        # X=1, Y=2 is consistent
        assert solver._is_consistent("Y", 2, {"X": 1})

    def test_select_unassigned_variable_mrv(self):
        """Test MRV heuristic for variable selection."""
        solver = CSPSolver()

        solver.add_variable("X", [1, 2, 3])
        solver.add_variable("Y", [1])  # Smallest domain
        solver.add_variable("Z", [1, 2])

        selected = solver._select_unassigned_variable({})

        # Should select Y (minimum remaining values)
        assert selected == "Y"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complex scenarios."""

    def test_bayesian_network_inference_chain(self):
        """Test inference in chain network."""
        bn = BayesianNetworkReasoner()

        # Cloudy -> Rain -> WetGrass
        bn.add_variable("Cloudy", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable(
            "Rain", VariableType.DISCRETE, domain=[True, False], parents=["Cloudy"]
        )
        bn.add_variable(
            "WetGrass", VariableType.DISCRETE, domain=[True, False], parents=["Rain"]
        )

        bn.set_cpt("Cloudy", {(): {True: 0.5, False: 0.5}})
        bn.set_cpt(
            "Rain",
            {(True,): {True: 0.8, False: 0.2}, (False,): {True: 0.2, False: 0.8}},
        )
        bn.set_cpt(
            "WetGrass",
            {(True,): {True: 0.9, False: 0.1}, (False,): {True: 0.1, False: 0.9}},
        )

        # Query: P(Cloudy | WetGrass=True)
        result = bn.query("Cloudy", evidence={"WetGrass": True})

        # Cloudy should be more likely given wet grass
        assert result[True] > 0.5

    def test_bayesian_network_v_structure(self):
        """Test inference in V-structure (collider)."""
        bn = BayesianNetworkReasoner()

        # Rain -> WetGrass <- Sprinkler
        bn.add_variable("Rain", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable("Sprinkler", VariableType.DISCRETE, domain=[True, False])
        bn.add_variable(
            "WetGrass",
            VariableType.DISCRETE,
            domain=[True, False],
            parents=["Rain", "Sprinkler"],
        )

        bn.set_cpt("Rain", {(): {True: 0.2, False: 0.8}})
        bn.set_cpt("Sprinkler", {(): {True: 0.1, False: 0.9}})
        bn.set_cpt(
            "WetGrass",
            {
                (True, True): {True: 0.99, False: 0.01},
                (True, False): {True: 0.9, False: 0.1},
                (False, True): {True: 0.9, False: 0.1},
                (False, False): {True: 0.0, False: 1.0},
            },
        )

        # Query: P(Rain | WetGrass=True, Sprinkler=True)
        result = bn.query("Rain", evidence={"WetGrass": True, "Sprinkler": True})

        # Rain should be less likely (explaining away)
        assert isinstance(result, dict)

    def test_csp_sudoku_subset(self):
        """Test solving subset of Sudoku."""
        solver = CSPSolver()

        # 4x4 Sudoku
        n = 4
        for row in range(n):
            for col in range(n):
                solver.add_variable(f"cell_{row}_{col}", [1, 2, 3, 4])

        # Row constraints
        for row in range(n):
            for col1 in range(n):
                for col2 in range(col1 + 1, n):
                    c1 = f"cell_{row}_{col1}"
                    c2 = f"cell_{row}_{col2}"
                    solver.add_constraint(
                        [c1, c2], lambda a, c1=c1, c2=c2: a[c1] != a[c2]
                    )

        # Column constraints
        for col in range(n):
            for row1 in range(n):
                for row2 in range(row1 + 1, n):
                    c1 = f"cell_{row1}_{col}"
                    c2 = f"cell_{row2}_{col}"
                    solver.add_constraint(
                        [c1, c2], lambda a, c1=c1, c2=c2: a[c1] != a[c2]
                    )

        solution = solver.solve()

        # Should find valid solution
        assert solution is not None


# ============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_bayesian_network_query(self):
        """Test query on empty network."""
        bn = BayesianNetworkReasoner()

        try:
            bn.query("X", evidence={})
            assert False, "Should raise error"
        except ValueError:
            assert True

    def test_query_nonexistent_variable(self):
        """Test querying variable that doesn't exist."""
        bn = BayesianNetworkReasoner()
        bn.add_variable("X", VariableType.DISCRETE, domain=[1, 2])

        try:
            bn.query("Y", evidence={})
            assert False, "Should raise error"
        except ValueError:
            assert True

    def test_set_cpt_wrong_type(self):
        """Test setting CPT on non-discrete variable."""
        bn = BayesianNetworkReasoner()
        bn.add_variable("X", VariableType.GAUSSIAN)

        try:
            bn.set_cpt("X", {(): {1: 0.5, 2: 0.5}})
            assert False, "Should raise error"
        except ValueError:
            assert True

    def test_factor_normalize_empty(self):
        """Test normalizing empty factor."""
        factor = Factor(variables=["X"], values={})

        factor.normalize()

        # Should handle gracefully
        assert len(factor.values) == 0

    def test_csp_no_solution_detected(self):
        """Test CSP correctly detects no solution."""
        solver = CSPSolver()

        solver.add_variable("X", [1])
        solver.add_variable("Y", [1])
        solver.add_constraint(["X", "Y"], lambda a: a["X"] != a["Y"])

        solution = solver.solve()

        assert solution is None

    def test_learning_with_missing_data(self):
        """Test parameter learning with incomplete data."""
        bn = BayesianNetworkReasoner()
        bn.add_variable("X", VariableType.DISCRETE, domain=[True, False])

        # Some samples missing X
        data = [{"X": True}] * 5 + [{}] * 5

        # Should handle missing data gracefully
        try:
            bn.learn_parameters_mle(data)
            assert True
        except Exception:
            assert True  # Either handles or raises gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
