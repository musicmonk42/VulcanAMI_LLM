"""
Advanced reasoning systems.

Implements sophisticated reasoning methods beyond classical logic:
- Fuzzy logic with membership functions and inference
- Temporal reasoning with Allen's interval algebra
- Meta-level reasoning for strategy selection
- Proof learning with pattern extraction

These systems handle:
- Uncertain and imprecise information (fuzzy)
- Time-based reasoning (temporal)
- Self-reflective reasoning (meta)
- Knowledge extraction from proofs (learning)
"""

import heapq
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .core import (Clause, Constant, Function, ProofNode, Term, Variable)

logger = logging.getLogger(__name__)


# ============================================================================
# FUZZY LOGIC REASONER - FIXED VERSION
# ============================================================================


@dataclass
class FuzzySetMetadata:
    """
    Metadata for fuzzy sets to track properties.

    Attributes:
        name: Fuzzy set name
        set_type: Type of membership function (triangular, trapezoidal, gaussian, custom)
        support_range: (min, max) where membership > 0
        core_range: (min, max) where membership = 1 (None if no core)
        peak_value: Point(s) with maximum membership
        parameters: Dict of parameters defining the set
    """

    name: str
    set_type: str
    support_range: Tuple[float, float]
    core_range: Optional[Tuple[float, float]] = None
    peak_value: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class FuzzyLogicReasoner:
    """
    COMPLETE FIXED IMPLEMENTATION: Fuzzy logic inference engine.

    Implements:
    - Fuzzy sets with membership functions (triangular, trapezoidal, Gaussian)
    - T-norms for AND operations (min, product, Łukasiewicz)
    - S-norms for OR operations (max, probabilistic sum, Łukasiewicz)
    - Fuzzy rule evaluation (Mamdani and Sugeno style)
    - Defuzzification methods (centroid, maximum)
    - FIXED: Dynamic universe of discourse detection
    - FIXED: Fuzzy set metadata tracking

    Fuzzy logic extends classical logic to handle imprecise/vague information:
    - Classical: "Temperature is hot" (true/false)
    - Fuzzy: "Temperature is hot" (degree: 0.7)

    Features:
    - Multiple membership function types
    - Flexible T-norm/S-norm selection
    - Mamdani inference (linguistic output)
    - Sugeno inference (functional output)
    - Multiple defuzzification strategies
    - Dynamic universe of discourse
    - Metadata tracking for all fuzzy sets

    Example:
        >>> fuzzy = FuzzyLogicReasoner()
        >>> # Define fuzzy sets for temperature
        >>> fuzzy.add_triangular_set('cold', 0, 0, 20)
        >>> fuzzy.add_triangular_set('warm', 15, 25, 35)
        >>> fuzzy.add_triangular_set('hot', 30, 40, 40)
        >>> # Add fuzzy rule
        >>> fuzzy.add_rule(
        ...     antecedent={'temp': 'hot'},
        ...     consequent={'fan_speed': 'high'},
        ...     weight=1.0
        ... )
        >>> # Perform inference
        >>> outputs = fuzzy.infer({'temp': 35})
    """

    def __init__(self):
        """Initialize fuzzy logic reasoner."""
        self.fuzzy_sets = {}
        self.fuzzy_sets_metadata = {}  # FIXED: Track metadata
        self.fuzzy_rules = []
        self.variables = {}

    def add_fuzzy_set(
        self,
        name: str,
        membership_func: Callable[[float], float],
        support_range: Tuple[float, float] = None,
        core_range: Tuple[float, float] = None,
    ):
        """
        Add fuzzy set with custom membership function.

        FIXED: Now tracks metadata for the fuzzy set.

        Args:
            name: Fuzzy set name
            membership_func: Function mapping value to membership degree [0, 1]
            support_range: Optional (min, max) where membership > 0
            core_range: Optional (min, max) where membership = 1

        Example:
            >>> def custom_membership(x):
            ...     return max(0, min(1, x / 100))
            >>> fuzzy.add_fuzzy_set('custom', custom_membership,
            ...                     support_range=(0, 100))
        """
        self.fuzzy_sets[name] = membership_func

        # FIXED: Store metadata
        if support_range is None:
            # Try to auto-detect by sampling
            support_range = self._estimate_support_range(membership_func)

        metadata = FuzzySetMetadata(
            name=name,
            set_type="custom",
            support_range=support_range,
            core_range=core_range,
            parameters={"custom": True},
        )
        self.fuzzy_sets_metadata[name] = metadata

    def _estimate_support_range(
        self,
        membership_func: Callable[[float], float],
        sample_range: Tuple[float, float] = (-100, 100),
        num_samples: int = 1000,
        threshold: float = 0.01,
    ) -> Tuple[float, float]:
        """
        Estimate support range by sampling membership function.

        Args:
            membership_func: Membership function to sample
            sample_range: Range to sample over
            num_samples: Number of sample points
            threshold: Minimum membership to consider as support

        Returns:
            Estimated (min, max) support range
        """
        try:
            import numpy as np

            x_values = np.linspace(sample_range[0], sample_range[1], num_samples)
            memberships = [membership_func(x) for x in x_values]

            # Find points where membership > threshold
            support_indices = [i for i, m in enumerate(memberships) if m > threshold]

            if support_indices:
                min_idx = min(support_indices)
                max_idx = max(support_indices)
                return (float(x_values[min_idx]), float(x_values[max_idx]))
            else:
                return sample_range
        except ImportError:
            # Fallback without numpy
            x_values = [
                sample_range[0] + i * (sample_range[1] - sample_range[0]) / num_samples
                for i in range(num_samples)
            ]
            memberships = [membership_func(x) for x in x_values]

            support_indices = [i for i, m in enumerate(memberships) if m > threshold]

            if support_indices:
                min_idx = min(support_indices)
                max_idx = max(support_indices)
                return (x_values[min_idx], x_values[max_idx])
            else:
                return sample_range

    def add_triangular_set(self, name: str, a: float, b: float, c: float):
        """
        Add triangular membership function.

        FIXED: Now tracks metadata including support and core ranges.

        Triangular function:
        - 0 for x <= a or x >= c
        - Linear increase from a to b
        - Linear decrease from b to c
        - Peak at b

        Args:
            name: Fuzzy set name
            a: Left base point
            b: Peak point
            c: Right base point

        Example:
            >>> # "Medium" temperature: 15-25-35
            >>> fuzzy.add_triangular_set('medium', 15, 25, 35)
        """

        def triangular(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            else:  # b < x < c
                return (c - x) / (c - b)

        self.fuzzy_sets[name] = triangular

        # FIXED: Store comprehensive metadata
        metadata = FuzzySetMetadata(
            name=name,
            set_type="triangular",
            support_range=(a, c),
            core_range=(b, b),  # Single point core
            peak_value=b,
            parameters={"a": a, "b": b, "c": c},
        )
        self.fuzzy_sets_metadata[name] = metadata

    def add_trapezoidal_set(self, name: str, a: float, b: float, c: float, d: float):
        """
        Add trapezoidal membership function.

        FIXED: Now tracks metadata including support and core ranges.

        Trapezoidal function:
        - 0 for x <= a or x >= d
        - Linear increase from a to b
        - 1.0 from b to c (plateau)
        - Linear decrease from c to d

        Args:
            name: Fuzzy set name
            a: Left base point
            b: Left plateau point
            c: Right plateau point
            d: Right base point

        Example:
            >>> # "Hot" temperature: 25-30-35-40
            >>> fuzzy.add_trapezoidal_set('hot', 25, 30, 35, 40)
        """

        def trapezoidal(x: float) -> float:
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x <= c:
                return 1.0
            else:  # c < x < d
                return (d - x) / (d - c)

        self.fuzzy_sets[name] = trapezoidal

        # FIXED: Store comprehensive metadata
        metadata = FuzzySetMetadata(
            name=name,
            set_type="trapezoidal",
            support_range=(a, d),
            core_range=(b, c),  # Plateau is the core
            peak_value=(b + c) / 2,  # Center of plateau
            parameters={"a": a, "b": b, "c": c, "d": d},
        )
        self.fuzzy_sets_metadata[name] = metadata

    def add_gaussian_set(self, name: str, mean: float, std: float):
        """
        Add Gaussian membership function.

        FIXED: Now tracks metadata including support range.

        Gaussian function: exp(-((x - mean)² / (2 * std²)))
        - Smooth, bell-shaped curve
        - Peak at mean
        - Width controlled by std

        Args:
            name: Fuzzy set name
            mean: Center of Gaussian
            std: Standard deviation (width)

        Example:
            >>> # "Normal" temperature centered at 25°C
            >>> fuzzy.add_gaussian_set('normal', mean=25, std=5)
        """
        import math

        def gaussian(x: float) -> float:
            return math.exp(-((x - mean) ** 2) / (2 * std**2))

        self.fuzzy_sets[name] = gaussian

        # FIXED: Store comprehensive metadata
        # Gaussian support is technically infinite, but use 3*std as practical range
        support_range = (mean - 3 * std, mean + 3 * std)

        metadata = FuzzySetMetadata(
            name=name,
            set_type="gaussian",
            support_range=support_range,
            core_range=(mean, mean),  # Peak at mean
            peak_value=mean,
            parameters={"mean": mean, "std": std},
        )
        self.fuzzy_sets_metadata[name] = metadata

    def add_rule(
        self,
        antecedent: Dict[str, str],
        consequent: Dict[str, str],
        weight: float = 1.0,
    ):
        """
        Add fuzzy rule.

        Rule format: IF antecedent THEN consequent
        Example: IF temperature IS hot AND humidity IS high THEN fan_speed IS high

        Args:
            antecedent: Dict mapping variables to fuzzy set names (IF part)
            consequent: Dict mapping variables to fuzzy set names (THEN part)
            weight: Rule weight/confidence [0, 1]

        Example:
            >>> fuzzy.add_rule(
            ...     antecedent={'temp': 'hot', 'humidity': 'high'},
            ...     consequent={'fan_speed': 'high'},
            ...     weight=0.9
            ... )
        """
        self.fuzzy_rules.append(
            {"antecedent": antecedent, "consequent": consequent, "weight": weight}
        )

    def evaluate_membership(self, fuzzy_set_name: str, value: float) -> float:
        """
        Evaluate membership degree of value in fuzzy set.

        Args:
            fuzzy_set_name: Name of fuzzy set
            value: Input value

        Returns:
            Membership degree [0, 1]

        Example:
            >>> fuzzy.evaluate_membership('hot', 35)
            0.75
        """
        if fuzzy_set_name not in self.fuzzy_sets:
            return 0.0

        return self.fuzzy_sets[fuzzy_set_name](value)

    def t_norm_min(self, a: float, b: float) -> float:
        """
        Minimum T-norm (Zadeh AND).

        Standard fuzzy AND operation.

        Args:
            a: First value [0, 1]
            b: Second value [0, 1]

        Returns:
            min(a, b)
        """
        return min(a, b)

    def t_norm_product(self, a: float, b: float) -> float:
        """
        Product T-norm (probabilistic AND).

        Args:
            a: First value [0, 1]
            b: Second value [0, 1]

        Returns:
            a * b
        """
        return a * b

    def t_norm_lukasiewicz(self, a: float, b: float) -> float:
        """
        Łukasiewicz T-norm (bounded difference).

        Args:
            a: First value [0, 1]
            b: Second value [0, 1]

        Returns:
            max(0, a + b - 1)
        """
        return max(0, a + b - 1)

    def s_norm_max(self, a: float, b: float) -> float:
        """
        Maximum S-norm (Zadeh OR).

        Standard fuzzy OR operation.

        Args:
            a: First value [0, 1]
            b: Second value [0, 1]

        Returns:
            max(a, b)
        """
        return max(a, b)

    def s_norm_probabilistic(self, a: float, b: float) -> float:
        """
        Probabilistic sum S-norm.

        Args:
            a: First value [0, 1]
            b: Second value [0, 1]

        Returns:
            a + b - a * b
        """
        return a + b - a * b

    def s_norm_lukasiewicz(self, a: float, b: float) -> float:
        """
        Łukasiewicz S-norm (bounded sum).

        Args:
            a: First value [0, 1]
            b: Second value [0, 1]

        Returns:
            min(1, a + b)
        """
        return min(1, a + b)

    def infer(
        self,
        inputs: Dict[str, float],
        t_norm: str = "min",
        s_norm: str = "max",
        defuzz_method: str = "centroid",
    ) -> Dict[str, float]:
        """
        Perform fuzzy inference.

        Inference process:
        1. Fuzzification: Convert crisp inputs to fuzzy values
        2. Rule evaluation: Apply fuzzy rules
        3. Aggregation: Combine rule outputs
        4. Defuzzification: Convert to crisp outputs

        Args:
            inputs: Dict mapping input variables to crisp values
            t_norm: T-norm to use ('min', 'product', 'lukasiewicz')
            s_norm: S-norm to use ('max', 'probabilistic', 'lukasiewicz')
            defuzz_method: Defuzzification method ('centroid', 'maximum')

        Returns:
            Dict mapping output variables to crisp values

        Example:
            >>> outputs = fuzzy.infer(
            ...     inputs={'temperature': 35, 'humidity': 70},
            ...     t_norm='min',
            ...     defuzz_method='centroid'
            ... )
            >>> outputs['fan_speed']
            85.5
        """
        # Select T-norm and S-norm
        if t_norm == "min":
            t_norm_func = self.t_norm_min
        elif t_norm == "product":
            t_norm_func = self.t_norm_product
        else:
            t_norm_func = self.t_norm_lukasiewicz

        if s_norm == "max":
            s_norm_func = self.s_norm_max
        elif s_norm == "probabilistic":
            s_norm_func = self.s_norm_probabilistic
        else:
            s_norm_func = self.s_norm_lukasiewicz

        # Fuzzification: compute membership degrees for inputs
        fuzzified_inputs = {}
        for var_name, value in inputs.items():
            fuzzified_inputs[var_name] = {}
            for set_name, membership_func in self.fuzzy_sets.items():
                if var_name in set_name or set_name.startswith(var_name + "_"):
                    fuzzified_inputs[var_name][set_name] = membership_func(value)

        # Rule evaluation
        activated_outputs = defaultdict(list)

        for rule in self.fuzzy_rules:
            # Evaluate antecedent
            antecedent_degrees = []

            for var_name, fuzzy_set_name in rule["antecedent"].items():
                if var_name in fuzzified_inputs:
                    degree = fuzzified_inputs[var_name].get(fuzzy_set_name, 0.0)
                    antecedent_degrees.append(degree)

            # Aggregate antecedent (using T-norm)
            if antecedent_degrees:
                antecedent_strength = antecedent_degrees[0]
                for degree in antecedent_degrees[1:]:
                    antecedent_strength = t_norm_func(antecedent_strength, degree)

                # Apply rule weight
                antecedent_strength *= rule["weight"]

                # Apply to consequent
                for var_name, fuzzy_set_name in rule["consequent"].items():
                    activated_outputs[var_name].append(
                        {"fuzzy_set": fuzzy_set_name, "degree": antecedent_strength}
                    )

        # Aggregation and defuzzification
        outputs = {}

        for var_name, activations in activated_outputs.items():
            if not activations:
                outputs[var_name] = 0.0
                continue

            # Aggregate using S-norm
            aggregated_membership = {}

            for activation in activations:
                fuzzy_set_name = activation["fuzzy_set"]
                degree = activation["degree"]

                if fuzzy_set_name in aggregated_membership:
                    aggregated_membership[fuzzy_set_name] = s_norm_func(
                        aggregated_membership[fuzzy_set_name], degree
                    )
                else:
                    aggregated_membership[fuzzy_set_name] = degree

            # Defuzzification
            if defuzz_method == "centroid":
                outputs[var_name] = self._defuzzify_centroid(aggregated_membership)
            else:  # maximum
                outputs[var_name] = self._defuzzify_maximum(aggregated_membership)

        return outputs

    def _detect_universe_range(
        self, aggregated_membership: Dict[str, float], padding_factor: float = 0.1
    ) -> Tuple[float, float]:
        """
        FIXED: Automatically detect universe of discourse from fuzzy sets.

        Determines the appropriate range for defuzzification by examining
        the support ranges of all involved fuzzy sets.

        Args:
            aggregated_membership: Dict mapping fuzzy sets to activation degrees
            padding_factor: Add padding as fraction of total range (default 10%)

        Returns:
            (min, max) tuple for universe of discourse

        Example:
            >>> # If fuzzy sets have ranges [0, 50] and [40, 100]
            >>> range = fuzzy._detect_universe_range({'set1': 0.7, 'set2': 0.3})
            >>> range
            (-5.0, 105.0)  # With 10% padding
        """
        min_val = float("inf")
        max_val = float("-inf")

        # Find the overall range from metadata
        for fuzzy_set_name in aggregated_membership.keys():
            if fuzzy_set_name in self.fuzzy_sets_metadata:
                metadata = self.fuzzy_sets_metadata[fuzzy_set_name]
                support_range = metadata.support_range

                min_val = min(min_val, support_range[0])
                max_val = max(max_val, support_range[1])

        # If no metadata found, fall back to sampling
        if min_val == float("inf") or max_val == float("-inf"):
            min_val = -10.0
            max_val = 10.0

            # Sample each membership function to estimate range
            for fuzzy_set_name in aggregated_membership.keys():
                if fuzzy_set_name in self.fuzzy_sets:
                    estimated_range = self._estimate_support_range(
                        self.fuzzy_sets[fuzzy_set_name]
                    )
                    min_val = min(min_val, estimated_range[0])
                    max_val = max(max_val, estimated_range[1])

        # Add padding
        range_span = max_val - min_val
        padding = range_span * padding_factor

        return (min_val - padding, max_val + padding)

    def _defuzzify_centroid(
        self,
        aggregated_membership: Dict[str, float],
        universe_range: Tuple[float, float] = None,
        num_samples: int = 200,
    ) -> float:
        """
        FIXED: Centroid defuzzification (center of gravity) with dynamic universe.

        Computes weighted average of fuzzy set centroids using the area
        under the aggregated membership function.

        Args:
            aggregated_membership: Dict mapping fuzzy sets to activation degrees
            universe_range: (min, max) range for universe, auto-detected if None
            num_samples: Number of sample points for integration

        Returns:
            Crisp output value

        Example:
            >>> outputs = fuzzy._defuzzify_centroid(
            ...     {'hot': 0.7, 'medium': 0.3},
            ...     universe_range=(0, 100)
            ... )
            >>> outputs
            72.5
        """
        if not aggregated_membership:
            return 0.0

        # FIXED: Auto-detect universe range if not provided
        if universe_range is None:
            universe_range = self._detect_universe_range(aggregated_membership)

        numerator = 0.0
        denominator = 0.0

        # Sample points for each fuzzy set
        try:
            import numpy as np

            x_values = np.linspace(universe_range[0], universe_range[1], num_samples)

            # Compute aggregated membership at each point
            for x in x_values:
                # Take maximum of all activated fuzzy sets at this point
                max_membership = 0.0

                for fuzzy_set_name, degree in aggregated_membership.items():
                    if fuzzy_set_name not in self.fuzzy_sets:
                        continue

                    membership_func = self.fuzzy_sets[fuzzy_set_name]
                    membership = min(membership_func(x), degree)
                    max_membership = max(max_membership, membership)

                # Accumulate for centroid calculation
                numerator += x * max_membership
                denominator += max_membership

        except ImportError:
            # Fallback without numpy
            step = (universe_range[1] - universe_range[0]) / num_samples
            x_values = [universe_range[0] + i * step for i in range(num_samples)]

            for x in x_values:
                max_membership = 0.0

                for fuzzy_set_name, degree in aggregated_membership.items():
                    if fuzzy_set_name not in self.fuzzy_sets:
                        continue

                    membership_func = self.fuzzy_sets[fuzzy_set_name]
                    membership = min(membership_func(x), degree)
                    max_membership = max(max_membership, membership)

                numerator += x * max_membership
                denominator += max_membership

        if denominator < 1e-10:
            # No significant membership, return center of universe
            return (universe_range[0] + universe_range[1]) / 2

        return numerator / denominator

    def _defuzzify_maximum(
        self,
        aggregated_membership: Dict[str, float],
        universe_range: Tuple[float, float] = None,
        num_samples: int = 200,
    ) -> float:
        """
        FIXED: Maximum defuzzification with dynamic universe.

        Returns the point with maximum membership degree in the aggregated
        membership function.

        Args:
            aggregated_membership: Dict mapping fuzzy sets to activation degrees
            universe_range: (min, max) range for universe, auto-detected if None
            num_samples: Number of sample points

        Returns:
            Crisp output value at maximum

        Example:
            >>> output = fuzzy._defuzzify_maximum({'hot': 0.8})
            >>> output
            40.0  # Peak of 'hot' fuzzy set
        """
        if not aggregated_membership:
            return 0.0

        # FIXED: Auto-detect universe range if not provided
        if universe_range is None:
            universe_range = self._detect_universe_range(aggregated_membership)

        max_membership_value = 0.0
        max_x = universe_range[0]

        try:
            import numpy as np

            x_values = np.linspace(universe_range[0], universe_range[1], num_samples)

            for x in x_values:
                # Compute maximum membership at this point
                current_membership = 0.0

                for fuzzy_set_name, degree in aggregated_membership.items():
                    if fuzzy_set_name not in self.fuzzy_sets:
                        continue

                    membership_func = self.fuzzy_sets[fuzzy_set_name]
                    membership = min(membership_func(x), degree)
                    current_membership = max(current_membership, membership)

                # Track maximum
                if current_membership > max_membership_value:
                    max_membership_value = current_membership
                    max_x = x

        except ImportError:
            # Fallback without numpy
            step = (universe_range[1] - universe_range[0]) / num_samples
            x_values = [universe_range[0] + i * step for i in range(num_samples)]

            for x in x_values:
                current_membership = 0.0

                for fuzzy_set_name, degree in aggregated_membership.items():
                    if fuzzy_set_name not in self.fuzzy_sets:
                        continue

                    membership_func = self.fuzzy_sets[fuzzy_set_name]
                    membership = min(membership_func(x), degree)
                    current_membership = max(current_membership, membership)

                if current_membership > max_membership_value:
                    max_membership_value = current_membership
                    max_x = x

        return max_x

    def add_mamdani_rule(
        self,
        antecedent_conditions: List[Tuple[str, str]],
        consequent_actions: List[Tuple[str, str]],
        weight: float = 1.0,
    ):
        """
        Add Mamdani-style fuzzy rule.

        Mamdani rules have linguistic consequents:
        IF temp IS hot THEN fan IS fast

        Args:
            antecedent_conditions: List of (variable, fuzzy_set) tuples for IF part
            consequent_actions: List of (variable, fuzzy_set) tuples for THEN part
            weight: Rule weight/confidence

        Example:
            >>> fuzzy.add_mamdani_rule(
            ...     antecedent_conditions=[('temp', 'hot'), ('humidity', 'high')],
            ...     consequent_actions=[('fan', 'fast')],
            ...     weight=0.9
            ... )
        """
        antecedent = {var: fset for var, fset in antecedent_conditions}
        consequent = {var: fset for var, fset in consequent_actions}

        self.add_rule(antecedent, consequent, weight)

    def add_sugeno_rule(
        self,
        antecedent_conditions: List[Tuple[str, str]],
        output_function: Callable[[Dict[str, float]], float],
        output_variable: str,
        weight: float = 1.0,
    ):
        """
        Add Sugeno-style fuzzy rule.

        Sugeno rules have functional consequents:
        IF temp IS hot THEN output = 2*temp + humidity

        Args:
            antecedent_conditions: List of (variable, fuzzy_set) tuples for IF part
            output_function: Function that computes crisp output from inputs
            output_variable: Name of output variable
            weight: Rule weight

        Example:
            >>> fuzzy.add_sugeno_rule(
            ...     antecedent_conditions=[('temp', 'hot')],
            ...     output_function=lambda inputs: 2 * inputs['temp'],
            ...     output_variable='fan_speed',
            ...     weight=1.0
            ... )
        """
        # For Sugeno, consequent is a function
        antecedent = {var: fset for var, fset in antecedent_conditions}
        consequent = {output_variable: output_function}

        self.add_rule(antecedent, consequent, weight)


# ============================================================================
# TEMPORAL REASONER - FIXED VERSION
# ============================================================================


@dataclass
class TimeInterval:
    """
    Represents a time interval with support for uncertainty.

    Attributes:
        start: Start time (point or range)
        end: End time (point or range)
        start_uncertain: Whether start time is uncertain
        end_uncertain: Whether end time is uncertain
        granularity: Time granularity (seconds, minutes, hours, days, etc.)
    """

    start: Union[float, Tuple[float, float]]
    end: Union[float, Tuple[float, float]]
    start_uncertain: bool = False
    end_uncertain: bool = False
    granularity: str = "seconds"

    def get_duration(self) -> Union[float, Tuple[float, float]]:
        """
        Calculate duration with uncertainty propagation.

        Returns:
            Duration as point or range
        """
        if self.start_uncertain or self.end_uncertain:
            # Handle uncertain intervals
            start_min = self.start[0] if self.start_uncertain else self.start
            start_max = self.start[1] if self.start_uncertain else self.start
            end_min = self.end[0] if self.end_uncertain else self.end
            end_max = self.end[1] if self.end_uncertain else self.end

            # Duration range: [min, max]
            min_duration = end_min - start_max
            max_duration = end_max - start_min
            return (max(0, min_duration), max(0, max_duration))
        else:
            # Certain duration
            return self.end - self.start

    def overlaps_with(self, other: "TimeInterval") -> bool:
        """
        Check if this interval overlaps with another.

        Args:
            other: Another time interval

        Returns:
            True if intervals overlap
        """
        # Extract effective ranges
        self_start = self.start[0] if self.start_uncertain else self.start
        self_end = self.end[1] if self.end_uncertain else self.end
        other_start = other.start[0] if other.start_uncertain else other.start
        other_end = other.end[1] if other.end_uncertain else other.end

        return not (self_end <= other_start or other_end <= self_start)


@dataclass
class RecurringEvent:
    """
    Represents a recurring temporal event.

    Attributes:
        pattern: Recurrence pattern ('daily', 'weekly', 'monthly', 'yearly', 'custom')
        interval: Base time interval for single occurrence
        start_date: When recurrence starts
        end_date: When recurrence ends (None for infinite)
        exceptions: Dates to skip
        custom_rule: Custom recurrence rule function
    """

    pattern: str
    interval: TimeInterval
    start_date: float
    end_date: Optional[float] = None
    exceptions: Set[float] = field(default_factory=set)
    custom_rule: Optional[Callable[[float], bool]] = None

    def occurs_on(self, date: float) -> bool:
        """
        Check if event occurs on given date.

        Args:
            date: Date to check

        Returns:
            True if event occurs on this date
        """
        if date < self.start_date:
            return False

        if self.end_date is not None and date > self.end_date:
            return False

        if date in self.exceptions:
            return False

        if self.custom_rule is not None:
            return self.custom_rule(date)

        # Built-in pattern matching
        if self.pattern == "daily":
            return True
        elif self.pattern == "weekly":
            return int((date - self.start_date) % 7) == 0
        elif self.pattern == "monthly":
            return int((date - self.start_date) % 30) == 0
        elif self.pattern == "yearly":
            return int((date - self.start_date) % 365) == 0

        return False


@dataclass
class EventHierarchy:
    """
    Represents hierarchical event structure.

    Attributes:
        event_id: Event identifier
        parent_id: Parent event ID (None for root)
        children: List of child event IDs
        level: Hierarchy level (0 = root)
    """

    event_id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    level: int = 0


class TemporalReasoner:
    """
    COMPLETE FIXED IMPLEMENTATION: Temporal reasoning system.

    Handles temporal logic, event sequencing, and timeline management using
    Allen's interval algebra for representing temporal relations.

    Allen's 13 temporal relations:
    - before, after: X entirely before/after Y
    - meets, met-by: X ends exactly when Y starts
    - overlaps, overlapped-by: X and Y overlap
    - starts, started-by: X starts when Y starts
    - finishes, finished-by: X finishes when Y finishes
    - during, contains: X entirely during Y
    - equals: X and Y are identical

    FIXED Features:
    - Event representation with time intervals
    - Allen's interval algebra
    - Complete path consistency with bidirectional propagation
    - Cycle detection in temporal graphs
    - Priority queue optimization for constraint propagation
    - Support for uncertain time intervals (ranges)
    - Recurring events
    - Event hierarchies
    - Interval arithmetic
    - Temporal query answering
    - Event sequencing

    Example:
        >>> temporal = TemporalReasoner()
        >>> temporal.add_event('breakfast', start_time=7.0, end_time=8.0)
        >>> temporal.add_event('work', start_time=9.0, end_time=17.0)
        >>> temporal.add_temporal_relation('breakfast', 'work', 'before')
        >>> temporal.check_consistency()
        True
    """

    def __init__(self):
        """Initialize temporal reasoner."""
        self.events = {}  # event_id -> event dict
        self.temporal_relations = []
        self.constraints = []
        self.recurring_events = {}  # event_id -> RecurringEvent
        self.event_hierarchy = {}  # event_id -> EventHierarchy
        self._init_allen_composition_table()

    def _init_allen_composition_table(self):
        """
        Initialize complete Allen's interval algebra composition table.

        This is the full 13x13 composition table for all Allen relations.
        For any two relations R1(X,Y) and R2(Y,Z), gives possible relations R3(X,Z).
        """
        # Complete Allen's composition table
        # Format: self.composition_table[rel1][rel2] = {set of possible composed relations}
        self.composition_table = {
            "before": {
                "before": {"before"},
                "after": {
                    "before",
                    "after",
                    "meets",
                    "met-by",
                    "overlaps",
                    "overlapped-by",
                    "starts",
                    "started-by",
                    "finishes",
                    "finished-by",
                    "during",
                    "contains",
                    "equals",
                },
                "meets": {"before"},
                "met-by": {"before", "meets", "overlaps", "starts", "during"},
                "overlaps": {"before"},
                "overlapped-by": {"before", "meets", "overlaps", "starts", "during"},
                "starts": {"before"},
                "started-by": {"before", "meets", "overlaps", "starts", "during"},
                "finishes": {"before"},
                "finished-by": {"before", "meets", "overlaps", "starts", "during"},
                "during": {"before"},
                "contains": {"before", "meets", "overlaps", "starts", "during"},
                "equals": {"before"},
            },
            "after": {
                "before": {
                    "before",
                    "after",
                    "meets",
                    "met-by",
                    "overlaps",
                    "overlapped-by",
                    "starts",
                    "started-by",
                    "finishes",
                    "finished-by",
                    "during",
                    "contains",
                    "equals",
                },
                "after": {"after"},
                "meets": {"after", "met-by", "overlapped-by", "finishes", "contains"},
                "met-by": {"after"},
                "overlaps": {
                    "after",
                    "met-by",
                    "overlapped-by",
                    "finishes",
                    "contains",
                },
                "overlapped-by": {"after"},
                "starts": {"after", "met-by", "overlapped-by", "finishes", "contains"},
                "started-by": {"after"},
                "finishes": {
                    "after",
                    "met-by",
                    "overlapped-by",
                    "finishes",
                    "contains",
                },
                "finished-by": {"after"},
                "during": {"after", "met-by", "overlapped-by", "finishes", "contains"},
                "contains": {"after"},
                "equals": {"after"},
            },
            "meets": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"before"},
                "met-by": {"equals"},
                "overlaps": {"before"},
                "overlapped-by": {"after"},
                "starts": {"meets"},
                "started-by": {"met-by"},
                "finishes": {"before"},
                "finished-by": {"after"},
                "during": {"meets"},
                "contains": {"met-by"},
                "equals": {"meets"},
            },
            "met-by": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"equals"},
                "met-by": {"after"},
                "overlaps": {"before"},
                "overlapped-by": {"after"},
                "starts": {"met-by"},
                "started-by": {"meets"},
                "finishes": {"met-by"},
                "finished-by": {"meets"},
                "during": {"met-by"},
                "contains": {"meets"},
                "equals": {"met-by"},
            },
            "overlaps": {
                "before": {"before"},
                "after": {"after", "met-by", "overlapped-by", "finishes", "contains"},
                "meets": {"before"},
                "met-by": {"overlaps", "starts", "during"},
                "overlaps": {"before"},
                "overlapped-by": {
                    "after",
                    "met-by",
                    "overlapped-by",
                    "finishes",
                    "contains",
                },
                "starts": {"overlaps"},
                "started-by": {"overlaps", "starts", "during"},
                "finishes": {"before"},
                "finished-by": {
                    "after",
                    "met-by",
                    "overlapped-by",
                    "finishes",
                    "contains",
                },
                "during": {"overlaps"},
                "contains": {"overlaps", "starts", "during"},
                "equals": {"overlaps"},
            },
            "overlapped-by": {
                "before": {"before", "meets", "overlaps", "starts", "during"},
                "after": {"after"},
                "meets": {"finishes", "finished-by", "contains"},
                "met-by": {"after"},
                "overlaps": {"before", "meets", "overlaps", "starts", "during"},
                "overlapped-by": {"after"},
                "starts": {"during", "finishes", "contains"},
                "started-by": {"overlapped-by"},
                "finishes": {"finishes", "finished-by", "contains"},
                "finished-by": {"overlapped-by"},
                "during": {"during", "finishes", "contains"},
                "contains": {"overlapped-by"},
                "equals": {"overlapped-by"},
            },
            "starts": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"before"},
                "met-by": {"met-by"},
                "overlaps": {"before"},
                "overlapped-by": {"after"},
                "starts": {"starts"},
                "started-by": {"equals"},
                "finishes": {"during"},
                "finished-by": {"contains"},
                "during": {"during"},
                "contains": {"contains"},
                "equals": {"starts"},
            },
            "started-by": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"meets"},
                "met-by": {"after"},
                "overlaps": {"overlaps"},
                "overlapped-by": {"after"},
                "starts": {"equals"},
                "started-by": {"started-by"},
                "finishes": {"started-by"},
                "finished-by": {"overlaps", "starts", "during"},
                "during": {"started-by"},
                "contains": {"overlaps", "starts", "during"},
                "equals": {"started-by"},
            },
            "finishes": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"before"},
                "met-by": {"meets"},
                "overlaps": {"before"},
                "overlapped-by": {"overlapped-by"},
                "starts": {"during"},
                "started-by": {"contains"},
                "finishes": {"finishes"},
                "finished-by": {"equals"},
                "during": {"during"},
                "contains": {"contains"},
                "equals": {"finishes"},
            },
            "finished-by": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"before", "meets", "overlaps", "starts", "during"},
                "met-by": {"met-by"},
                "overlaps": {"before", "meets", "overlaps", "starts", "during"},
                "overlapped-by": {"overlapped-by"},
                "starts": {"contains"},
                "started-by": {"overlapped-by", "finishes", "contains"},
                "finishes": {"equals"},
                "finished-by": {"finished-by"},
                "during": {"contains"},
                "contains": {"overlapped-by", "finishes", "contains"},
                "equals": {"finished-by"},
            },
            "during": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"before"},
                "met-by": {"after"},
                "overlaps": {"before"},
                "overlapped-by": {"after"},
                "starts": {"during"},
                "started-by": {"contains"},
                "finishes": {"during"},
                "finished-by": {"contains"},
                "during": {"during"},
                "contains": {"contains"},
                "equals": {"during"},
            },
            "contains": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"overlaps", "starts", "during"},
                "met-by": {"overlapped-by", "finishes", "contains"},
                "overlaps": {"overlaps", "starts", "during"},
                "overlapped-by": {"overlapped-by", "finishes", "contains"},
                "starts": {"contains"},
                "started-by": {"contains"},
                "finishes": {"contains"},
                "finished-by": {"contains"},
                "during": {"contains"},
                "contains": {"contains"},
                "equals": {"contains"},
            },
            "equals": {
                "before": {"before"},
                "after": {"after"},
                "meets": {"meets"},
                "met-by": {"met-by"},
                "overlaps": {"overlaps"},
                "overlapped-by": {"overlapped-by"},
                "starts": {"starts"},
                "started-by": {"started-by"},
                "finishes": {"finishes"},
                "finished-by": {"finished-by"},
                "during": {"during"},
                "contains": {"contains"},
                "equals": {"equals"},
            },
        }

    def add_event(
        self,
        event_id: str,
        start_time: Union[float, Tuple[float, float]] = None,
        end_time: Union[float, Tuple[float, float]] = None,
        interval: TimeInterval = None,
        properties: Dict = None,
    ):
        """
        FIXED: Add temporal event with support for intervals and uncertainty.

        Args:
            event_id: Unique event identifier
            start_time: Event start time (point or range for uncertainty)
            end_time: Event end time (point or range for uncertainty)
            interval: TimeInterval object (alternative to start_time/end_time)
            properties: Additional event properties

        Examples:
            >>> # Certain time
            >>> temporal.add_event('meeting', start_time=14.0, end_time=15.5)

            >>> # Uncertain time (ranges)
            >>> temporal.add_event('lunch', start_time=(12.0, 12.5), end_time=(13.0, 13.5))

            >>> # Using TimeInterval
            >>> interval = TimeInterval(start=9.0, end=17.0, granularity='hours')
            >>> temporal.add_event('workday', interval=interval)
        """
        if interval is None:
            # Create interval from start_time and end_time
            start_uncertain = isinstance(start_time, tuple)
            end_uncertain = isinstance(end_time, tuple)

            interval = TimeInterval(
                start=start_time,
                end=end_time,
                start_uncertain=start_uncertain,
                end_uncertain=end_uncertain,
            )

        event = {
            "id": event_id,
            "interval": interval,
            "start": interval.start,
            "end": interval.end,
            "duration": interval.get_duration(),
            "properties": properties or {},
            "uncertain": interval.start_uncertain or interval.end_uncertain,
        }

        self.events[event_id] = event

    def add_recurring_event(
        self,
        event_id: str,
        pattern: str,
        base_interval: TimeInterval,
        start_date: float,
        end_date: Optional[float] = None,
        exceptions: Set[float] = None,
    ):
        """
        FIXED: Add recurring temporal event.

        Args:
            event_id: Unique event identifier
            pattern: Recurrence pattern ('daily', 'weekly', 'monthly', 'yearly', 'custom')
            base_interval: Time interval for single occurrence
            start_date: When recurrence starts
            end_date: When recurrence ends (None for infinite)
            exceptions: Dates to skip

        Example:
            >>> # Daily standup meeting
            >>> base = TimeInterval(start=9.0, end=9.25)  # 9:00-9:15
            >>> temporal.add_recurring_event(
            ...     'standup',
            ...     pattern='daily',
            ...     base_interval=base,
            ...     start_date=0.0
            ... )
        """
        recurring = RecurringEvent(
            pattern=pattern,
            interval=base_interval,
            start_date=start_date,
            end_date=end_date,
            exceptions=exceptions or set(),
        )

        self.recurring_events[event_id] = recurring

        # Add base event to events dict
        self.add_event(event_id, interval=base_interval, properties={"recurring": True})

    def add_event_to_hierarchy(self, event_id: str, parent_id: Optional[str] = None):
        """
        FIXED: Add event to hierarchy.

        Args:
            event_id: Event to add to hierarchy
            parent_id: Parent event (None for root)

        Example:
            >>> temporal.add_event_to_hierarchy('project_phase1')
            >>> temporal.add_event_to_hierarchy('task1', parent_id='project_phase1')
            >>> temporal.add_event_to_hierarchy('subtask1', parent_id='task1')
        """
        level = 0
        if parent_id is not None:
            if parent_id in self.event_hierarchy:
                level = self.event_hierarchy[parent_id].level + 1
                self.event_hierarchy[parent_id].children.append(event_id)

        hierarchy = EventHierarchy(event_id=event_id, parent_id=parent_id, level=level)

        self.event_hierarchy[event_id] = hierarchy

    def get_event_descendants(self, event_id: str) -> List[str]:
        """
        Get all descendant events in hierarchy.

        Args:
            event_id: Root event

        Returns:
            List of all descendant event IDs
        """
        if event_id not in self.event_hierarchy:
            return []

        descendants = []
        queue = deque([event_id])

        while queue:
            current = queue.popleft()
            if current in self.event_hierarchy:
                children = self.event_hierarchy[current].children
                descendants.extend(children)
                queue.extend(children)

        return descendants

    def add_temporal_relation(self, event1_id: str, event2_id: str, relation_type: str):
        """
        Add Allen's interval algebra relation.

        Relation types:
        - 'before': event1 entirely before event2
        - 'after': event1 entirely after event2
        - 'meets': event1 ends exactly when event2 starts
        - 'met-by': event2 ends exactly when event1 starts
        - 'overlaps': event1 and event2 overlap
        - 'overlapped-by': event2 overlaps event1
        - 'starts': event1 starts when event2 starts
        - 'started-by': event2 starts when event1 starts
        - 'finishes': event1 finishes when event2 finishes
        - 'finished-by': event2 finishes when event1 finishes
        - 'during': event1 entirely during event2
        - 'contains': event2 entirely during event1
        - 'equals': event1 and event2 are identical

        Args:
            event1_id: First event ID
            event2_id: Second event ID
            relation_type: Allen relation type

        Example:
            >>> temporal.add_temporal_relation('breakfast', 'lunch', 'before')
            >>> temporal.add_temporal_relation('meeting1', 'meeting2', 'meets')
        """
        self.temporal_relations.append(
            {"event1": event1_id, "event2": event2_id, "type": relation_type}
        )

    def check_consistency(self) -> bool:
        """
        FIXED: Check temporal consistency using complete constraint propagation.

        Uses enhanced path consistency algorithm with:
        - Bidirectional constraint propagation
        - Early termination on empty constraints
        - Cycle detection
        - Priority queue for efficiency

        Returns:
            True if consistent, False if contradictory constraints exist

        Example:
            >>> temporal.add_event('A', 0, 5)
            >>> temporal.add_event('B', 10, 15)
            >>> temporal.add_event('C', 3, 8)
            >>> temporal.add_temporal_relation('A', 'B', 'before')
            >>> temporal.add_temporal_relation('B', 'C', 'before')
            >>> temporal.add_temporal_relation('C', 'A', 'before')  # Inconsistent!
            >>> temporal.check_consistency()
            False
        """
        # Build constraint network
        constraints = self._build_constraint_network()

        # FIXED: Check for cycles first
        if self._has_cycles(constraints):
            logger.warning("Temporal graph contains cycles")

        # FIXED: Complete path consistency with optimizations
        return self._path_consistency_optimized(constraints)

    def _has_cycles(self, network: Dict) -> bool:
        """
        FIXED: Detect cycles in temporal constraint graph.

        Uses depth-first search to detect cycles.

        Args:
            network: Constraint network

        Returns:
            True if cycles detected
        """
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            if node in network:
                for neighbor in network[node]:
                    if neighbor not in visited:
                        if dfs(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        for node in network:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def _build_constraint_network(self) -> Dict:
        """
        Build temporal constraint network.

        Returns:
            Network dict mapping event pairs to possible relations
        """
        network = defaultdict(lambda: defaultdict(set))

        # Add explicit relations
        for rel in self.temporal_relations:
            e1, e2 = rel["event1"], rel["event2"]
            rel_type = rel["type"]
            network[e1][e2].add(rel_type)

            # FIXED: Add inverse relation (bidirectional)
            inverse = self._inverse_relation(rel_type)
            if inverse:
                network[e2][e1].add(inverse)

        # Add transitive constraints
        for e1 in list(network.keys()):
            for e2 in [network[e1].keys()):
                for e3 in [network[e2].keys()):
                    if e3 != e1:
                        # Compose relations
                        composed = self._compose_relations(
                            network[e1][e2], network[e2][e3]
                        )
                        network[e1][e3].update(composed)

        return network

    def _inverse_relation(self, relation: str) -> Optional[str]:
        """
        Get inverse Allen relation.

        Args:
            relation: Allen relation

        Returns:
            Inverse relation
        """
        inverses = {
            "before": "after",
            "after": "before",
            "meets": "met-by",
            "met-by": "meets",
            "overlaps": "overlapped-by",
            "overlapped-by": "overlaps",
            "starts": "started-by",
            "started-by": "starts",
            "finishes": "finished-by",
            "finished-by": "finishes",
            "during": "contains",
            "contains": "during",
            "equals": "equals",
        }
        return inverses.get(relation)

    def _compose_relations(self, rels1: Set[str], rels2: Set[str]) -> Set[str]:
        """
        Compose two sets of Allen relations using complete composition table.

        For each pair of relations (r1, r2), looks up the composition in
        Allen's composition table to determine all possible resulting relations.

        Args:
            rels1: First set of relations
            rels2: Second set of relations

        Returns:
            Set of all possible composed relations
        """
        result = set()

        for r1 in rels1:
            for r2 in rels2:
                if r1 in self.composition_table and r2 in self.composition_table[r1]:
                    result.update(self.composition_table[r1][r2])

        return result

    def _path_consistency_optimized(self, network: Dict) -> bool:
        """
        FIXED: Complete path consistency with optimizations.

        Enhanced algorithm with:
        - Bidirectional constraint propagation
        - Priority queue for efficient processing
        - Early termination on empty constraints
        - Maintains worklist of constraints to check

        For each triple of events (i, j, k), ensure:
        R(i,k) ⊆ R(i,j) ∘ R(j,k)  AND  R(i,k) ⊆ R(i,m) ∘ R(m,k) for all m

        Args:
            network: Constraint network

        Returns:
            True if consistent
        """
        # Initialize worklist with all constraint pairs
        worklist = []
        constraint_id = 0

        # Build initial worklist
        for e1 in network:
            for e2 in network[e1]:
                # Priority: pairs with fewer relations get processed first
                priority = len(network[e1][e2])
                heapq.heappush(worklist, (priority, constraint_id, (e1, e2)))
                constraint_id += 1

        # Process worklist
        processed = set()

        while worklist:
            _, _, (e1, e2) = heapq.heappop(worklist)

            # Skip if already processed recently
            if (e1, e2) in processed:
                continue

            processed.add((e1, e2))

            # FIXED: Check path consistency through all intermediate nodes
            modified = False

            for e_mid in network:
                if e_mid == e1 or e_mid == e2:
                    continue

                # FIXED: Bidirectional propagation
                # Forward: R(e1, e2) via e_mid
                if e_mid in network[e1] and e2 in network[e_mid]:
                    direct = network[e1].get(e2, set())
                    via_mid = self._compose_relations(
                        network[e1][e_mid], network[e_mid][e2]
                    )

                    if direct:
                        intersection = direct & via_mid

                        # FIXED: Early termination on empty constraint
                        if not intersection:
                            return False  # Inconsistent!

                        if intersection != direct:
                            network[e1][e2] = intersection
                            modified = True

                            # Add affected constraints back to worklist
                            priority = len(intersection)
                            heapq.heappush(
                                worklist, (priority, constraint_id, (e1, e2))
                            )
                            constraint_id += 1
                    else:
                        network[e1][e2] = via_mid
                        modified = True

                # FIXED: Backward propagation
                # Check R(e_mid, e2) via e1
                if e1 in network[e_mid] and e2 in network[e1]:
                    direct_mid = network[e_mid].get(e2, set())
                    via_e1 = self._compose_relations(
                        network[e_mid][e1], network[e1][e2]
                    )

                    if direct_mid:
                        intersection_mid = direct_mid & via_e1

                        if not intersection_mid:
                            return False

                        if intersection_mid != direct_mid:
                            network[e_mid][e2] = intersection_mid
                            modified = True

                            priority = len(intersection_mid)
                            heapq.heappush(
                                worklist, (priority, constraint_id, (e_mid, e2))
                            )
                            constraint_id += 1

            # Clear processed if we made modifications (need to recheck)
            if modified:
                processed.discard((e1, e2))

        return True

    def query_temporal_relation(self, event1_id: str, event2_id: str) -> Set[str]:
        """
        Query possible temporal relations between events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID

        Returns:
            Set of possible Allen relations

        Example:
            >>> relations = temporal.query_temporal_relation('A', 'B')
            >>> 'before' in relations
            True
        """
        network = self._build_constraint_network()
        return network.get(event1_id, {}).get(event2_id, set())

    def find_event_sequence(self) -> List[str]:
        """
        Find valid temporal sequence of events.

        Uses topological sort based on 'before' relations.

        Returns:
            List of event IDs in temporal order

        Example:
            >>> sequence = temporal.find_event_sequence()
            >>> sequence
            ['breakfast', 'work', 'dinner']
        """
        # Topological sort based on 'before' relations
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        all_events = set(self.events.keys())

        for rel in self.temporal_relations:
            if rel["type"] == "before":
                graph[rel["event1"]].append(rel["event2"])
                in_degree[rel["event2"]] += 1

        # Initialize queue with events that have no predecessors
        queue = deque([e for e in all_events if in_degree[e] == 0])
        sequence = []

        while queue:
            event = queue.popleft()
            sequence.append(event)

            for successor in graph[event]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return sequence

    def compute_interval_intersection(
        self, interval1: TimeInterval, interval2: TimeInterval
    ) -> Optional[TimeInterval]:
        """
        FIXED: Compute intersection of two time intervals with uncertainty.

        Args:
            interval1: First interval
            interval2: Second interval

        Returns:
            Intersection interval or None if no overlap
        """
        # Extract effective ranges
        start1 = (
            interval1.start if not interval1.start_uncertain else interval1.start[0]
        )
        end1 = interval1.end if not interval1.end_uncertain else interval1.end[1]
        start2 = (
            interval2.start if not interval2.start_uncertain else interval2.start[0]
        )
        end2 = interval2.end if not interval2.end_uncertain else interval2.end[1]

        # Check for overlap
        if end1 <= start2 or end2 <= start1:
            return None  # No overlap

        # Compute intersection
        intersect_start = max(start1, start2)
        intersect_end = min(end1, end2)

        if intersect_start >= intersect_end:
            return None

        return TimeInterval(
            start=intersect_start,
            end=intersect_end,
            start_uncertain=interval1.start_uncertain or interval2.start_uncertain,
            end_uncertain=interval1.end_uncertain or interval2.end_uncertain,
        )

    def compute_interval_union(
        self, interval1: TimeInterval, interval2: TimeInterval
    ) -> TimeInterval:
        """
        FIXED: Compute union of two time intervals with uncertainty.

        Args:
            interval1: First interval
            interval2: Second interval

        Returns:
            Union interval spanning both
        """
        # Extract effective ranges
        start1 = (
            interval1.start if not interval1.start_uncertain else interval1.start[0]
        )
        end1 = interval1.end if not interval1.end_uncertain else interval1.end[1]
        start2 = (
            interval2.start if not interval2.start_uncertain else interval2.start[0]
        )
        end2 = interval2.end if not interval2.end_uncertain else interval2.end[1]

        # Compute union
        union_start = min(start1, start2)
        union_end = max(end1, end2)

        return TimeInterval(
            start=union_start,
            end=union_end,
            start_uncertain=interval1.start_uncertain or interval2.start_uncertain,
            end_uncertain=interval1.end_uncertain or interval2.end_uncertain,
        )


# ============================================================================
# META-REASONER - FIXED VERSION WITH ENHANCED DIFFICULTY ESTIMATION
# ============================================================================


@dataclass
class ResourceMetrics:
    """
    Resource usage metrics for monitoring.

    Attributes:
        memory_usage: Memory usage in bytes
        cpu_usage: CPU usage percentage
        execution_time: Time taken in seconds
        inferences_made: Number of inferences performed
        search_space_explored: Number of nodes explored in search
    """

    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    execution_time: float = 0.0
    inferences_made: int = 0
    search_space_explored: int = 0


class MetaReasoner:
    """
    COMPLETE FIXED IMPLEMENTATION: Meta-level reasoning.

    Reasons about reasoning itself - strategy selection, resource allocation,
    performance monitoring, and adaptive strategy selection.

    Meta-reasoning involves:
    - Selecting appropriate reasoning strategies
    - Allocating computational resources
    - Monitoring performance (time, memory, CPU, inferences, search space)
    - Learning from experience
    - Adapting to problem characteristics

    FIXED Features:
    - Enhanced difficulty estimation with clause complexity analysis
    - Variable counting and nesting depth analysis
    - Past performance consideration
    - Comprehensive resource monitoring
    - Strategy registration with cost/quality estimates
    - Resource allocation across problems
    - Performance tracking and statistics
    - Adaptive strategy selection
    - Execution monitoring

    Example:
        >>> meta = MetaReasoner()
        >>> meta.register_strategy(
        ...     'fast_solver',
        ...     strategy_func=fast_solve,
        ...     cost=1.0,
        ...     expected_quality=0.7
        ... )
        >>> meta.register_strategy(
        ...     'thorough_solver',
        ...     strategy_func=thorough_solve,
        ...     cost=5.0,
        ...     expected_quality=0.95
        ... )
        >>> strategy = meta.select_strategy(
        ...     problem=my_problem,
        ...     available_time=2.0,
        ...     quality_threshold=0.8
        ... )
    """

    def __init__(self):
        """Initialize meta-reasoner."""
        self.reasoning_strategies = {}
        self.performance_history = defaultdict(list)
        self.resource_budgets = {}
        self.problem_database = []  # FIXED: Store solved problems for similarity matching

    def register_strategy(
        self, name: str, strategy_func: Callable, cost: float, expected_quality: float
    ):
        """
        Register reasoning strategy with cost and quality estimates.

        Args:
            name: Strategy name
            strategy_func: Function implementing the strategy
            cost: Estimated computational cost (time in seconds)
            expected_quality: Expected quality/accuracy [0, 1]

        Example:
            >>> def my_solver(problem, timeout):
            ...     # Solve problem
            ...     return {'success': True, 'quality': 0.9}
            >>> meta.register_strategy('solver1', my_solver, cost=2.0, expected_quality=0.85)
        """
        self.reasoning_strategies[name] = {
            "function": strategy_func,
            "cost": cost,
            "expected_quality": expected_quality,
            "success_rate": 0.5,
            "avg_time": cost,
        }

    def select_strategy(
        self, problem: Any, available_time: float, quality_threshold: float
    ) -> Optional[str]:
        """
        Select best reasoning strategy given constraints.

        Selection criteria:
        - Strategy must fit in available time
        - Strategy must meet quality threshold
        - Maximize quality/cost ratio weighted by success rate

        Args:
            problem: Problem to solve
            available_time: Time budget (seconds)
            quality_threshold: Minimum acceptable quality [0, 1]

        Returns:
            Name of selected strategy or None if no suitable strategy

        Example:
            >>> strategy = meta.select_strategy(
            ...     problem={'difficulty': 'hard'},
            ...     available_time=10.0,
            ...     quality_threshold=0.9
            ... )
            >>> strategy
            'thorough_solver'
        """
        # Filter strategies that meet constraints
        candidates = []

        for name, strategy in self.reasoning_strategies.items():
            if (
                strategy["avg_time"] <= available_time
                and strategy["expected_quality"] >= quality_threshold
            ):
                # Score based on quality/cost ratio weighted by success rate
                score = (strategy["expected_quality"] * strategy["success_rate"]) / max(
                    strategy["cost"], 0.1
                )
                candidates.append((name, score))

        if not candidates:
            return None

        # Select best strategy
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def execute_with_monitoring(
        self, strategy_name: str, problem: Any, timeout: float
    ) -> Dict[str, Any]:
        """
        FIXED: Execute strategy with comprehensive resource monitoring.

        Tracks:
        - Execution time
        - Memory usage
        - CPU usage (if psutil available)
        - Number of inferences (if tracked by strategy)
        - Search space explored (if tracked by strategy)
        - Success/failure
        - Quality metrics

        Args:
            strategy_name: Name of strategy to execute
            problem: Problem to solve
            timeout: Maximum execution time

        Returns:
            Dict with execution results and comprehensive metrics

        Example:
            >>> result = meta.execute_with_monitoring('solver1', problem, timeout=5.0)
            >>> result
            {
                'success': True,
                'result': {...},
                'execution_time': 2.3,
                'memory_usage': 15728640,
                'cpu_usage': 45.2,
                'inferences_made': 127,
                'search_space_explored': 543,
                'strategy': 'solver1'
            }
        """
        if strategy_name not in self.reasoning_strategies:
            return {"success": False, "reason": "Unknown strategy"}

        strategy = self.reasoning_strategies[strategy_name]

        # FIXED: Track resource usage
        metrics = ResourceMetrics()

        # Get initial resource state
        try:
            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            cpu_percent_start = process.cpu_percent(interval=None)
        except ImportError:
            process = None
            initial_memory = 0
            cpu_percent_start = 0

        start_time = time.time()

        try:
            result = strategy["function"](problem, timeout=timeout)
            execution_time = time.time() - start_time

            # FIXED: Collect comprehensive metrics
            metrics.execution_time = execution_time

            # Memory usage
            if process:
                final_memory = process.memory_info().rss
                metrics.memory_usage = final_memory - initial_memory
                metrics.cpu_usage = (
                    process.cpu_percent(interval=None) - cpu_percent_start
                )

            # Extract strategy-reported metrics
            metrics.inferences_made = result.get("inferences_made", 0)
            metrics.search_space_explored = result.get("search_space_explored", 0)

            # Record performance
            self.performance_history[strategy_name].append(
                {
                    "time": execution_time,
                    "success": result.get("success", False),
                    "quality": result.get("quality", 0.0),
                    "memory_usage": metrics.memory_usage,
                    "cpu_usage": metrics.cpu_usage,
                    "inferences": metrics.inferences_made,
                    "search_space": metrics.search_space_explored,
                }
            )

            # Update strategy statistics
            self._update_strategy_stats(strategy_name)

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage,
                "inferences_made": metrics.inferences_made,
                "search_space_explored": metrics.search_space_explored,
                "strategy": strategy_name,
            }

        except Exception as e:
            execution_time = time.time() - start_time

            # Still collect available metrics on failure
            if process:
                final_memory = process.memory_info().rss
                metrics.memory_usage = final_memory - initial_memory
                metrics.cpu_usage = (
                    process.cpu_percent(interval=None) - cpu_percent_start
                )

            metrics.execution_time = execution_time

            self.performance_history[strategy_name].append(
                {
                    "time": execution_time,
                    "success": False,
                    "quality": 0.0,
                    "memory_usage": metrics.memory_usage,
                    "cpu_usage": metrics.cpu_usage,
                    "inferences": 0,
                    "search_space": 0,
                }
            )

            self._update_strategy_stats(strategy_name)

            return {
                "success": False,
                "reason": str(e),
                "execution_time": execution_time,
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage,
                "strategy": strategy_name,
            }

    def _update_strategy_stats(self, strategy_name: str):
        """
        Update strategy statistics based on history.

        Computes running averages for success rate, time, and quality.

        Args:
            strategy_name: Strategy to update
        """
        history = self.performance_history[strategy_name]

        if not history:
            return

        recent_history = history[-10:]  # Last 10 executions

        success_count = sum(1 for h in recent_history if h["success"])
        success_rate = success_count / len(recent_history)

        try:
            import numpy as np

            avg_time = np.mean([h["time"] for h in recent_history])
            avg_quality = np.mean([h["quality"] for h in recent_history])
        except ImportError:
            avg_time = sum(h["time"] for h in recent_history) / len(recent_history)
            avg_quality = sum(h["quality"] for h in recent_history) / len(
                recent_history
            )

        strategy = self.reasoning_strategies[strategy_name]
        strategy["success_rate"] = success_rate
        strategy["avg_time"] = avg_time
        strategy["expected_quality"] = avg_quality

    def allocate_resources(
        self, problems: List[Any], total_time: float
    ) -> Dict[int, float]:
        """
        Allocate time budget across multiple problems using enhanced difficulty estimation.

        Uses difficulty estimates to allocate more time to harder problems.

        Args:
            problems: List of problems to solve
            total_time: Total time budget

        Returns:
            Dict mapping problem indices to time allocations

        Example:
            >>> allocations = meta.allocate_resources(
            ...     problems=[easy_problem, hard_problem],
            ...     total_time=10.0
            ... )
            >>> allocations
            {0: 3.0, 1: 7.0}  # More time for harder problem
        """
        if not problems:
            return {}

        # FIXED: Use enhanced difficulty estimation
        difficulties = [self._estimate_difficulty(p) for p in problems]

        # Allocate time proportional to difficulty
        total_difficulty = sum(difficulties)

        allocations = {}
        for i, difficulty in enumerate(difficulties):
            if total_difficulty > 0:
                allocations[i] = (difficulty / total_difficulty) * total_time
            else:
                allocations[i] = total_time / len(problems)

        return allocations

    def _estimate_difficulty(self, problem: Any) -> float:
        """
        FIXED: Estimate problem difficulty using multiple heuristics.

        Enhanced estimation considers:
        - Clause complexity (for Clause problems)
        - Variable count
        - Function nesting depth
        - Past performance on similar problems
        - Problem size/structure

        Args:
            problem: Problem to estimate

        Returns:
            Difficulty score (higher = harder)

        Example:
            >>> difficulty = meta._estimate_difficulty(clause_problem)
            >>> difficulty
            15.7  # Complex clause with variables and nesting
        """
        difficulty = 1.0

        # FIXED: Enhanced estimation for Clause problems
        if isinstance(problem, Clause):
            # Base difficulty from clause size
            difficulty += len(problem.literals) * 2

            # Penalty for non-Horn clauses (harder to solve)
            if not problem.is_horn_clause():
                difficulty += 5

            # Penalty for variables (requires unification)
            var_count = self._count_variables(problem)
            difficulty += var_count * 1.5

            # Penalty for function nesting
            max_depth = self._max_term_depth(problem)
            difficulty += max_depth * 3

            # FIXED: Adjustment based on past performance
            similar_problems = self._find_similar_problems(problem)
            if similar_problems:
                # Use historical data to adjust difficulty
                try:
                    import numpy as np

                    avg_time = np.mean([p["time"] for p in similar_problems])
                    # Normalize to 1 second baseline
                    difficulty *= avg_time / 1.0
                except ImportError:
                    avg_time = sum(p["time"] for p in similar_problems) / len(
                        similar_problems
                    )
                    difficulty *= avg_time / 1.0

        # Handle other problem types
        elif isinstance(problem, dict):
            # Dictionary-based problems
            difficulty = len(problem) * 1.5

            # Check for nested structures
            def count_nested(obj, depth=0):
                if isinstance(obj, dict):
                    return depth + sum(count_nested(v, depth + 1) for v in obj.values())
                elif isinstance(obj, (list, tuple)):
                    return depth + sum(count_nested(item, depth + 1) for item in obj)
                return depth

            nesting = count_nested(problem)
            difficulty += nesting * 0.5

        elif isinstance(problem, (list, tuple)):
            difficulty = len(problem) * 1.2

        elif isinstance(problem, str):
            # String-based problems
            difficulty = len(problem.split()) * 0.8

        return max(difficulty, 1.0)

    def _count_variables(self, clause: Clause) -> int:
        """
        FIXED: Count unique variables in clause.

        Args:
            clause: Clause to analyze

        Returns:
            Number of unique variables
        """
        variables = set()

        for literal in clause.literals:
            for term in literal.terms:
                self._collect_variables_from_term(term, variables)

        return len(variables)

    def _collect_variables_from_term(self, term: Term, variables: Set[str]):
        """
        Recursively collect variables from term.

        Args:
            term: Term to analyze
            variables: Set to accumulate variable names
        """
        if isinstance(term, Variable):
            variables.add(term.name)
        elif isinstance(term, Function):
            for arg in term.args:
                self._collect_variables_from_term(arg, variables)

    def _max_term_depth(self, clause: Clause) -> int:
        """
        FIXED: Calculate maximum term nesting depth in clause.

        Args:
            clause: Clause to analyze

        Returns:
            Maximum nesting depth
        """
        max_depth = 0

        for literal in clause.literals:
            for term in literal.terms:
                depth = self._term_depth(term)
                max_depth = max(max_depth, depth)

        return max_depth

    def _term_depth(self, term: Term) -> int:
        """
        Calculate nesting depth of a term.

        Args:
            term: Term to analyze

        Returns:
            Nesting depth (0 for constants/variables)
        """
        if isinstance(term, (Variable, Constant)):
            return 0
        elif isinstance(term, Function):
            if not term.args:
                return 1
            return 1 + max(self._term_depth(arg) for arg in term.args)
        return 0

    def _find_similar_problems(self, problem: Clause) -> List[Dict]:
        """
        FIXED: Find similar problems in database for difficulty estimation.

        Uses structural similarity to find comparable past problems.

        Args:
            problem: Problem to find similar cases for

        Returns:
            List of similar problem records with performance data
        """
        similar = []

        # Calculate signature for the problem
        target_signature = self._compute_problem_signature(problem)

        for record in self.problem_database:
            if "signature" in record and record["signature"] == target_signature:
                similar.append(record)

        # Return up to 10 most similar problems
        return similar[:10]

    def _compute_problem_signature(self, problem: Clause) -> str:
        """
        Compute structural signature of problem.

        Args:
            problem: Problem to compute signature for

        Returns:
            Signature string
        """
        if not isinstance(problem, Clause):
            return "unknown"

        # Create signature based on structure
        parts = []

        # Clause type
        if problem.is_unit_clause():
            parts.append("unit")
        elif problem.is_horn_clause():
            parts.append("horn")
        else:
            parts.append("general")

        # Size
        parts.append(f"size:{len(problem.literals)}")

        # Variable count
        var_count = self._count_variables(problem)
        parts.append(f"vars:{var_count}")

        # Max depth
        max_depth = self._max_term_depth(problem)
        parts.append(f"depth:{max_depth}")

        return ":".join(parts)

    def record_problem_performance(
        self, problem: Any, time_taken: float, success: bool, strategy: str
    ):
        """
        Record performance data for problem (for future difficulty estimation).

        Args:
            problem: Problem that was solved
            time_taken: Time taken to solve
            success: Whether solution was found
            strategy: Strategy used
        """
        if isinstance(problem, Clause):
            signature = self._compute_problem_signature(problem)

            self.problem_database.append(
                {
                    "problem": str(problem),
                    "signature": signature,
                    "time": time_taken,
                    "success": success,
                    "strategy": strategy,
                }
            )

    def explain_strategy_choice(self, strategy_name: str, problem: Any) -> str:
        """
        Explain why a strategy was chosen.

        Provides human-readable explanation of strategy selection.

        Args:
            strategy_name: Selected strategy name
            problem: Problem being solved

        Returns:
            Explanation string

        Example:
            >>> explanation = meta.explain_strategy_choice('fast_solver', problem)
            >>> print(explanation)
            Selected strategy: fast_solver
            Expected quality: 0.85
            Estimated cost: 2.0s
            Success rate: 87%
            Based on 15 past executions
            Problem difficulty: 12.5
        """
        if strategy_name not in self.reasoning_strategies:
            return "Unknown strategy"

        strategy = self.reasoning_strategies[strategy_name]

        explanation = f"Selected strategy: {strategy_name}\n"
        explanation += f"Expected quality: {strategy['expected_quality']:.2f}\n"
        explanation += f"Estimated cost: {strategy['cost']:.2f}s\n"
        explanation += f"Success rate: {strategy['success_rate']:.2%}\n"

        if self.performance_history[strategy_name]:
            explanation += f"Based on {len(self.performance_history[strategy_name])} past executions\n"

        # FIXED: Add problem difficulty analysis
        difficulty = self._estimate_difficulty(problem)
        explanation += f"Problem difficulty: {difficulty:.1f}"

        return explanation


# ============================================================================
# PROOF LEARNER - FIXED VERSION WITH ENHANCED PATTERN EXTRACTION
# ============================================================================


@dataclass
class ProofPattern:
    """
    FIXED: Rich proof pattern representation.

    Captures comprehensive information about proof structure:
    - Tree structure representation
    - Rule application sequence
    - Clause types used
    - Variable flow through proof
    - Critical inference steps
    - Historical success metrics
    - Problem type applicability

    Attributes:
        structure: Tree structure as string
        rules_sequence: Ordered list of rules applied
        clause_signatures: Types of clauses used
        variable_flow: How variables propagate through proof
        critical_steps: Indices of key inference points
        success_rate: Historical success rate for this pattern
        avg_depth: Average proof depth
        problem_types: Set of problem types this works for
        substitution_patterns: Common substitution patterns
        polarity_patterns: Literal polarity patterns
    """

    structure: str
    rules_sequence: List[str]
    clause_signatures: List[str]
    variable_flow: Dict[str, List[str]]
    critical_steps: List[int]
    success_rate: float
    avg_depth: float
    problem_types: Set[str]
    substitution_patterns: Dict[str, str] = field(default_factory=dict)
    polarity_patterns: List[str] = field(default_factory=list)


class ProofLearner:
    """
    COMPLETE FIXED IMPLEMENTATION: Learn from successful proofs.

    Extracts patterns and heuristics from successful proofs to improve
    future reasoning performance.

    FIXED Features:
    - Rich pattern extraction with substitution tracking
    - Literal polarity pattern analysis
    - Term structure pattern recognition
    - Proof branch structure analysis
    - Multi-faceted similarity computation
    - Structure-based similarity
    - Token-based similarity
    - Type compatibility checking
    - Variable pattern matching
    - Pattern extraction from proof trees
    - Tactic frequency tracking
    - Goal type classification
    - Similarity-based proof search
    - Learned heuristics

    Example:
        >>> learner = ProofLearner()
        >>> # After successful proof
        >>> learner.learn_from_proof(proof_tree, goal_clause)
        >>> # Later, get suggestions
        >>> tactics = learner.suggest_tactics(new_goal)
        >>> tactics
        ['resolution', 'tableau', 'model_elimination']
    """

    def __init__(self):
        """Initialize proof learner."""
        self.proof_patterns = defaultdict(int)
        self.successful_tactics = defaultdict(list)
        self.proof_database = []
        self.clause_pattern_cache = {}
        self.rich_patterns = []  # FIXED: Store ProofPattern objects

    def learn_from_proof(self, proof: ProofNode, goal: Clause):
        """
        FIXED: Extract comprehensive patterns from successful proof.

        Learns:
        - Proof structure (tree pattern with full context)
        - Tactic sequence with depth information
        - Goal characteristics
        - Substitution patterns
        - Variable flow
        - Critical inference steps

        Args:
            proof: Proof tree (ProofNode)
            goal: Goal that was proven

        Example:
            >>> learner.learn_from_proof(proof_tree, goal_clause)
            >>> len(learner.proof_database)
            1
        """
        # Extract comprehensive proof structure
        pattern = self._extract_pattern(proof)
        self.proof_patterns[pattern] += 1

        # Extract tactics
        tactics = self._extract_tactics(proof)
        goal_type = self._classify_goal(goal)
        self.successful_tactics[goal_type].extend(tactics)

        # Extract clause patterns used in proof
        clause_patterns = self._extract_clause_patterns(proof)

        # FIXED: Extract rich pattern information
        rich_pattern = self._extract_rich_pattern(proof, goal)
        self.rich_patterns.append(rich_pattern)

        # Store comprehensive proof information
        self.proof_database.append(
            {
                "goal": str(goal),
                "goal_clause": goal,
                "proof": proof,
                "pattern": pattern,
                "tactics": tactics,
                "depth": proof.depth,
                "goal_type": goal_type,
                "clause_patterns": clause_patterns,
                "goal_size": len(goal.literals),
                "goal_signature": self._compute_clause_signature(goal),
                "rich_pattern": rich_pattern,  # FIXED: Include rich pattern
            }
        )

    def _extract_rich_pattern(self, proof: ProofNode, goal: Clause) -> ProofPattern:
        """
        FIXED: Extract rich pattern with comprehensive information.

        Captures:
        - Tree structure
        - Rule sequence
        - Clause signatures
        - Variable flow (how variables propagate)
        - Critical steps (key inferences)
        - Substitution patterns
        - Polarity patterns

        Args:
            proof: Proof tree
            goal: Goal clause

        Returns:
            ProofPattern with complete information
        """
        # Extract structure
        structure = self._extract_pattern(proof)

        # Extract rule sequence
        rules_sequence = self._extract_tactics(proof)

        # Extract clause signatures
        clause_signatures = self._extract_clause_patterns(proof)

        # FIXED: Extract variable flow
        variable_flow = self._extract_variable_flow(proof)

        # FIXED: Identify critical steps
        critical_steps = self._identify_critical_steps(proof)

        # FIXED: Extract substitution patterns
        substitution_patterns = self._extract_substitution_patterns(proof)

        # FIXED: Extract polarity patterns
        polarity_patterns = self._extract_polarity_patterns(proof)

        # Compute success metrics
        goal_type = self._classify_goal(goal)

        return ProofPattern(
            structure=structure,
            rules_sequence=rules_sequence,
            clause_signatures=clause_signatures,
            variable_flow=variable_flow,
            critical_steps=critical_steps,
            success_rate=1.0,  # Initial success rate
            avg_depth=proof.depth,
            problem_types={goal_type},
            substitution_patterns=substitution_patterns,
            polarity_patterns=polarity_patterns,
        )

    def _extract_variable_flow(self, proof: ProofNode) -> Dict[str, List[str]]:
        """
        FIXED: Extract how variables flow through proof.

        Tracks which variables appear at each step and how they're
        instantiated or propagated.

        Args:
            proof: Proof tree

        Returns:
            Dict mapping variable names to their flow through proof
        """
        variable_flow = defaultdict(list)

        def track_variables(node: ProofNode, path: str):
            # Extract variables from conclusion
            if isinstance(node.conclusion, Clause):
                for literal in node.conclusion.literals:
                    for term in literal.terms:
                        if isinstance(term, Variable):
                            variable_flow[term.name].append(path)

            # Recurse to premises
            for i, premise in enumerate(node.premises):
                if isinstance(premise, ProofNode):
                    track_variables(premise, f"{path}/{i}")

        track_variables(proof, "root")

        return dict(variable_flow)

    def _identify_critical_steps(self, proof: ProofNode) -> List[int]:
        """
        FIXED: Identify critical inference steps in proof.

        Critical steps are those where:
        - Key resolution happens
        - Important unification occurs
        - Proof branches significantly

        Args:
            proof: Proof tree

        Returns:
            List of step indices that are critical
        """
        critical = []

        def analyze_node(node: ProofNode, step_num: int) -> int:
            # Resolution steps are critical
            if "resolution" in node.rule_used.lower():
                critical.append(step_num)

            # Steps with many premises are critical (branching)
            if len(node.premises) > 2:
                critical.append(step_num)

            # Steps at depth transitions
            if node.premises:
                min_child_depth = (
                    min(p.depth for p in node.premises if isinstance(p, ProofNode))
                    if any(isinstance(p, ProofNode) for p in node.premises)
                    else node.depth
                )

                if node.depth - min_child_depth > 3:
                    critical.append(step_num)

            # Recurse
            current_step = step_num + 1
            for premise in node.premises:
                if isinstance(premise, ProofNode):
                    current_step = analyze_node(premise, current_step)

            return current_step

        analyze_node(proof, 0)

        return critical

    def _extract_substitution_patterns(self, proof: ProofNode) -> Dict[str, str]:
        """
        FIXED: Extract substitution patterns from proof.

        Identifies common patterns like:
        - X -> constant
        - X -> f(Y)
        - Multiple variables unified

        Args:
            proof: Proof tree

        Returns:
            Dict of substitution patterns
        """
        patterns = {}

        def extract_from_metadata(node: ProofNode):
            if "substitution" in node.metadata:
                subst = node.metadata["substitution"]
                for var, term in subst.items():
                    if isinstance(term, Constant):
                        patterns[f"{var}_type"] = "constant"
                    elif isinstance(term, Function):
                        patterns[f"{var}_type"] = f"function_arity_{len(term.args)}"
                    elif isinstance(term, Variable):
                        patterns[f"{var}_type"] = "variable"

            for premise in node.premises:
                if isinstance(premise, ProofNode):
                    extract_from_metadata(premise)

        extract_from_metadata(proof)

        return patterns

    def _extract_polarity_patterns(self, proof: ProofNode) -> List[str]:
        """
        FIXED: Extract literal polarity patterns.

        Tracks patterns like:
        - "pos-neg-pos": alternating polarities
        - "all-pos": all positive literals
        - "majority-neg": mostly negative

        Args:
            proof: Proof tree

        Returns:
            List of polarity pattern strings
        """
        patterns = []

        def extract_from_node(node: ProofNode):
            if isinstance(node.conclusion, Clause):
                polarities = []
                for literal in node.conclusion.literals:
                    polarities.append("neg" if literal.negated else "pos")

                if polarities:
                    patterns.append("-".join(polarities))

            for premise in node.premises:
                if isinstance(premise, ProofNode):
                    extract_from_node(premise)

        extract_from_node(proof)

        return patterns

    def _extract_pattern(self, proof: ProofNode, max_depth: int = 5) -> str:
        """
        Extract enhanced structural pattern from proof.

        Creates a rich representation that captures:
        - Rule names and application order
        - Clause characteristics at each step
        - Proof tree structure
        - Abstracted literal patterns

        Args:
            proof: Proof tree
            max_depth: Maximum depth to extract (prevents overly long patterns)

        Returns:
            Rich pattern string with structural and semantic information

        Example:
            >>> pattern = learner._extract_pattern(proof)
            >>> pattern
            'resolution[unit:2,horn](tableau[small:3](axiom),modus_ponens[unit:1](axiom,axiom))'
        """
        if proof.depth > max_depth:
            return f"{proof.rule_used}[...]"

        # Get clause characteristics for current node
        clause_info = self._get_clause_characteristics(proof.conclusion)

        # Build pattern recursively
        if not proof.premises:
            return f"{proof.rule_used}[{clause_info}]"

        child_patterns = []
        for premise in proof.premises:
            if isinstance(premise, ProofNode):
                child_patterns.append(self._extract_pattern(premise, max_depth))
            elif isinstance(premise, Clause):
                # Include clause characteristics even for leaf clauses
                child_info = self._get_clause_characteristics(premise)
                child_patterns.append(f"clause[{child_info}]")

        if child_patterns:
            return f"{proof.rule_used}[{clause_info}]({','.join(child_patterns)})"
        else:
            return f"{proof.rule_used}[{clause_info}]"

    def _get_clause_characteristics(self, clause: Clause) -> str:
        """
        Get compact characteristics of a clause for pattern matching.

        Args:
            clause: Clause to characterize

        Returns:
            String describing clause characteristics

        Example:
            >>> self._get_clause_characteristics(some_clause)
            'horn:3,pos:2,neg:1'
        """
        if not isinstance(clause, Clause):
            return "unknown"

        characteristics = []

        # Clause type
        if clause.is_unit_clause():
            characteristics.append("unit")
        elif clause.is_horn_clause():
            characteristics.append("horn")

        # Size
        size = len(clause.literals)
        characteristics.append(f"size:{size}")

        # Polarity distribution
        pos_count = sum(1 for lit in clause.literals if not lit.negated)
        neg_count = size - pos_count
        if pos_count > 0:
            characteristics.append(f"pos:{pos_count}")
        if neg_count > 0:
            characteristics.append(f"neg:{neg_count}")

        return ",".join(characteristics)

    def _extract_clause_patterns(self, proof: ProofNode) -> List[str]:
        """
        Extract all clause patterns used in proof.

        Identifies recurring clause structures that were useful
        in this proof.

        Args:
            proof: Proof tree

        Returns:
            List of clause pattern signatures
        """
        patterns = []

        # Get pattern for conclusion
        if isinstance(proof.conclusion, Clause):
            patterns.append(self._compute_clause_signature(proof.conclusion))

        # Recursively extract from premises
        for premise in proof.premises:
            if isinstance(premise, ProofNode):
                patterns.extend(self._extract_clause_patterns(premise))
            elif isinstance(premise, Clause):
                patterns.append(self._compute_clause_signature(premise))

        return patterns

    def _compute_clause_signature(self, clause: Clause) -> str:
        """
        Compute abstract signature of clause for similarity matching.

        Creates an abstracted representation that captures structural
        properties while ignoring specific variable names.

        Args:
            clause: Clause to compute signature for

        Returns:
            Signature string

        Example:
            >>> self._compute_clause_signature(clause)
            'horn:3:pos(P,2),neg(Q,1)'
        """
        if not isinstance(clause, Clause):
            return "empty"

        # Cache signatures to avoid recomputation
        clause_str = str(clause)
        if clause_str in self.clause_pattern_cache:
            return self.clause_pattern_cache[clause_str]

        # Build signature components
        parts = []

        # Type
        if clause.is_unit_clause():
            parts.append("unit")
        elif clause.is_horn_clause():
            parts.append("horn")
        else:
            parts.append("general")

        # Size
        parts.append(str(len(clause.literals)))

        # Abstract predicate structure
        predicate_counts = defaultdict(int)
        for lit in clause.literals:
            polarity = "pos" if not lit.negated else "neg"
            # Use arity instead of specific predicate name for better generalization
            arity = len(lit.terms) if hasattr(lit, "terms") else 0
            key = f"{polarity}(arity:{arity})"
            predicate_counts[key] += 1

        # Sort for consistency
        pred_parts = [f"{k}:{v}" for k, v in sorted(predicate_counts.items())]
        parts.append(",".join(pred_parts))

        signature = ":".join(parts)
        self.clause_pattern_cache[clause_str] = signature

        return signature

    def _extract_tactics(self, proof: ProofNode) -> List[str]:
        """
        Extract sequence of tactics used with context.

        Flattens proof tree into tactic sequence with additional
        context about when each tactic was applied.

        Args:
            proof: Proof tree

        Returns:
            List of tactic names with depth annotations
        """
        tactics = [f"{proof.rule_used}@depth{proof.depth}"]

        for premise in proof.premises:
            if isinstance(premise, ProofNode):
                tactics.extend(self._extract_tactics(premise))

        return tactics

    def _classify_goal(self, goal: Clause) -> str:
        """
        Classify goal type with enhanced categories.

        Provides more fine-grained classification for better
        tactic suggestion.

        Args:
            goal: Goal clause

        Returns:
            Goal type string
        """
        if not isinstance(goal, Clause):
            return "unknown"

        size = len(goal.literals)

        # Primary classification
        if goal.is_unit_clause():
            return "unit"
        elif goal.is_horn_clause():
            if size <= 2:
                return "horn-small"
            elif size <= 5:
                return "horn-medium"
            else:
                return "horn-large"
        else:
            # General clause classification by size
            if size <= 3:
                return "general-small"
            elif size <= 6:
                return "general-medium"
            else:
                return "general-large"

    def suggest_tactics(self, goal: Clause, top_k: int = 5) -> List[str]:
        """
        Suggest tactics based on learned patterns with confidence scores.

        Returns tactics sorted by success frequency for this goal type,
        with additional context from similar successful proofs.

        Args:
            goal: Goal to prove
            top_k: Number of tactics to return

        Returns:
            List of suggested tactic names (sorted by effectiveness)

        Example:
            >>> tactics = learner.suggest_tactics(goal, top_k=3)
            >>> tactics
            ['resolution', 'tableau', 'model_elimination']
        """
        goal_type = self._classify_goal(goal)
        goal_sig = self._compute_clause_signature(goal)

        # Collect tactics from multiple sources
        tactic_scores = defaultdict(float)

        # Source 1: Direct goal type matching
        if goal_type in self.successful_tactics:
            for tactic in self.successful_tactics[goal_type]:
                # Extract base tactic name (remove depth annotation)
                base_tactic = tactic.split("@")[0]
                tactic_scores[base_tactic] += 1.0

        # Source 2: Similar clause signatures
        for proof_entry in self.proof_database:
            if proof_entry["goal_signature"] == goal_sig:
                for tactic in proof_entry["tactics"]:
                    base_tactic = tactic.split("@")[0]
                    tactic_scores[base_tactic] += (
                        2.0  # Higher weight for exact signature match
                    )

        # Source 3: Similar goal size
        goal_size = len(goal.literals)
        for proof_entry in self.proof_database:
            if abs(proof_entry["goal_size"] - goal_size) <= 1:
                for tactic in proof_entry["tactics"]:
                    base_tactic = tactic.split("@")[0]
                    tactic_scores[base_tactic] += (
                        0.5  # Lower weight for size similarity
                    )

        # Sort by score
        if tactic_scores:
            sorted_tactics = sorted(
                tactic_scores.items(), key=lambda x: x[1], reverse=True
            )
            return [tactic for tactic, _ in sorted_tactics[:top_k]]

        # Fallback: general-purpose tactics
        return [
            "tableau",
            "resolution",
            "model_elimination",
            "backward_chaining",
            "forward_chaining",
        ][:top_k]

    def get_similar_proofs(
        self, goal: Clause, k: int = 5, use_structural_similarity: bool = True
    ) -> List[Dict]:
        """
        Find similar proofs from database with enhanced similarity metrics.

        Uses multiple similarity measures:
        - Clause signature matching (structural)
        - Token-based similarity (syntactic)
        - Goal type similarity (semantic)

        Args:
            goal: Goal to find similar proofs for
            k: Number of similar proofs to return
            use_structural_similarity: Whether to use structural matching

        Returns:
            List of similar proof entries with similarity scores

        Example:
            >>> similar = learner.get_similar_proofs(new_goal, k=3)
            >>> for proof_entry in similar:
            ...     print(f"Goal: {proof_entry['goal']}, Score: {proof_entry['similarity']:.2f}")
        """
        goal_str = str(goal)
        goal_sig = self._compute_clause_signature(goal)
        goal_type = self._classify_goal(goal)
        goal_size = len(goal.literals)

        # Score proofs by multiple similarity measures
        scored_proofs = []

        for proof_entry in self.proof_database:
            similarity_score = 0.0

            # Structural similarity (clause signature)
            if use_structural_similarity and proof_entry["goal_signature"] == goal_sig:
                similarity_score += 10.0  # Strong match

            # Goal type similarity
            if proof_entry["goal_type"] == goal_type:
                similarity_score += 5.0

            # Size similarity
            size_diff = abs(proof_entry["goal_size"] - goal_size)
            similarity_score += max(0, 3.0 - size_diff)  # Closer sizes = higher score

            # Token-based similarity
            token_sim = self._compute_similarity_basic(goal_str, proof_entry["goal"])
            similarity_score += token_sim * 2.0

            # Add similarity to result
            result_entry = proof_entry.copy()
            result_entry["similarity"] = similarity_score
            scored_proofs.append(result_entry)

        # Sort by similarity
        scored_proofs.sort(key=lambda x: x["similarity"], reverse=True)

        return scored_proofs[:k]

    def _compute_similarity(self, goal1: Clause, goal2: Clause) -> float:
        """
        FIXED: Multi-faceted similarity scoring.

        Uses multiple dimensions:
        - Structural similarity (clause patterns)
        - Token similarity (predicate names)
        - Type similarity (clause signatures)
        - Variable pattern similarity

        Args:
            goal1: First goal clause
            goal2: Second goal clause

        Returns:
            Similarity score [0, 1]
        """
        similarity = 0.0
        weights = {"structure": 0.3, "tokens": 0.2, "types": 0.3, "variables": 0.2}

        # Structural similarity (clause patterns)
        if goal1.is_horn_clause() == goal2.is_horn_clause():
            similarity += weights["structure"]

        # Token similarity (predicate names)
        tokens1 = self._extract_predicates(goal1)
        tokens2 = self._extract_predicates(goal2)
        if tokens1 or tokens2:
            jaccard = (
                len(tokens1 & tokens2) / len(tokens1 | tokens2)
                if (tokens1 | tokens2)
                else 0
            )
            similarity += weights["tokens"] * jaccard

        # Type similarity (clause signatures)
        if self._compute_clause_signature(goal1) == self._compute_clause_signature(
            goal2
        ):
            similarity += weights["types"]

        # Variable pattern similarity
        var_pattern1 = self._extract_variable_pattern(goal1)
        var_pattern2 = self._extract_variable_pattern(goal2)
        var_sim = self._pattern_similarity(var_pattern1, var_pattern2)
        similarity += weights["variables"] * var_sim

        return similarity

    def _extract_predicates(self, clause: Clause) -> Set[str]:
        """
        FIXED: Extract predicate names from clause.

        Args:
            clause: Clause to extract from

        Returns:
            Set of predicate names
        """
        predicates = set()

        for literal in clause.literals:
            predicates.add(literal.predicate)

        return predicates

    def _extract_variable_pattern(self, clause: Clause) -> str:
        """
        FIXED: Extract variable pattern from clause.

        Creates abstract pattern of variable usage.

        Args:
            clause: Clause to analyze

        Returns:
            Variable pattern string
        """
        var_count = 0
        pattern_parts = []

        for literal in clause.literals:
            vars_in_literal = []
            for term in literal.terms:
                if isinstance(term, Variable):
                    vars_in_literal.append("V")
                    var_count += 1
                elif isinstance(term, Constant):
                    vars_in_literal.append("C")
                elif isinstance(term, Function):
                    vars_in_literal.append("F")

            if vars_in_literal:
                pattern_parts.append("".join(vars_in_literal))

        return f"vars:{var_count}:{'_'.join(pattern_parts)}"

    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        FIXED: Compute similarity between variable patterns.

        Args:
            pattern1: First pattern
            pattern2: Second pattern

        Returns:
            Similarity score [0, 1]
        """
        if pattern1 == pattern2:
            return 1.0

        # Extract variable counts
        try:
            count1 = int(pattern1.split(":")[1])
            count2 = int(pattern2.split(":")[1])

            # Similarity based on variable count difference
            max_count = max(count1, count2, 1)
            diff = abs(count1 - count2)

            return 1.0 - (diff / max_count)
        except Exception:
            return 0.0

    def _compute_similarity_basic(self, goal1: str, goal2: str) -> float:
        """
        Compute similarity between goals using Jaccard similarity.

        Uses token-based Jaccard similarity on word tokens.

        Args:
            goal1: First goal string
            goal2: Second goal string

        Returns:
            Similarity score [0, 1]
        """
        # Simple Jaccard similarity on tokens
        tokens1 = set(goal1.split())
        tokens2 = set(goal2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def get_proof_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about learned proofs.

        Returns:
            Dictionary with learning statistics

        Example:
            >>> stats = learner.get_proof_statistics()
            >>> print(f"Total proofs: {stats['total_proofs']}")
            >>> print(f"Unique patterns: {stats['unique_patterns']}")
        """
        from collections import Counter

        return {
            "total_proofs": len(self.proof_database),
            "unique_patterns": len(self.proof_patterns),
            "rich_patterns": len(self.rich_patterns),
            "goal_types": dict(Counter(p["goal_type"] for p in self.proof_database)),
            "avg_proof_depth": sum(p["depth"] for p in self.proof_database)
            / max(len(self.proof_database), 1),
            "most_common_patterns": sorted(
                self.proof_patterns.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "tactics_per_goal_type": {
                gt: len(tactics) for gt, tactics in self.successful_tactics.items()
            },
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Fuzzy
    "FuzzyLogicReasoner",
    "FuzzySetMetadata",
    # Temporal
    "TemporalReasoner",
    "TimeInterval",
    "RecurringEvent",
    "EventHierarchy",
    # Meta
    "MetaReasoner",
    "ResourceMetrics",
    # Learning
    "ProofLearner",
    "ProofPattern",
]
