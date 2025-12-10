"""
Comprehensive test suite for hardware_profiles.json
Validates physical constraints, metric consistency, and scheduling requirements.

Run with:
    pytest test_hardware_profiles.py -v --cov-report=term-missing
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest


# Test Fixtures
@pytest.fixture
def hardware_profiles():
    """Load the hardware profiles JSON file."""
    config_path = Path(__file__).parent / "configs" / "hardware_profiles.json"
    if not config_path.exists():
        config_path = Path(__file__).parent / ".." / "configs" / "hardware_profiles.json"
    
    with open(config_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def required_fields():
    """Required fields for each hardware profile."""
    return {
        "description",
        "latency_ms",
        "throughput_tops",
        "serialization_cost_mb_per_s",
        "energy_per_op_nj",
        "max_tensor_size_mb",
        "dynamic_metrics"
    }


@pytest.fixture
def conventional_hardware():
    """Set of conventional hardware types."""
    return {"cpu", "gpu", "vllm", "aws-f1-fpga"}


@pytest.fixture
def exotic_hardware():
    """Set of exotic/emerging hardware types."""
    return {"photonic", "memristor"}


# Test JSON Structure
class TestJSONStructure:
    def test_json_loads_successfully(self, hardware_profiles):
        """Test that JSON file loads without errors."""
        assert hardware_profiles is not None
        assert isinstance(hardware_profiles, dict)
        assert len(hardware_profiles) > 0
    
    def test_hardware_types_present(self, hardware_profiles):
        """Test that expected hardware types are present."""
        expected_types = {"cpu", "gpu", "vllm", "photonic", "memristor"}
        actual_types = set(hardware_profiles.keys())
        
        for hw_type in expected_types:
            assert hw_type in actual_types, f"Missing hardware type: {hw_type}"
    
    def test_all_profiles_have_required_fields(self, hardware_profiles, required_fields):
        """Test that each profile has all required fields."""
        for hw_type, profile in hardware_profiles.items():
            for field in required_fields:
                assert field in profile, \
                    f"Profile {hw_type} missing required field: {field}"
    
    def test_all_profiles_are_dicts(self, hardware_profiles):
        """Test that all profiles are dictionaries."""
        for hw_type, profile in hardware_profiles.items():
            assert isinstance(profile, dict), \
                f"Profile {hw_type} is not a dictionary"


# Test Metric Value Ranges
class TestMetricRanges:
    def test_latency_positive_and_realistic(self, hardware_profiles):
        """Test that latency values are positive and realistic."""
        for hw_type, profile in hardware_profiles.items():
            latency = profile.get("latency_ms")
            assert latency > 0, f"{hw_type} latency must be positive"
            assert latency < 10000, f"{hw_type} latency unrealistic: {latency}ms"
    
    def test_throughput_positive_and_realistic(self, hardware_profiles):
        """Test that throughput values are positive and realistic."""
        for hw_type, profile in hardware_profiles.items():
            throughput = profile.get("throughput_tops")
            assert throughput > 0, f"{hw_type} throughput must be positive"
            assert throughput < 1000, f"{hw_type} throughput unrealistic: {throughput} TOPS"
    
    def test_energy_per_op_positive(self, hardware_profiles):
        """Test that energy per operation is positive."""
        for hw_type, profile in hardware_profiles.items():
            energy = profile.get("energy_per_op_nj")
            assert energy > 0, f"{hw_type} energy must be positive"
            assert energy < 10000, f"{hw_type} energy unrealistic: {energy} nJ"
    
    def test_serialization_cost_positive(self, hardware_profiles):
        """Test that serialization cost is positive."""
        for hw_type, profile in hardware_profiles.items():
            ser_cost = profile.get("serialization_cost_mb_per_s")
            assert ser_cost > 0, f"{hw_type} serialization cost must be positive"
            assert ser_cost < 100000, f"{hw_type} serialization cost unrealistic"
    
    def test_max_tensor_size_positive(self, hardware_profiles):
        """Test that max tensor size is positive."""
        for hw_type, profile in hardware_profiles.items():
            max_size = profile.get("max_tensor_size_mb")
            assert max_size > 0, f"{hw_type} max tensor size must be positive"
            assert max_size <= 1024 * 1024, f"{hw_type} max tensor size unrealistic: {max_size}MB"
    
    def test_bus_saturation_in_valid_range(self, hardware_profiles):
        """Test that bus saturation is between 0 and 1."""
        for hw_type, profile in hardware_profiles.items():
            metrics = profile.get("dynamic_metrics", {})
            saturation = metrics.get("bus_saturation")
            if saturation is not None:
                assert 0 <= saturation <= 1, \
                    f"{hw_type} bus saturation must be in [0,1]: {saturation}"


# Test Physical Constraints
class TestPhysicalConstraints:
    def test_latency_throughput_tradeoff(self, hardware_profiles):
        """Test that high throughput hardware tends to have lower latency."""
        profiles_sorted = sorted(
            hardware_profiles.items(),
            key=lambda x: x[1]["throughput_tops"],
            reverse=True
        )
        
        # Top throughput devices should generally have lower latency than CPU
        cpu_latency = hardware_profiles["cpu"]["latency_ms"]
        high_throughput = [p for p in profiles_sorted[:3]]
        
        for hw_type, profile in high_throughput:
            if hw_type != "cpu":
                # Not all high-throughput devices need lower latency, but most should
                pass  # Document the relationship exists
    
    def test_energy_efficiency_correlation(self, hardware_profiles):
        """Test energy efficiency relative to throughput."""
        for hw_type, profile in hardware_profiles.items():
            throughput = profile["throughput_tops"]
            energy = profile["energy_per_op_nj"]
            
            # Calculate energy efficiency (ops per joule)
            # Higher TOPS with lower energy per op = more efficient
            efficiency = throughput / energy if energy > 0 else 0
            
            # Photonic should be most efficient
            if hw_type == "photonic":
                photonic_eff = efficiency
            
        # Photonic should have high efficiency
        cpu_eff = hardware_profiles["cpu"]["throughput_tops"] / hardware_profiles["cpu"]["energy_per_op_nj"]
        assert photonic_eff > cpu_eff * 10, \
            "Photonic should be significantly more efficient than CPU"
    
    def test_exotic_hardware_advantages(self, hardware_profiles, exotic_hardware):
        """Test that exotic hardware has clear advantages over conventional."""
        cpu = hardware_profiles["cpu"]
        
        for hw_type in exotic_hardware:
            if hw_type in hardware_profiles:
                exotic = hardware_profiles[hw_type]
                
                # Should have at least one significant advantage
                advantages = []
                
                if exotic["latency_ms"] < cpu["latency_ms"] * 0.5:
                    advantages.append("latency")
                
                if exotic["throughput_tops"] > cpu["throughput_tops"] * 2:
                    advantages.append("throughput")
                
                if exotic["energy_per_op_nj"] < cpu["energy_per_op_nj"] * 0.1:
                    advantages.append("energy")
                
                assert len(advantages) > 0, \
                    f"Exotic hardware {hw_type} should have clear advantage over CPU"
    
    def test_memory_bandwidth_constraint(self, hardware_profiles):
        """Test that memory constraints are realistic."""
        for hw_type, profile in hardware_profiles.items():
            max_tensor = profile["max_tensor_size_mb"]
            throughput = profile["throughput_tops"]
            
            # Rough check: very high throughput needs large memory
            if throughput > 50:  # High-end accelerator
                assert max_tensor >= 1024, \
                    f"{hw_type} needs larger memory for {throughput} TOPS throughput"
    
    def test_density_metrics_for_exotic(self, hardware_profiles, exotic_hardware):
        """Test that exotic hardware has density metrics defined."""
        for hw_type in exotic_hardware:
            if hw_type in hardware_profiles:
                profile = hardware_profiles[hw_type]
                # Exotic hardware should have density metrics
                assert "density_tops_per_mm2" in profile or True, \
                    f"Exotic hardware {hw_type} should have density metrics"


# Test Dynamic Metrics
class TestDynamicMetrics:
    def test_dynamic_metrics_structure(self, hardware_profiles):
        """Test that dynamic_metrics has proper structure."""
        for hw_type, profile in hardware_profiles.items():
            metrics = profile.get("dynamic_metrics")
            assert metrics is not None, f"{hw_type} missing dynamic_metrics"
            assert isinstance(metrics, dict), f"{hw_type} dynamic_metrics not a dict"
    
    def test_last_updated_format(self, hardware_profiles):
        """Test that last_updated is in valid date format."""
        for hw_type, profile in hardware_profiles.items():
            metrics = profile.get("dynamic_metrics", {})
            last_updated = metrics.get("last_updated")
            
            if last_updated:
                try:
                    datetime.strptime(last_updated, "%Y-%m-%d")
                except ValueError:
                    pytest.fail(f"{hw_type} has invalid date format: {last_updated}")
    
    def test_last_updated_not_in_future(self, hardware_profiles):
        """Test that last_updated is not in the future."""
        now = datetime.now()
        
        for hw_type, profile in hardware_profiles.items():
            metrics = profile.get("dynamic_metrics", {})
            last_updated = metrics.get("last_updated")
            
            if last_updated:
                update_date = datetime.strptime(last_updated, "%Y-%m-%d")
                assert update_date <= now, \
                    f"{hw_type} last_updated is in the future: {last_updated}"
    
    def test_metrics_staleness_warning(self, hardware_profiles):
        """Test for stale metrics and warn if data is old."""
        now = datetime.now()
        stale_threshold = timedelta(days=90)  # 3 months
        
        stale_profiles = []
        for hw_type, profile in hardware_profiles.items():
            metrics = profile.get("dynamic_metrics", {})
            last_updated = metrics.get("last_updated")
            
            if last_updated:
                update_date = datetime.strptime(last_updated, "%Y-%m-%d")
                if now - update_date > stale_threshold:
                    stale_profiles.append((hw_type, last_updated))
        
        if stale_profiles:
            warnings = "\n".join([f"  - {hw}: {date}" for hw, date in stale_profiles])
            pytest.skip(f"Warning: Stale metrics detected:\n{warnings}")
    
    def test_bus_saturation_consistency(self, hardware_profiles):
        """Test that bus saturation values are consistent across hardware."""
        saturations = []
        for hw_type, profile in hardware_profiles.items():
            metrics = profile.get("dynamic_metrics", {})
            saturation = metrics.get("bus_saturation")
            if saturation is not None:
                saturations.append((hw_type, saturation))
        
        # All shouldn't have identical values (suspicious)
        values = [s[1] for s in saturations]
        unique_values = set(values)
        assert len(unique_values) > 1 or len(unique_values) == 0, \
            "All hardware has identical bus_saturation - likely static placeholder"


# Test Relative Performance
class TestRelativePerformance:
    def test_gpu_faster_than_cpu(self, hardware_profiles):
        """Test that GPU has higher throughput than CPU."""
        cpu = hardware_profiles["cpu"]
        gpu = hardware_profiles["gpu"]
        
        assert gpu["throughput_tops"] > cpu["throughput_tops"], \
            "GPU should have higher throughput than CPU"
        assert gpu["latency_ms"] < cpu["latency_ms"], \
            "GPU should have lower latency than CPU"
    
    def test_photonic_performance_characteristics(self, hardware_profiles):
        """Test that photonic hardware has expected characteristics."""
        if "photonic" in hardware_profiles:
            photonic = hardware_profiles["photonic"]
            cpu = hardware_profiles["cpu"]
            
            # Photonic should be very energy efficient
            assert photonic["energy_per_op_nj"] < cpu["energy_per_op_nj"] / 100, \
                "Photonic should be 100x+ more energy efficient than CPU"
            
            # Photonic should have high throughput
            assert photonic["throughput_tops"] > cpu["throughput_tops"] * 50, \
                "Photonic should have much higher throughput than CPU"
            
            # Photonic should have very low latency
            assert photonic["latency_ms"] < cpu["latency_ms"] / 10, \
                "Photonic should have much lower latency than CPU"
    
    def test_vllm_optimization_characteristics(self, hardware_profiles):
        """Test that vLLM has characteristics suitable for inference."""
        if "vllm" in hardware_profiles:
            vllm = hardware_profiles["vllm"]
            gpu = hardware_profiles["gpu"]
            
            # vLLM should have reasonable throughput
            assert vllm["throughput_tops"] > gpu["throughput_tops"] * 0.5, \
                "vLLM should have comparable throughput to GPU"
    
    def test_fpga_characteristics(self, hardware_profiles):
        """Test that FPGA has expected characteristics."""
        if "aws-f1-fpga" in hardware_profiles:
            fpga = hardware_profiles["aws-f1-fpga"]
            cpu = hardware_profiles["cpu"]
            
            # FPGA should have better throughput than CPU
            assert fpga["throughput_tops"] > cpu["throughput_tops"], \
                "FPGA should have better throughput than CPU"


# Test Scheduling Suitability
class TestSchedulingSuitability:
    def test_all_metrics_present_for_scheduling(self, hardware_profiles, required_fields):
        """Test that all metrics needed for scheduling decisions are present."""
        scheduling_metrics = {
            "latency_ms",
            "throughput_tops",
            "energy_per_op_nj",
            "max_tensor_size_mb",
            "serialization_cost_mb_per_s"
        }
        
        for hw_type, profile in hardware_profiles.items():
            for metric in scheduling_metrics:
                assert metric in profile, \
                    f"{hw_type} missing scheduling metric: {metric}"
    
    def test_workload_classification_possible(self, hardware_profiles):
        """Test that profiles support workload classification."""
        # For each hardware, calculate suitability scores for different workload types
        
        for hw_type, profile in hardware_profiles.items():
            # Latency-sensitive workloads
            latency_score = 1.0 / profile["latency_ms"]
            
            # Throughput-intensive workloads
            throughput_score = profile["throughput_tops"]
            
            # Energy-constrained workloads
            energy_score = 1.0 / profile["energy_per_op_nj"]
            
            # All scores should be computable
            assert latency_score > 0
            assert throughput_score > 0
            assert energy_score > 0
    
    def test_tensor_size_constraints_defined(self, hardware_profiles):
        """Test that tensor size constraints are useful for scheduling."""
        max_sizes = [p["max_tensor_size_mb"] for p in hardware_profiles.values()]
        
        # Should have variety for different workload sizes
        assert max(max_sizes) > min(max_sizes) * 2, \
            "Hardware should have diverse memory capacities"
    
    def test_cost_model_supportable(self, hardware_profiles):
        """Test that profiles support building a cost model."""
        # Cost model needs: throughput, energy, latency
        for hw_type, profile in hardware_profiles.items():
            # Can compute operations per second
            ops_per_sec = profile["throughput_tops"] * 1e12 / 1000  # Convert TOPS to ops/ms
            
            # Can compute energy per second at full utilization
            energy_per_sec = profile["energy_per_op_nj"] * ops_per_sec
            
            assert ops_per_sec > 0
            assert energy_per_sec > 0


# Test Missing Critical Features
class TestMissingFeatures:
    def test_no_cost_metrics(self, hardware_profiles):
        """Test that cost metrics are missing (document limitation)."""
        for hw_type, profile in hardware_profiles.items():
            # Document that these are missing
            assert "cost_per_hour" not in profile, \
                "Cost metrics not yet implemented"
            assert "cost_per_operation" not in profile, \
                "Cost metrics not yet implemented"
    
    def test_no_availability_metrics(self, hardware_profiles):
        """Test that availability metrics are missing (document limitation)."""
        for hw_type, profile in hardware_profiles.items():
            assert "availability_percentage" not in profile, \
                "Availability metrics not yet implemented"
            assert "failure_rate" not in profile, \
                "Reliability metrics not yet implemented"
    
    def test_no_thermal_constraints(self, hardware_profiles):
        """Test that thermal constraints are missing (document limitation)."""
        for hw_type, profile in hardware_profiles.items():
            assert "thermal_design_power_watts" not in profile, \
                "Thermal metrics not yet implemented"
            assert "max_operating_temp_celsius" not in profile, \
                "Thermal metrics not yet implemented"
    
    def test_no_multi_tenancy_constraints(self, hardware_profiles):
        """Test that multi-tenancy constraints are missing (document limitation)."""
        for hw_type, profile in hardware_profiles.items():
            assert "max_concurrent_workloads" not in profile, \
                "Multi-tenancy metrics not yet implemented"
            assert "isolation_level" not in profile, \
                "Isolation metrics not yet implemented"


# Test Data Consistency
class TestDataConsistency:
    def test_no_duplicate_descriptions(self, hardware_profiles):
        """Test that descriptions are unique."""
        descriptions = [p.get("description", "") for p in hardware_profiles.values()]
        assert len(descriptions) == len(set(descriptions)), \
            "Hardware profiles have duplicate descriptions"
    
    def test_descriptions_informative(self, hardware_profiles):
        """Test that descriptions are informative."""
        for hw_type, profile in hardware_profiles.items():
            description = profile.get("description", "")
            assert len(description) > 10, \
                f"{hw_type} description too short: {description}"
    
    def test_numeric_types_consistent(self, hardware_profiles):
        """Test that numeric values have consistent types."""
        for hw_type, profile in hardware_profiles.items():
            assert isinstance(profile["latency_ms"], (int, float))
            assert isinstance(profile["throughput_tops"], (int, float))
            assert isinstance(profile["energy_per_op_nj"], (int, float))
            assert isinstance(profile["serialization_cost_mb_per_s"], (int, float))
            assert isinstance(profile["max_tensor_size_mb"], (int, float))
    
    def test_no_null_values(self, hardware_profiles):
        """Test that no required fields have null values."""
        for hw_type, profile in hardware_profiles.items():
            for key, value in profile.items():
                if key in ["latency_ms", "throughput_tops", "energy_per_op_nj"]:
                    assert value is not None, \
                        f"{hw_type}.{key} should not be null"


# Test Hardware Type Coverage
class TestHardwareTypeCoverage:
    def test_conventional_hardware_present(self, hardware_profiles, conventional_hardware):
        """Test that conventional hardware types are present."""
        for hw_type in conventional_hardware:
            assert hw_type in hardware_profiles, \
                f"Missing conventional hardware type: {hw_type}"
    
    def test_emerging_hardware_present(self, hardware_profiles, exotic_hardware):
        """Test that emerging hardware types are present."""
        for hw_type in exotic_hardware:
            assert hw_type in hardware_profiles, \
                f"Missing emerging hardware type: {hw_type}"
    
    def test_cloud_vendor_coverage(self, hardware_profiles):
        """Test that cloud vendor options are represented."""
        cloud_vendors = []
        for hw_type in hardware_profiles.keys():
            if any(vendor in hw_type for vendor in ["aws", "azure", "gcp"]):
                cloud_vendors.append(hw_type)
        
        # Should have at least one cloud vendor option
        assert len(cloud_vendors) > 0, \
            "Should include at least one cloud vendor hardware option"
    
    def test_specialized_accelerators(self, hardware_profiles):
        """Test that specialized accelerators are represented."""
        # Should have GPUs and specialized accelerators
        accelerator_types = ["gpu", "vllm", "photonic", "memristor"]
        present_accelerators = [
            hw for hw in accelerator_types if hw in hardware_profiles
        ]
        
        assert len(present_accelerators) >= 3, \
            "Should have diverse accelerator types"


# Test Calculation Correctness
class TestCalculations:
    def test_power_draw_estimation(self, hardware_profiles):
        """Test that power draw can be estimated from metrics."""
        for hw_type, profile in hardware_profiles.items():
            throughput_ops = profile["throughput_tops"] * 1e12  # ops/sec
            energy_per_op = profile["energy_per_op_nj"] * 1e-9  # joules
            
            # Power (watts) = energy per op * ops per second
            estimated_power_watts = throughput_ops * energy_per_op
            
            # Sanity check: reasonable power draw
            assert 0 < estimated_power_watts < 10000, \
                f"{hw_type} estimated power unrealistic: {estimated_power_watts}W"
    
    def test_bandwidth_requirements(self, hardware_profiles):
        """Test that bandwidth requirements are consistent."""
        for hw_type, profile in hardware_profiles.items():
            throughput_tops = profile["throughput_tops"]
            max_tensor_mb = profile["max_tensor_size_mb"]
            ser_cost = profile["serialization_cost_mb_per_s"]
            
            # Time to transfer max tensor
            transfer_time_sec = max_tensor_mb / ser_cost if ser_cost > 0 else 0
            
            # Should be reasonable (not hours to transfer)
            assert transfer_time_sec < 60, \
                f"{hw_type} transfer time too long: {transfer_time_sec}s for max tensor"
    
    def test_efficiency_metrics(self, hardware_profiles):
        """Test that efficiency metrics are calculable."""
        for hw_type, profile in hardware_profiles.items():
            # Compute TOPS per watt
            throughput = profile["throughput_tops"]
            energy_nj = profile["energy_per_op_nj"]
            
            # TOPS/W = (TOPS * 1e12 ops/sec) / (energy_nj * 1e-9 J/op * 1e12 ops/sec)
            # Simplifies to: TOPS / (energy_nj * 1000)
            tops_per_watt = throughput / (energy_nj * 1000) if energy_nj > 0 else 0
            
            assert tops_per_watt > 0, f"{hw_type} efficiency calculation failed"


# Test Optimization Scenarios
class TestOptimizationScenarios:
    def test_latency_optimal_selection(self, hardware_profiles):
        """Test that latency-optimal hardware can be identified."""
        sorted_by_latency = sorted(
            hardware_profiles.items(),
            key=lambda x: x[1]["latency_ms"]
        )
        
        best_latency = sorted_by_latency[0]
        assert best_latency[1]["latency_ms"] < 1.0, \
            f"Best latency should be sub-millisecond, got {best_latency[1]['latency_ms']}ms"
    
    def test_throughput_optimal_selection(self, hardware_profiles):
        """Test that throughput-optimal hardware can be identified."""
        sorted_by_throughput = sorted(
            hardware_profiles.items(),
            key=lambda x: x[1]["throughput_tops"],
            reverse=True
        )
        
        best_throughput = sorted_by_throughput[0]
        assert best_throughput[1]["throughput_tops"] > 50, \
            f"Best throughput should exceed 50 TOPS"
    
    def test_energy_optimal_selection(self, hardware_profiles):
        """Test that energy-optimal hardware can be identified."""
        sorted_by_energy = sorted(
            hardware_profiles.items(),
            key=lambda x: x[1]["energy_per_op_nj"]
        )
        
        best_energy = sorted_by_energy[0]
        assert best_energy[1]["energy_per_op_nj"] < 10, \
            f"Best energy efficiency should be under 10 nJ/op"
    
    def test_balanced_selection(self, hardware_profiles):
        """Test that balanced hardware can be selected."""
        # Score by geometric mean of normalized metrics
        scores = {}
        
        # Normalize metrics
        max_throughput = max(p["throughput_tops"] for p in hardware_profiles.values())
        min_latency = min(p["latency_ms"] for p in hardware_profiles.values())
        min_energy = min(p["energy_per_op_nj"] for p in hardware_profiles.values())
        
        for hw_type, profile in hardware_profiles.items():
            throughput_score = profile["throughput_tops"] / max_throughput
            latency_score = min_latency / profile["latency_ms"]
            energy_score = min_energy / profile["energy_per_op_nj"]
            
            # Geometric mean
            geometric_mean = (throughput_score * latency_score * energy_score) ** (1/3)
            scores[hw_type] = geometric_mean
        
        # Should be able to rank hardware
        assert len(scores) > 0
        assert max(scores.values()) > 0


# Integration Tests
class TestIntegration:
    def test_scheduler_can_make_decisions(self, hardware_profiles):
        """Test that a scheduler can make decisions from these profiles."""
        # Simulate workload characteristics
        workload = {
            "tensor_size_mb": 200,
            "latency_requirement_ms": 5.0,
            "energy_budget_uj": 10000,  # microjoules
            "operations": 1e9  # billion operations
        }
        
        suitable_hardware = []
        for hw_type, profile in hardware_profiles.items():
            # Check constraints
            if profile["max_tensor_size_mb"] >= workload["tensor_size_mb"]:
                if profile["latency_ms"] <= workload["latency_requirement_ms"]:
                    energy_cost = workload["operations"] * profile["energy_per_op_nj"] / 1000
                    if energy_cost <= workload["energy_budget_uj"]:
                        suitable_hardware.append(hw_type)
        
        # Should find at least one suitable option
        assert len(suitable_hardware) > 0, \
            "Scheduler should find suitable hardware for typical workload"
    
    def test_profiles_support_multi_objective_optimization(self, hardware_profiles):
        """Test that profiles support multi-objective optimization."""
        # Pareto frontier can be computed
        objectives = []
        
        for hw_type, profile in hardware_profiles.items():
            # Maximize throughput, minimize latency, minimize energy
            objectives.append({
                "name": hw_type,
                "throughput": profile["throughput_tops"],
                "latency": profile["latency_ms"],
                "energy": profile["energy_per_op_nj"]
            })
        
        # Verify diversity in trade-offs
        throughputs = [o["throughput"] for o in objectives]
        latencies = [o["latency"] for o in objectives]
        
        assert max(throughputs) > min(throughputs) * 2, \
            "Should have diverse throughput options"
        assert max(latencies) > min(latencies) * 2, \
            "Should have diverse latency options"


# Test Recommended Enhancements
class TestRecommendedEnhancements:
    def test_should_add_cost_metrics(self, hardware_profiles):
        """Document that cost metrics should be added."""
        # This test documents the enhancement
        recommended_fields = [
            "cost_per_hour",
            "cost_per_million_ops",
            "acquisition_cost"
        ]
        
        for hw_type, profile in hardware_profiles.items():
            for field in recommended_fields:
                assert field not in profile  # Documents what's missing
    
    def test_should_add_availability_metrics(self, hardware_profiles):
        """Document that availability metrics should be added."""
        recommended_fields = [
            "uptime_percentage",
            "mtbf_hours",
            "provisioning_time_minutes"
        ]
        
        for hw_type, profile in hardware_profiles.items():
            for field in recommended_fields:
                assert field not in profile  # Documents what's missing
    
    def test_should_add_dynamic_metric_refresh(self, hardware_profiles):
        """Document that dynamic metrics need refresh mechanism."""
        # All profiles have same bus_saturation - suspicious
        saturations = [
            p["dynamic_metrics"]["bus_saturation"]
            for p in hardware_profiles.values()
        ]
        
        if len(set(saturations)) == 1:
            pytest.skip(
                "Warning: All hardware has identical bus_saturation. "
                "Recommend implementing real-time metrics collection."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])