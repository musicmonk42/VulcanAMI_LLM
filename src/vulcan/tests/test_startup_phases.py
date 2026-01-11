"""
Unit tests for startup phases module.

Tests the StartupPhase enum and phase metadata.
"""

import pytest

from vulcan.server.startup.phases import (
    StartupPhase,
    PhaseMetadata,
    PHASE_METADATA,
    get_phase_metadata,
    get_critical_phases,
    is_critical_phase,
)


class TestStartupPhase:
    """Test suite for StartupPhase enum."""
    
    def test_startup_phase_values(self):
        """Test that all expected phases exist."""
        expected_phases = [
            "configuration",
            "core_services",
            "reasoning_systems",
            "memory_systems",
            "preloading",
            "monitoring",
        ]
        
        actual_phases = [phase.value for phase in StartupPhase]
        assert set(actual_phases) == set(expected_phases)
    
    def test_startup_phase_order(self):
        """Test that phases are in logical dependency order."""
        phases = list(StartupPhase)
        
        # Configuration must come first
        assert phases[0] == StartupPhase.CONFIGURATION
        
        # Core services must come before others
        assert phases[1] == StartupPhase.CORE_SERVICES
        
        # Monitoring should come last
        assert phases[-1] == StartupPhase.MONITORING
    
    def test_phase_enum_members(self):
        """Test accessing individual phase members."""
        assert StartupPhase.CONFIGURATION.value == "configuration"
        assert StartupPhase.CORE_SERVICES.value == "core_services"
        assert StartupPhase.REASONING_SYSTEMS.value == "reasoning_systems"
        assert StartupPhase.MEMORY_SYSTEMS.value == "memory_systems"
        assert StartupPhase.PRELOADING.value == "preloading"
        assert StartupPhase.MONITORING.value == "monitoring"


class TestPhaseMetadata:
    """Test suite for PhaseMetadata dataclass."""
    
    def test_phase_metadata_creation(self):
        """Test creating PhaseMetadata instances."""
        metadata = PhaseMetadata(
            name="Test Phase",
            critical=True,
            timeout_seconds=60.0,
            description="Test description"
        )
        
        assert metadata.name == "Test Phase"
        assert metadata.critical is True
        assert metadata.timeout_seconds == 60.0
        assert metadata.description == "Test description"
    
    def test_phase_metadata_attributes(self):
        """Test all attributes are accessible."""
        metadata = PhaseMetadata(
            name="Test",
            critical=False,
            timeout_seconds=30.0,
            description="Desc"
        )
        
        # All attributes should be accessible
        assert hasattr(metadata, 'name')
        assert hasattr(metadata, 'critical')
        assert hasattr(metadata, 'timeout_seconds')
        assert hasattr(metadata, 'description')


class TestPhaseMetadataConfiguration:
    """Test suite for PHASE_METADATA configuration."""
    
    def test_all_phases_have_metadata(self):
        """Test that all phases have metadata configured."""
        for phase in StartupPhase:
            assert phase in PHASE_METADATA, f"Phase {phase} missing metadata"
    
    def test_configuration_phase_is_critical(self):
        """Test that CONFIGURATION phase is marked as critical."""
        metadata = PHASE_METADATA[StartupPhase.CONFIGURATION]
        assert metadata.critical is True
        assert metadata.name == "Configuration Loading"
    
    def test_core_services_phase_is_critical(self):
        """Test that CORE_SERVICES phase is marked as critical."""
        metadata = PHASE_METADATA[StartupPhase.CORE_SERVICES]
        assert metadata.critical is True
        assert metadata.name == "Core Services"
    
    def test_non_critical_phases(self):
        """Test that other phases are not critical."""
        non_critical_phases = [
            StartupPhase.REASONING_SYSTEMS,
            StartupPhase.MEMORY_SYSTEMS,
            StartupPhase.PRELOADING,
            StartupPhase.MONITORING,
        ]
        
        for phase in non_critical_phases:
            metadata = PHASE_METADATA[phase]
            assert metadata.critical is False, f"Phase {phase} should not be critical"
    
    def test_all_phases_have_timeouts(self):
        """Test that all phases have reasonable timeout values."""
        for phase, metadata in PHASE_METADATA.items():
            assert metadata.timeout_seconds > 0
            assert metadata.timeout_seconds <= 300  # Max 5 minutes
    
    def test_all_phases_have_descriptions(self):
        """Test that all phases have non-empty descriptions."""
        for phase, metadata in PHASE_METADATA.items():
            assert metadata.description
            assert len(metadata.description) > 0
    
    def test_phase_timeout_values(self):
        """Test specific timeout values are reasonable."""
        # Configuration should be quick
        assert PHASE_METADATA[StartupPhase.CONFIGURATION].timeout_seconds <= 60
        
        # Preloading can take longer (model loading)
        assert PHASE_METADATA[StartupPhase.PRELOADING].timeout_seconds >= 60


class TestPhaseHelperFunctions:
    """Test suite for phase helper functions."""
    
    def test_get_phase_metadata(self):
        """Test getting metadata for a phase."""
        metadata = get_phase_metadata(StartupPhase.CONFIGURATION)
        
        assert isinstance(metadata, PhaseMetadata)
        assert metadata.name == "Configuration Loading"
        assert metadata.critical is True
    
    def test_get_phase_metadata_all_phases(self):
        """Test getting metadata for all phases."""
        for phase in StartupPhase:
            metadata = get_phase_metadata(phase)
            assert isinstance(metadata, PhaseMetadata)
            assert metadata.name  # Has a name
    
    def test_get_critical_phases(self):
        """Test getting set of critical phases."""
        critical_phases = get_critical_phases()
        
        assert isinstance(critical_phases, set)
        assert StartupPhase.CONFIGURATION in critical_phases
        assert StartupPhase.CORE_SERVICES in critical_phases
        
        # Non-critical phases should not be in the set
        assert StartupPhase.PRELOADING not in critical_phases
        assert StartupPhase.MONITORING not in critical_phases
    
    def test_get_critical_phases_count(self):
        """Test that exactly 2 phases are critical."""
        critical_phases = get_critical_phases()
        assert len(critical_phases) == 2
    
    def test_is_critical_phase_true(self):
        """Test checking if phase is critical (true cases)."""
        assert is_critical_phase(StartupPhase.CONFIGURATION) is True
        assert is_critical_phase(StartupPhase.CORE_SERVICES) is True
    
    def test_is_critical_phase_false(self):
        """Test checking if phase is critical (false cases)."""
        assert is_critical_phase(StartupPhase.REASONING_SYSTEMS) is False
        assert is_critical_phase(StartupPhase.MEMORY_SYSTEMS) is False
        assert is_critical_phase(StartupPhase.PRELOADING) is False
        assert is_critical_phase(StartupPhase.MONITORING) is False
    
    def test_is_critical_phase_all_phases(self):
        """Test is_critical_phase for all phases."""
        for phase in StartupPhase:
            result = is_critical_phase(phase)
            assert isinstance(result, bool)
            
            # Result should match metadata
            metadata = get_phase_metadata(phase)
            assert result == metadata.critical


class TestPhaseIntegration:
    """Integration tests for phase system."""
    
    def test_phase_iteration_order(self):
        """Test that phases can be iterated in order."""
        phases = list(StartupPhase)
        
        # Should be 6 phases total
        assert len(phases) == 6
        
        # First should be configuration
        assert phases[0] == StartupPhase.CONFIGURATION
        
        # Last should be monitoring
        assert phases[-1] == StartupPhase.MONITORING
    
    def test_phase_comparison(self):
        """Test that phases can be compared."""
        # Phases should be comparable via their enum ordering
        config = StartupPhase.CONFIGURATION
        core = StartupPhase.CORE_SERVICES
        
        # Can compare for equality
        assert config == StartupPhase.CONFIGURATION
        assert config != core
    
    def test_phase_string_representation(self):
        """Test string representation of phases."""
        phase = StartupPhase.CONFIGURATION
        
        # Should have string representation
        assert str(phase)
        assert "CONFIGURATION" in str(phase)
    
    def test_metadata_consistency(self):
        """Test that metadata is consistent across access methods."""
        phase = StartupPhase.CONFIGURATION
        
        # Direct access
        direct = PHASE_METADATA[phase]
        
        # Via helper function
        via_helper = get_phase_metadata(phase)
        
        # Should be the same object
        assert direct is via_helper
        assert direct.name == via_helper.name
        assert direct.critical == via_helper.critical
