"""
Tests for Survival Protocol - Network Failure Detection, Graceful Degradation, and Power Management

Tests the enhanced functionality in src/vulcan/planning.py:
- Network failure detection with multi-endpoint testing
- Graceful degradation with automatic fallbacks
- Power management with battery detection and emergency shutdown
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from collections import deque

# Import the classes we're testing
from src.vulcan.planning import (
    SurvivalProtocol,
    PowerManager,
    EnhancedResourceMonitor,
    OperationalMode,
    ConnectivityLevel,
    SystemState
)


class TestNetworkFailureDetection:
    """Test network failure detection functionality."""
    
    def test_detect_network_failure_offline(self):
        """Test detection when network is completely offline."""
        protocol = SurvivalProtocol()
        
        # Mock current state to show offline
        mock_state = Mock()
        mock_state.network_quality = 'offline'
        protocol.resource_monitor.current_state = mock_state
        protocol.resource_monitor.history['network_success'] = deque([0.0, 0.0, 0.0])
        
        result = protocol.detect_network_failure()
        
        assert result['failure_detected'] is True
        assert result['connectivity'] == 'offline'
        assert 'activate_survival_mode' in result['actions']
        assert 'disable_network_capabilities' in result['actions']
    
    def test_detect_network_failure_intermittent(self):
        """Test detection with intermittent connectivity."""
        protocol = SurvivalProtocol()
        
        mock_state = Mock()
        mock_state.network_quality = 'intermittent'
        protocol.resource_monitor.current_state = mock_state
        protocol.resource_monitor.history['network_success'] = deque([0.3, 0.4, 0.3])
        
        result = protocol.detect_network_failure()
        
        assert result['failure_detected'] is True
        assert result['connectivity'] == 'intermittent'
        assert 'enable_retry_logic' in result['actions']
    
    def test_detect_network_failure_good(self):
        """Test when network is functioning normally."""
        protocol = SurvivalProtocol()
        
        mock_state = Mock()
        mock_state.network_quality = 'good'
        protocol.resource_monitor.current_state = mock_state
        protocol.resource_monitor.history['network_success'] = deque([1.0, 1.0, 1.0])
        
        result = protocol.detect_network_failure()
        
        assert result['failure_detected'] is False
        assert result['connectivity'] == 'good'
        assert len(result['actions']) == 0
    
    @patch('socket.create_connection')
    def test_assess_network_quality_all_endpoints(self, mock_socket):
        """Test network quality assessment with all endpoints responding."""
        monitor = EnhancedResourceMonitor()
        
        # Create mock socket that supports context manager protocol
        mock_conn = MagicMock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        # Mock successful connections - return same mock for all calls
        mock_socket.return_value = mock_conn
        
        quality = monitor._assess_network_quality()
        
        # Should test 3 endpoints
        assert mock_socket.call_count >= 3
        # Quality should be good or excellent
        assert quality in ['excellent', 'good']
    
    @patch('socket.create_connection')
    def test_assess_network_quality_partial_failure(self, mock_socket):
        """Test network quality with partial endpoint failures."""
        monitor = EnhancedResourceMonitor()
        
        # Create successful mock connections
        mock_conn1 = MagicMock()
        mock_conn1.__enter__ = Mock(return_value=mock_conn1)
        mock_conn1.__exit__ = Mock(return_value=False)
        
        mock_conn2 = MagicMock()
        mock_conn2.__enter__ = Mock(return_value=mock_conn2)
        mock_conn2.__exit__ = Mock(return_value=False)
        
        # Mock: first call succeeds, second fails (OSError more realistic), third succeeds
        mock_socket.side_effect = [
            mock_conn1,
            OSError("Connection timeout"),
            mock_conn2
        ]
        
        quality = monitor._assess_network_quality()
        
        # Should detect degraded, good, or intermittent connectivity (not offline)
        assert quality in ['degraded', 'good', 'intermittent', 'excellent']
    
    @patch('socket.create_connection')
    def test_assess_network_quality_all_offline(self, mock_socket):
        """Test network quality when all endpoints fail."""
        monitor = EnhancedResourceMonitor()
        
        # Mock: socket.create_connection raises OSError on every call
        # This simulates network being completely unreachable
        mock_socket.side_effect = OSError("Network unreachable")
        
        quality = monitor._assess_network_quality()
        
        # When all endpoints fail, should report offline or very poor connectivity
        assert quality in ['offline', 'degraded', 'intermittent']


class TestGracefulDegradation:
    """Test graceful degradation functionality."""
    
    def test_apply_graceful_degradation_no_failure(self):
        """Test that no degradation occurs without failure."""
        protocol = SurvivalProtocol()
        initial_mode = protocol.current_mode
        
        failure_info = {
            'failure_detected': False,
            'connectivity': 'good',
            'actions': []
        }
        
        protocol.apply_graceful_degradation(failure_info)
        
        # Mode should not change
        assert protocol.current_mode == initial_mode
    
    def test_apply_graceful_degradation_offline(self):
        """Test degradation when network goes offline."""
        protocol = SurvivalProtocol()
        
        failure_info = {
            'failure_detected': True,
            'connectivity': 'offline',
            'actions': ['switch_to_local_mode', 'disable_network_capabilities', 'activate_survival_mode']
        }
        
        protocol.apply_graceful_degradation(failure_info)
        
        # Should switch to survival mode
        assert protocol.current_mode == OperationalMode.SURVIVAL
    
    def test_switch_to_local_mode(self):
        """Test switching to local-only operation."""
        protocol = SurvivalProtocol()
        
        # Set up a network-dependent capability
        protocol.capabilities['test_network_cap'] = {
            'enabled': True,
            'network_required': True,
            'fallback': 'test_local_cap'
        }
        protocol.capabilities['test_local_cap'] = {
            'enabled': False
        }
        
        protocol._switch_to_local_mode()
        
        # Network capability should be disabled
        assert protocol.capabilities['test_network_cap']['enabled'] is False
        # Fallback should be enabled
        assert protocol.capabilities['test_local_cap']['enabled'] is True
    
    def test_disable_network_capabilities(self):
        """Test disabling all network-dependent capabilities."""
        protocol = SurvivalProtocol()
        
        # Add some network capabilities
        protocol.capabilities['net_cap_1'] = {
            'enabled': True,
            'network_required': True
        }
        protocol.capabilities['net_cap_2'] = {
            'enabled': True,
            'network_required': True
        }
        protocol.capabilities['local_cap'] = {
            'enabled': True,
            'network_required': False
        }
        
        protocol._disable_network_capabilities()
        
        # Network capabilities should be disabled
        assert protocol.capabilities['net_cap_1']['enabled'] is False
        assert protocol.capabilities['net_cap_2']['enabled'] is False
        # Local capability should still be enabled
        assert protocol.capabilities['local_cap']['enabled'] is True
    
    def test_enable_network_retry(self):
        """Test enabling network retry logic."""
        protocol = SurvivalProtocol()
        
        protocol._enable_network_retry()
        
        assert hasattr(protocol, 'network_retry_enabled')
        assert protocol.network_retry_enabled is True
    
    def test_reduce_network_load(self):
        """Test reducing network load."""
        protocol = SurvivalProtocol()
        
        protocol._reduce_network_load()
        
        assert hasattr(protocol, 'network_batch_size')
        assert protocol.network_batch_size == 10
        assert hasattr(protocol, 'network_priority_threshold')
        assert protocol.network_priority_threshold == 0.8


class TestPowerManagement:
    """Test power management functionality."""
    
    @patch('psutil.sensors_battery')
    def test_detect_battery_available(self, mock_battery):
        """Test battery detection when battery is present."""
        mock_battery.return_value = Mock(percent=80, power_plugged=True)
        
        power_mgr = PowerManager()
        
        assert power_mgr.battery_available is True
    
    @patch('psutil.sensors_battery')
    def test_detect_battery_not_available(self, mock_battery):
        """Test battery detection when no battery present."""
        mock_battery.return_value = None
        
        power_mgr = PowerManager()
        
        assert power_mgr.battery_available is False
    
    @patch('psutil.sensors_battery')
    def test_check_power_status_on_ac(self, mock_battery):
        """Test power status check when on AC power."""
        battery_info = Mock(percent=85, power_plugged=True, secsleft=-1)
        mock_battery.return_value = battery_info
        
        power_mgr = PowerManager()
        power_mgr.battery_available = True
        
        status = power_mgr.check_power_status()
        
        assert status['on_battery'] is False
        assert status['battery_percent'] == 85
        assert status['power_warning'] is False
    
    @patch('psutil.sensors_battery')
    def test_check_power_status_critical_battery(self, mock_battery):
        """Test power status at critical battery level."""
        battery_info = Mock(percent=4, power_plugged=False, secsleft=300)
        mock_battery.return_value = battery_info
        
        power_mgr = PowerManager()
        power_mgr.battery_available = True
        
        status = power_mgr.check_power_status()
        
        assert status['on_battery'] is True
        assert status['battery_percent'] == 4
        assert status['power_warning'] is True
        assert 'emergency_shutdown' in status['actions']
    
    @patch('psutil.sensors_battery')
    def test_check_power_status_low_battery(self, mock_battery):
        """Test power status at low battery level."""
        battery_info = Mock(percent=12, power_plugged=False, secsleft=900)
        mock_battery.return_value = battery_info
        
        power_mgr = PowerManager()
        power_mgr.battery_available = True
        
        status = power_mgr.check_power_status()
        
        assert status['on_battery'] is True
        assert status['battery_percent'] == 12
        assert status['power_warning'] is True
        assert 'activate_survival_mode' in status['actions']
        assert 'save_state' in status['actions']
    
    @patch('psutil.sensors_battery')
    def test_check_power_status_moderate_battery(self, mock_battery):
        """Test power status at moderate battery level on battery power."""
        battery_info = Mock(percent=25, power_plugged=False, secsleft=1800)
        mock_battery.return_value = battery_info
        
        power_mgr = PowerManager()
        power_mgr.battery_available = True
        
        status = power_mgr.check_power_status()
        
        assert status['on_battery'] is True
        assert status['battery_percent'] == 25
        assert status['power_warning'] is True
        assert 'activate_power_saver' in status['actions']
    
    def test_apply_power_management_emergency(self):
        """Test applying emergency power management."""
        power_mgr = PowerManager()
        
        power_status = {
            'on_battery': True,
            'battery_percent': 3,
            'power_warning': True,
            'actions': ['emergency_shutdown']
        }
        
        power_mgr.apply_power_management(power_status)
        
        # Should set survival profile
        assert power_mgr.current_profile == 'survival'
    
    def test_apply_power_management_survival(self):
        """Test applying survival mode power management."""
        power_mgr = PowerManager()
        
        power_status = {
            'on_battery': True,
            'battery_percent': 12,
            'power_warning': True,
            'actions': ['activate_survival_mode', 'save_state']
        }
        
        power_mgr.apply_power_management(power_status)
        
        assert power_mgr.current_profile == 'survival'
    
    def test_apply_power_management_power_saver(self):
        """Test applying power saver mode."""
        power_mgr = PowerManager()
        
        power_status = {
            'on_battery': True,
            'battery_percent': 28,
            'power_warning': True,
            'actions': ['activate_power_saver']
        }
        
        power_mgr.apply_power_management(power_status)
        
        assert power_mgr.current_profile == 'power_saver'
    
    def test_apply_power_management_no_warning(self):
        """Test that no power management is applied without warning."""
        power_mgr = PowerManager()
        initial_profile = power_mgr.current_profile
        
        power_status = {
            'on_battery': False,
            'battery_percent': 85,
            'power_warning': False,
            'actions': []
        }
        
        power_mgr.apply_power_management(power_status)
        
        # Profile should not change
        assert power_mgr.current_profile == initial_profile
    
    def test_emergency_shutdown(self):
        """Test emergency shutdown procedure."""
        power_mgr = PowerManager()
        power_mgr.battery_percent = 2
        
        power_mgr._emergency_shutdown()
        
        # Should set most restrictive profile
        assert power_mgr.current_profile == 'survival'
    
    def test_thermal_throttling(self):
        """Test thermal throttling activation."""
        power_mgr = PowerManager()
        
        # Test high temperature
        is_throttled = power_mgr.check_thermal_status(90.0)
        assert is_throttled is True
        assert power_mgr.thermal_throttle_active is True
        
        # Test temperature return to normal
        is_throttled = power_mgr.check_thermal_status(60.0)
        assert is_throttled is False
        assert power_mgr.thermal_throttle_active is False


class TestIntegration:
    """Integration tests for survival protocol components."""
    
    def test_full_degradation_cycle(self):
        """Test complete degradation cycle from detection to action."""
        protocol = SurvivalProtocol()
        
        # Simulate network failure
        mock_state = Mock()
        mock_state.network_quality = 'offline'
        protocol.resource_monitor.current_state = mock_state
        protocol.resource_monitor.history['network_success'] = deque([0.0, 0.0, 0.0])
        
        # Detect failure
        failure_info = protocol.detect_network_failure()
        assert failure_info['failure_detected'] is True
        
        # Apply degradation
        protocol.apply_graceful_degradation(failure_info)
        
        # Verify system adapted
        assert protocol.current_mode == OperationalMode.SURVIVAL
    
    @patch('psutil.sensors_battery')
    def test_combined_power_and_network_failure(self, mock_battery):
        """Test handling simultaneous power and network failures."""
        # Setup power failure
        battery_info = Mock(percent=10, power_plugged=False, secsleft=600)
        mock_battery.return_value = battery_info
        
        power_mgr = PowerManager()
        power_mgr.battery_available = True
        
        # Setup network failure
        protocol = SurvivalProtocol()
        mock_state = Mock()
        mock_state.network_quality = 'offline'
        protocol.resource_monitor.current_state = mock_state
        
        # Check power
        power_status = power_mgr.check_power_status()
        assert power_status['power_warning'] is True
        
        # Check network
        network_failure = protocol.detect_network_failure()
        assert network_failure['failure_detected'] is True
        
        # Both should trigger survival mode
        power_mgr.apply_power_management(power_status)
        protocol.apply_graceful_degradation(network_failure)
        
        assert power_mgr.current_profile == 'survival'
        assert protocol.current_mode == OperationalMode.SURVIVAL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
