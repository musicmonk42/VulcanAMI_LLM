import unittest
from src.safety_systems import SafetySystem

class TestSafetySystem(unittest.TestCase):
    def setUp(self):
        self.safety_system = SafetySystem()

    def test_system_activation(self):
        self.safety_system.activate()
        self.assertTrue(self.safety_system.is_active)

    def test_system_deactivation(self):
        self.safety_system.activate()
        self.safety_system.deactivate()
        self.assertFalse(self.safety_system.is_active)

    def test_emergency_protocol(self):
        self.safety_system.activate()
        self.safety_system.trigger_emergency_protocol()
        self.assertTrue(self.safety_system.emergency_protocol_active)

    def test_governance_compliance(self):
        self.safety_system.activate()
        compliance = self.safety_system.check_governance_compliance()
        self.assertTrue(compliance)

if __name__ == '__main__':
    unittest.main()