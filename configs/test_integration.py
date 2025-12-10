#!/usr/bin/env python3
"""
Integration test suite for VulcanAMI_LLM configs with the platform.

This module tests that all configuration files can be loaded and integrated
correctly with the VulcanAMI platform components including:
- src/vulcan/config.py (ConfigurationManager)
- src/vulcan/world_model/ (meta-reasoning, self-improvement)
- Tool selection system
- Hardware profiles
- Type system

Tests verify:
1. Config files are loadable and parseable
2. Structures match expected schemas
3. Integration points work correctly with platform code
4. ConfigurationManager can load and use configs
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_manager_integration():
    """Test that ConfigurationManager loads configs correctly"""
    print("Testing ConfigurationManager integration...")
    print("-" * 80)

    try:
        from vulcan.config import (ConfigurationManager, ProfileType,
                                   get_config, get_intrinsic_drives_config,
                                   get_tool_selection_config,
                                   load_intrinsic_drives_from_file,
                                   load_profile)

        # Test basic config manager
        print("  ✓ Config module imports successfully")

        # Test profile loading
        config_mgr = ConfigurationManager()
        success = config_mgr.load_profile(ProfileType.DEVELOPMENT)
        if success:
            print(f"  ✓ Loaded DEVELOPMENT profile")
        else:
            print(f"  ✗ Failed to load DEVELOPMENT profile")
            return False

        # Test getting config values
        agent_id = get_config('agent_config.agent_id', 'default')
        print(f"  ✓ Retrieved agent_id: {agent_id}")

        # Test intrinsic drives config
        intrinsic_config = get_intrinsic_drives_config()
        if intrinsic_config:
            print(f"  ✓ Retrieved intrinsic drives config: enabled={intrinsic_config.get('enabled', False)}")

        # Test tool selection config
        tool_config = get_tool_selection_config()
        if tool_config:
            print(f"  ✓ Retrieved tool selection config: mode={tool_config.get('default_selection_mode', 'balanced')}")

        # Test loading intrinsic drives from file
        drives = load_intrinsic_drives_from_file()
        if drives and 'drives' in drives:
            print(f"  ✓ Loaded intrinsic drives from file: {len(drives.get('drives', {}))} drive types")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_config_files_loadable():
    """Test that all config files are loadable as JSON/YAML"""
    print("Testing config file loading...")
    print("-" * 80)

    configs_dir = Path(__file__).parent

    # Test JSON files
    json_files = [
        'graphix_core_manifest.json',
        'graphix_core_ontology.json',
        'hardware_profiles.json',
        'intrinsic_drives.json',
        'platform_mapping.json',
        'profile_development.json',
        'profile_testing.json',
        'type_system_manifest.json',
    ]

    all_pass = True
    for json_file in json_files:
        try:
            with open(configs_dir / json_file) as f:
                data = json.load(f)
            print(f"  ✓ {json_file}: Loaded successfully")
        except Exception as e:
            print(f"  ✗ {json_file}: Failed - {e}")
            all_pass = False

    print()
    return all_pass

def test_hardware_profiles_structure():
    """Test hardware profiles have correct structure"""
    print("Testing hardware profiles structure...")
    print("-" * 80)

    try:
        with open(Path(__file__).parent / 'hardware_profiles.json') as f:
            profiles = json.load(f)

        required_fields = ['latency_ms', 'throughput_tops', 'energy_per_op_nj']
        all_pass = True

        for profile_name, profile_data in profiles.items():
            missing = [f for f in required_fields if f not in profile_data]
            if missing:
                print(f"  ✗ {profile_name}: Missing fields {missing}")
                all_pass = False
            else:
                print(f"  ✓ {profile_name}: Complete")

        print()
        return all_pass

    except Exception as e:
        print(f"  ✗ Error: {e}")
        print()
        return False

def test_intrinsic_drives_structure():
    """Test intrinsic drives config has correct structure"""
    print("Testing intrinsic drives structure...")
    print("-" * 80)

    try:
        with open(Path(__file__).parent / 'intrinsic_drives.json') as f:
            config = json.load(f)

        # Check top-level keys
        required_keys = ['drives', 'global_settings']
        missing = [k for k in required_keys if k not in config]
        if missing:
            print(f"  ✗ Missing top-level keys: {missing}")
            return False

        print(f"  ✓ Has required top-level keys")

        # Check drives structure
        drives = config.get('drives', {})
        drive_types = ['self_improvement', 'exploration', 'optimization', 'maintenance']
        for drive_type in drive_types:
            if drive_type in drives:
                drive = drives[drive_type]
                if 'enabled' in drive and 'priority' in drive:
                    print(f"  ✓ {drive_type}: enabled={drive['enabled']}, priority={drive['priority']}")
                else:
                    print(f"  ⚠ {drive_type}: Missing enabled/priority fields")

        # Check global settings
        global_settings = config.get('global_settings', {})
        if 'balance_drives' in global_settings:
            print(f"  ✓ Global settings: balance_drives={global_settings['balance_drives']}")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_type_system_structure():
    """Test type system manifest has correct structure"""
    print("Testing type system manifest structure...")
    print("-" * 80)

    try:
        with open(Path(__file__).parent / 'type_system_manifest.json') as f:
            config = json.load(f)

        if 'types' not in config:
            print(f"  ✗ Missing 'types' key")
            return False

        types = config['types']
        print(f"  ✓ Has {len(types)} type definitions")

        # Check some critical types
        critical_types = ['int', 'float', 'str', 'bool', 'dict', 'list']
        for type_name in critical_types:
            if type_name in types:
                print(f"  ✓ {type_name}: defined")
            else:
                print(f"  ✗ {type_name}: missing")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        print()
        return False

def test_graphix_configs():
    """Test Graphix configuration files"""
    print("Testing Graphix configurations...")
    print("-" * 80)

    try:
        # Test manifest
        with open(Path(__file__).parent / 'graphix_core_manifest.json') as f:
            manifest = json.load(f)

        required_keys = ['version', 'core_components', 'hardware_backends']
        missing = [k for k in required_keys if k not in manifest]
        if missing:
            print(f"  ✗ Manifest missing keys: {missing}")
            return False

        print(f"  ✓ Manifest: version={manifest['version']}, {len(manifest['core_components'])} components")

        # Test ontology
        with open(Path(__file__).parent / 'graphix_core_ontology.json') as f:
            ontology = json.load(f)

        if 'ontology' not in ontology:
            print(f"  ✗ Ontology missing 'ontology' key")
            return False

        ont = ontology['ontology']
        if 'classes' in ont and 'properties' in ont and 'relationships' in ont:
            print(f"  ✓ Ontology: {len(ont['classes'])} classes, {len(ont['relationships'])} relationships")
        else:
            print(f"  ✗ Ontology missing core structure")
            return False

        print()
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        print()
        return False

def test_profile_configs():
    """Test profile configurations"""
    print("Testing profile configurations...")
    print("-" * 80)

    profiles = ['development', 'testing']
    all_pass = True

    for profile_name in profiles:
        try:
            with open(Path(__file__).parent / f'profile_{profile_name}.json') as f:
                profile = json.load(f)

            # Check for required sections
            required_sections = ['agent_config', 'resource_limits']
            missing = [s for s in required_sections if s not in profile]
            if missing:
                print(f"  ✗ profile_{profile_name}.json: Missing sections {missing}")
                all_pass = False
            else:
                # Check for enable_self_improvement at root level
                if 'enable_self_improvement' in profile:
                    status = "enabled" if profile['enable_self_improvement'] else "disabled"
                    print(f"  ✓ profile_{profile_name}.json: Complete, self-improvement {status}")
                else:
                    print(f"  ⚠ profile_{profile_name}.json: Missing root-level enable_self_improvement")
                    all_pass = False

        except Exception as e:
            print(f"  ✗ profile_{profile_name}.json: Error - {e}")
            all_pass = False

    print()
    return all_pass

def test_platform_integration_points():
    """Test that configs integrate with platform components"""
    print("Testing platform integration points...")
    print("-" * 80)

    try:
        # Check that intrinsic drives config path matches what's used in code
        expected_path = "configs/intrinsic_drives.json"
        if Path(expected_path).exists():
            print(f"  ✓ Intrinsic drives config at expected path: {expected_path}")
        else:
            print(f"  ✗ Missing intrinsic drives config at: {expected_path}")
            return False

        # Check that profile configs exist
        for profile in ['development', 'testing']:
            path = f"configs/profile_{profile}.json"
            if Path(path).exists():
                print(f"  ✓ Profile config exists: {path}")
            else:
                print(f"  ✗ Missing profile config: {path}")
                return False

        # Check that tool_selection.yaml has expected structure
        import yaml
        with open('configs/tool_selection.yaml') as f:
            tool_config = yaml.safe_load(f)

        if 'defaults' in tool_config:
            print(f"  ✓ Tool selection config has 'defaults' section")
        else:
            print(f"  ⚠ Tool selection config missing 'defaults' section")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def main():
    """Run all integration tests"""
    print("=" * 80)
    print("VulcanAMI_LLM Config Integration Tests")
    print("=" * 80)
    print()

    tests = [
        ("Config Files Loadable", test_config_files_loadable),
        ("Hardware Profiles Structure", test_hardware_profiles_structure),
        ("Intrinsic Drives Structure", test_intrinsic_drives_structure),
        ("Type System Structure", test_type_system_structure),
        ("Graphix Configurations", test_graphix_configs),
        ("Profile Configurations", test_profile_configs),
        ("Platform Integration Points", test_platform_integration_points),
        ("ConfigurationManager Integration", test_config_manager_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: Exception - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("=" * 80)
    print("Integration Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All integration tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
