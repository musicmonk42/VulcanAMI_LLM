#!/usr/bin/env python3
"""
Comprehensive validation script for VulcanAMI_LLM/configs directory

This script validates:
- JSON/YAML syntax
- Schema compliance for key configuration files
- Cross-file consistency
- Security best practices
- Documentation completeness

Usage:
    python configs/validate_configs.py [--strict] [--fix]
    
Options:
    --strict    Exit with error on warnings
    --fix       Auto-fix fixable issues (not implemented yet)
"""

import json
import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Results from a validation check"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)


class ConfigValidator:
    """Validates configuration files in the configs directory"""
    
    def __init__(self, configs_dir: Path, strict: bool = False):
        self.configs_dir = configs_dir
        self.strict = strict
        self.results = ValidationResult(passed=True)
        
    def validate(self) -> ValidationResult:
        """Run all validation checks"""
        print("=" * 80)
        print("VulcanAMI_LLM Configuration Validation")
        print("=" * 80)
        print()
        
        checks = [
            ("Syntax Validation", self._validate_syntax),
            ("Required Files", self._validate_required_files),
            ("Schema Validation", self._validate_schemas),
            ("Consistency Checks", self._validate_consistency),
            ("Security Checks", self._validate_security),
            ("Documentation", self._validate_documentation),
        ]
        
        for check_name, check_func in checks:
            print(f"{check_name}...")
            print("-" * 80)
            check_func()
            print()
        
        self._print_summary()
        return self.results
    
    def _validate_syntax(self):
        """Validate JSON and YAML syntax"""
        # JSON files
        for json_file in self.configs_dir.rglob("*.json"):
            rel_path = json_file.relative_to(self.configs_dir)
            try:
                with open(json_file) as f:
                    json.load(f)
                print(f"  ✓ {rel_path}: Valid JSON")
            except json.JSONDecodeError as e:
                self.results.errors.append(f"{rel_path}: Invalid JSON - {e}")
                self.results.passed = False
                print(f"  ✗ {rel_path}: Invalid JSON")
            except Exception as e:
                self.results.errors.append(f"{rel_path}: Error - {e}")
                self.results.passed = False
                print(f"  ✗ {rel_path}: Error")
        
        # YAML files
        for yaml_file in list(self.configs_dir.rglob("*.yaml")) + list(self.configs_dir.rglob("*.yml")):
            rel_path = yaml_file.relative_to(self.configs_dir)
            try:
                with open(yaml_file) as f:
                    content = f.read()
                    # Support multi-document YAML
                    list(yaml.safe_load_all(content))
                print(f"  ✓ {rel_path}: Valid YAML")
            except yaml.YAMLError as e:
                self.results.errors.append(f"{rel_path}: Invalid YAML - {e}")
                self.results.passed = False
                print(f"  ✗ {rel_path}: Invalid YAML")
            except Exception as e:
                self.results.errors.append(f"{rel_path}: Error - {e}")
                self.results.passed = False
                print(f"  ✗ {rel_path}: Error")
    
    def _validate_required_files(self):
        """Check for required configuration files"""
        required_files = {
            "__init__.py": "Package initialization",
            "graphix_core_manifest.json": "Graphix core component manifest",
            "graphix_core_ontology.json": "Graphix IR ontology definitions",
            "hardware_profiles.json": "Hardware backend profiles",
            "intrinsic_drives.json": "Agent intrinsic motivation config",
            "platform_mapping.json": "API endpoint mappings",
            "profile_development.json": "Development profile",
            "type_system_manifest.json": "Type system definitions",
        }
        
        for filename, description in required_files.items():
            file_path = self.configs_dir / filename
            if file_path.exists():
                print(f"  ✓ {filename}: Present ({description})")
            else:
                self.results.errors.append(f"{filename}: Missing required file")
                self.results.passed = False
                print(f"  ✗ {filename}: Missing")
    
    def _validate_schemas(self):
        """Validate configuration schemas"""
        # Validate profile_development.json
        self._validate_profile_schema("profile_development.json")
        self._validate_profile_schema("profile_testing.json", required=False)
        
        # Validate hardware_profiles.json
        self._validate_hardware_profiles()
        
        # Validate graphix configs
        self._validate_graphix_manifest()
        self._validate_graphix_ontology()
        
        # Validate type_system_manifest
        self._validate_type_system()
    
    def _validate_profile_schema(self, filename: str, required: bool = True):
        """Validate agent profile configuration"""
        file_path = self.configs_dir / filename
        if not file_path.exists():
            if required:
                self.results.errors.append(f"{filename}: Missing")
                self.results.passed = False
            return
        
        try:
            with open(file_path) as f:
                profile = json.load(f)
            
            # Check required top-level keys
            required_keys = ["agent_config", "resource_limits", "safety_policies"]
            missing = [k for k in required_keys if k not in profile]
            if missing:
                self.results.warnings.append(f"{filename}: Missing keys: {missing}")
                print(f"  ⚠ {filename}: Missing keys: {missing}")
            else:
                print(f"  ✓ {filename}: Has required top-level keys")
            
            # Check for enable_self_improvement at root level (required by world_model_core.py)
            if "enable_self_improvement" in profile:
                print(f"  ✓ {filename}: Has root-level enable_self_improvement")
            else:
                self.results.warnings.append(f"{filename}: Missing root-level enable_self_improvement")
                print(f"  ⚠ {filename}: Missing root-level enable_self_improvement")
            
            # Check resource_limits structure
            if "resource_limits" in profile:
                limits = profile["resource_limits"]
                expected = ["max_memory_mb", "max_cpu_percent"]
                missing = [k for k in expected if k not in limits]
                if not missing:
                    print(f"  ✓ {filename}: resource_limits has expected fields")
                else:
                    self.results.info.append(f"{filename}: resource_limits missing: {missing}")
            
            # Check versioning info
            if "versioning" in profile:
                print(f"  ✓ {filename}: Has versioning info")
            else:
                self.results.info.append(f"{filename}: No versioning info")
                print(f"  ℹ {filename}: Consider adding versioning info")
                
        except Exception as e:
            self.results.errors.append(f"{filename}: Validation failed - {e}")
            self.results.passed = False
            print(f"  ✗ {filename}: Validation failed")
    
    def _validate_hardware_profiles(self):
        """Validate hardware profiles configuration"""
        file_path = self.configs_dir / "hardware_profiles.json"
        try:
            with open(file_path) as f:
                profiles = json.load(f)
            
            if not profiles:
                self.results.errors.append("hardware_profiles.json: Empty file")
                self.results.passed = False
                print(f"  ✗ hardware_profiles.json: Empty")
                return
            
            print(f"  ✓ hardware_profiles.json: Has {len(profiles)} profiles")
            
            # Validate each profile
            required_fields = ["latency_ms", "throughput_tops", "energy_per_op_nj"]
            for profile_name, profile_data in profiles.items():
                missing = [f for f in required_fields if f not in profile_data]
                if missing:
                    self.results.warnings.append(
                        f"hardware_profiles.json:{profile_name}: Missing {missing}"
                    )
                    print(f"  ⚠ {profile_name}: Missing {missing}")
                else:
                    print(f"  ✓ {profile_name}: Complete")
                    
        except Exception as e:
            self.results.errors.append(f"hardware_profiles.json: Validation failed - {e}")
            self.results.passed = False
            print(f"  ✗ hardware_profiles.json: Validation failed")
    
    def _validate_graphix_manifest(self):
        """Validate Graphix core manifest"""
        file_path = self.configs_dir / "graphix_core_manifest.json"
        try:
            with open(file_path) as f:
                manifest = json.load(f)
            
            required_keys = ["version", "core_components", "hardware_backends"]
            missing = [k for k in required_keys if k not in manifest]
            if missing:
                self.results.warnings.append(f"graphix_core_manifest.json: Missing {missing}")
                print(f"  ⚠ graphix_core_manifest.json: Missing {missing}")
            else:
                print(f"  ✓ graphix_core_manifest.json: Has required keys")
                
        except Exception as e:
            self.results.errors.append(f"graphix_core_manifest.json: Validation failed - {e}")
            self.results.passed = False
    
    def _validate_graphix_ontology(self):
        """Validate Graphix ontology"""
        file_path = self.configs_dir / "graphix_core_ontology.json"
        try:
            with open(file_path) as f:
                ontology = json.load(f)
            
            if "ontology" in ontology:
                ont = ontology["ontology"]
                if "classes" in ont and "properties" in ont and "relationships" in ont:
                    print(f"  ✓ graphix_core_ontology.json: Has complete ontology structure")
                else:
                    self.results.warnings.append("graphix_core_ontology.json: Incomplete ontology")
                    print(f"  ⚠ graphix_core_ontology.json: Incomplete ontology")
            else:
                self.results.warnings.append("graphix_core_ontology.json: Missing ontology key")
                print(f"  ⚠ graphix_core_ontology.json: Missing ontology key")
                
        except Exception as e:
            self.results.errors.append(f"graphix_core_ontology.json: Validation failed - {e}")
            self.results.passed = False
    
    def _validate_type_system(self):
        """Validate type system manifest"""
        file_path = self.configs_dir / "type_system_manifest.json"
        try:
            with open(file_path) as f:
                type_system = json.load(f)
            
            if "types" in type_system:
                print(f"  ✓ type_system_manifest.json: Has {len(type_system['types'])} types")
            else:
                self.results.warnings.append("type_system_manifest.json: Missing types key")
                print(f"  ⚠ type_system_manifest.json: Missing types key")
                
        except Exception as e:
            self.results.errors.append(f"type_system_manifest.json: Validation failed - {e}")
            self.results.passed = False
    
    def _validate_consistency(self):
        """Check for consistency across configuration files"""
        # Check platform_mapping.json and .yaml consistency
        try:
            with open(self.configs_dir / "platform_mapping.json") as f:
                json_mapping = json.load(f)
            with open(self.configs_dir / "platform_mapping.yaml") as f:
                yaml_mapping = yaml.safe_load(f)
            
            if json_mapping == yaml_mapping:
                print(f"  ✓ platform_mapping.json and .yaml are consistent")
            else:
                self.results.errors.append("platform_mapping.json and .yaml differ")
                self.results.passed = False
                print(f"  ✗ platform_mapping.json and .yaml differ")
        except Exception as e:
            self.results.warnings.append(f"Could not compare platform_mapping files: {e}")
            print(f"  ⚠ Could not compare platform_mapping files")
    
    def _validate_security(self):
        """Check for security issues in configurations"""
        # Check for hardcoded secrets
        sensitive_patterns = ["password", "secret", "api_key", "token", "credential"]
        files_with_keywords = []
        
        for config_file in self.configs_dir.rglob("*"):
            if config_file.is_file() and config_file.suffix in [".json", ".yaml", ".yml"]:
                try:
                    with open(config_file) as f:
                        content = f.read().lower()
                    
                    # Skip if it's a schema or example
                    if "schema" in str(config_file).lower() or "example" in content:
                        continue
                    
                    found = [p for p in sensitive_patterns if p in content]
                    if found:
                        files_with_keywords.append(config_file.relative_to(self.configs_dir))
                except:
                    pass
        
        if files_with_keywords:
            print(f"  ℹ {len(files_with_keywords)} files contain sensitive keywords")
            print(f"  ℹ Ensure these use environment variables or secret management")
        else:
            print(f"  ✓ No sensitive keywords found in configs")
    
    def _validate_documentation(self):
        """Check for documentation completeness"""
        # Check for README files in subdirectories
        subdirs = [d for d in self.configs_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
        
        missing_docs = []
        for subdir in subdirs:
            readme = subdir / "README.md"
            if readme.exists():
                print(f"  ✓ {subdir.name}/: Has README.md")
            else:
                missing_docs.append(subdir.name)
                print(f"  ℹ {subdir.name}/: No README.md")
        
        if missing_docs:
            self.results.info.append(f"Missing README.md in: {', '.join(missing_docs)}")
    
    def _print_summary(self):
        """Print validation summary"""
        print("=" * 80)
        print("Validation Summary")
        print("=" * 80)
        
        status = "✓ PASSED" if self.results.passed else "✗ FAILED"
        print(f"Status:   {status}")
        print(f"Errors:   {len(self.results.errors)}")
        print(f"Warnings: {len(self.results.warnings)}")
        print(f"Info:     {len(self.results.info)}")
        print()
        
        if self.results.errors:
            print("ERRORS:")
            for error in self.results.errors:
                print(f"  ✗ {error}")
            print()
        
        if self.results.warnings:
            print("WARNINGS:")
            for warning in self.results.warnings:
                print(f"  ⚠ {warning}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate VulcanAMI_LLM configuration files"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error on warnings"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix fixable issues (not implemented)"
    )
    args = parser.parse_args()
    
    # Determine configs directory
    script_dir = Path(__file__).parent
    configs_dir = script_dir
    
    # If script is in configs/, we're good
    # If script is elsewhere, look for configs/
    if not (configs_dir / "__init__.py").exists():
        configs_dir = Path.cwd() / "configs"
        if not configs_dir.exists():
            print("Error: Could not find configs directory")
            sys.exit(1)
    
    validator = ConfigValidator(configs_dir, strict=args.strict)
    result = validator.validate()
    
    # Exit with appropriate code
    if not result.passed:
        sys.exit(1)
    elif args.strict and result.warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
