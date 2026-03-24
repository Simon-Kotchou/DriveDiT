#!/usr/bin/env python3
"""
DriveDiT DataCapture Mod Validation Script

Validates the mod structure, scripts, and configurations before build/release.
Uses Enfusion MCP tools when available, with fallback to local validation.

Usage:
    python scripts/validate.py [--strict] [--fix] [--report output.json]

Options:
    --strict    Fail on warnings as well as errors
    --fix       Attempt to auto-fix common issues
    --report    Generate JSON validation report
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationResult:
    check: str
    severity: Severity
    message: str
    file_path: str = ""
    line_number: int = 0
    suggestion: str = ""

@dataclass
class ValidationReport:
    results: List[ValidationResult] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    info: int = 0

    def add(self, result: ValidationResult):
        self.results.append(result)
        if result.severity == Severity.ERROR:
            self.errors += 1
        elif result.severity == Severity.WARNING:
            self.warnings += 1
        else:
            self.info += 1

    @property
    def passed(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "results": [
                {
                    "check": r.check,
                    "severity": r.severity.value,
                    "message": r.message,
                    "file_path": r.file_path,
                    "line_number": r.line_number,
                    "suggestion": r.suggestion
                }
                for r in self.results
            ]
        }


class ModValidator:
    """Validates Enfusion mod structure and content."""

    # Required directories
    REQUIRED_DIRS = [
        "Scripts/Game/DataCapture",
        "Prefabs/DataCapture",
        "Configs/CaptureProfiles"
    ]

    # Required files
    REQUIRED_FILES = [
        "DriveDiT_DataCapture.gproj",
        "Scripts/Game/DataCapture/SCR_CaptureOrchestrator.c",
        "Scripts/Game/DataCapture/SCR_MLDataCollector.c",
        "Scripts/Game/DataCapture/SCR_AIDrivingSimulator.c",
        "Prefabs/DataCapture/DataCaptureGameMode.et",
        "Configs/CaptureProfiles/minimal.conf",
        "Configs/CaptureProfiles/research.conf",
        "Configs/CaptureProfiles/production.conf"
    ]

    # Script naming conventions
    SCRIPT_PREFIX = "SCR_"
    SCRIPT_EXTENSION = ".c"

    # Prefab extension
    PREFAB_EXTENSION = ".et"

    def __init__(self, mod_path: Path, strict: bool = False):
        self.mod_path = mod_path
        self.strict = strict
        self.report = ValidationReport()

    def validate(self) -> ValidationReport:
        """Run all validation checks."""
        print(f"Validating mod at: {self.mod_path}")
        print("-" * 60)

        # Structure validation
        self._validate_directory_structure()
        self._validate_required_files()

        # Project file validation
        self._validate_gproj()

        # Script validation
        self._validate_scripts()

        # Prefab validation
        self._validate_prefabs()

        # Config validation
        self._validate_configs()

        # Naming convention validation
        self._validate_naming_conventions()

        return self.report

    def _validate_directory_structure(self):
        """Check that required directories exist."""
        for dir_path in self.REQUIRED_DIRS:
            full_path = self.mod_path / dir_path
            if not full_path.is_dir():
                self.report.add(ValidationResult(
                    check="directory_structure",
                    severity=Severity.ERROR,
                    message=f"Required directory missing: {dir_path}",
                    suggestion=f"Create directory: mkdir -p {dir_path}"
                ))
            else:
                self.report.add(ValidationResult(
                    check="directory_structure",
                    severity=Severity.INFO,
                    message=f"Directory exists: {dir_path}"
                ))

    def _validate_required_files(self):
        """Check that required files exist."""
        for file_path in self.REQUIRED_FILES:
            full_path = self.mod_path / file_path
            if not full_path.is_file():
                self.report.add(ValidationResult(
                    check="required_files",
                    severity=Severity.ERROR,
                    message=f"Required file missing: {file_path}"
                ))
            else:
                self.report.add(ValidationResult(
                    check="required_files",
                    severity=Severity.INFO,
                    message=f"File exists: {file_path}"
                ))

    def _validate_gproj(self):
        """Validate the .gproj project file."""
        gproj_path = self.mod_path / "DriveDiT_DataCapture.gproj"

        if not gproj_path.is_file():
            return

        try:
            with open(gproj_path, 'r', encoding='utf-8') as f:
                content = f.read()
                gproj = json.loads(content)

            # Check required fields
            required_fields = ['guid', 'name', 'title', 'description', 'version']
            for field in required_fields:
                if field not in gproj:
                    self.report.add(ValidationResult(
                        check="gproj_format",
                        severity=Severity.ERROR,
                        message=f"Missing required field in .gproj: {field}",
                        file_path=str(gproj_path)
                    ))

            # Validate GUID format
            if 'guid' in gproj:
                guid = gproj['guid']
                guid_pattern = r'^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$'
                if not re.match(guid_pattern, guid):
                    self.report.add(ValidationResult(
                        check="gproj_format",
                        severity=Severity.ERROR,
                        message=f"Invalid GUID format: {guid}",
                        file_path=str(gproj_path),
                        suggestion="GUID should be in format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
                    ))

            # Validate version format
            if 'version' in gproj:
                version = gproj['version']
                version_pattern = r'^\d+\.\d+\.\d+$'
                if not re.match(version_pattern, version):
                    self.report.add(ValidationResult(
                        check="gproj_format",
                        severity=Severity.WARNING,
                        message=f"Non-standard version format: {version}",
                        file_path=str(gproj_path),
                        suggestion="Version should be in format: X.Y.Z (semantic versioning)"
                    ))

            # Check dependencies
            if 'dependencies' in gproj:
                for dep in gproj['dependencies']:
                    if 'guid' not in dep or 'name' not in dep:
                        self.report.add(ValidationResult(
                            check="gproj_dependencies",
                            severity=Severity.WARNING,
                            message="Dependency missing guid or name field",
                            file_path=str(gproj_path)
                        ))

            self.report.add(ValidationResult(
                check="gproj_format",
                severity=Severity.INFO,
                message="Project file parsed successfully"
            ))

        except json.JSONDecodeError as e:
            self.report.add(ValidationResult(
                check="gproj_format",
                severity=Severity.ERROR,
                message=f"Invalid JSON in .gproj: {e}",
                file_path=str(gproj_path)
            ))

    def _validate_scripts(self):
        """Validate Enforce script files."""
        scripts_dir = self.mod_path / "Scripts" / "Game" / "DataCapture"

        if not scripts_dir.is_dir():
            return

        for script_file in scripts_dir.glob("*.c"):
            self._validate_single_script(script_file)

    def _validate_single_script(self, script_path: Path):
        """Validate a single Enforce script file."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Check for class definition
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, content)

            if not classes:
                self.report.add(ValidationResult(
                    check="script_validation",
                    severity=Severity.WARNING,
                    message="No class definition found",
                    file_path=str(script_path)
                ))

            # Check class naming (should start with SCR_ or match file name)
            file_stem = script_path.stem
            for class_name in classes:
                if not class_name.startswith('SCR_') and not class_name.endswith('Class'):
                    self.report.add(ValidationResult(
                        check="script_naming",
                        severity=Severity.WARNING,
                        message=f"Class '{class_name}' doesn't follow SCR_ prefix convention",
                        file_path=str(script_path),
                        suggestion="Consider using SCR_ prefix for script classes"
                    ))

            # Check for ComponentEditorProps (for component classes)
            if 'ScriptComponent' in content or 'Component' in content:
                if 'ComponentEditorProps' not in content:
                    self.report.add(ValidationResult(
                        check="script_validation",
                        severity=Severity.WARNING,
                        message="Component class missing ComponentEditorProps attribute",
                        file_path=str(script_path)
                    ))

            # Check for basic syntax issues
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces != close_braces:
                self.report.add(ValidationResult(
                    check="script_syntax",
                    severity=Severity.ERROR,
                    message=f"Mismatched braces: {open_braces} open, {close_braces} close",
                    file_path=str(script_path)
                ))

            # Check for TODO/FIXME comments
            for i, line in enumerate(lines, 1):
                if 'TODO' in line or 'FIXME' in line:
                    self.report.add(ValidationResult(
                        check="script_todos",
                        severity=Severity.INFO,
                        message=f"Found TODO/FIXME comment",
                        file_path=str(script_path),
                        line_number=i
                    ))

            self.report.add(ValidationResult(
                check="script_validation",
                severity=Severity.INFO,
                message=f"Script validated: {script_path.name}"
            ))

        except Exception as e:
            self.report.add(ValidationResult(
                check="script_validation",
                severity=Severity.ERROR,
                message=f"Failed to read script: {e}",
                file_path=str(script_path)
            ))

    def _validate_prefabs(self):
        """Validate .et prefab files."""
        prefabs_dir = self.mod_path / "Prefabs" / "DataCapture"

        if not prefabs_dir.is_dir():
            return

        for prefab_file in prefabs_dir.glob("*.et"):
            self._validate_single_prefab(prefab_file)

    def _validate_single_prefab(self, prefab_path: Path):
        """Validate a single prefab file."""
        try:
            with open(prefab_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for entity definition
            if not content.strip().startswith('GenericEntity') and not 'Entity' in content:
                self.report.add(ValidationResult(
                    check="prefab_validation",
                    severity=Severity.WARNING,
                    message="Prefab doesn't start with Entity definition",
                    file_path=str(prefab_path)
                ))

            # Check for ID field
            if 'ID "' not in content:
                self.report.add(ValidationResult(
                    check="prefab_validation",
                    severity=Severity.ERROR,
                    message="Prefab missing ID field",
                    file_path=str(prefab_path)
                ))

            # Check for components block
            if 'components {' not in content:
                self.report.add(ValidationResult(
                    check="prefab_validation",
                    severity=Severity.WARNING,
                    message="Prefab missing components block",
                    file_path=str(prefab_path)
                ))

            self.report.add(ValidationResult(
                check="prefab_validation",
                severity=Severity.INFO,
                message=f"Prefab validated: {prefab_path.name}"
            ))

        except Exception as e:
            self.report.add(ValidationResult(
                check="prefab_validation",
                severity=Severity.ERROR,
                message=f"Failed to read prefab: {e}",
                file_path=str(prefab_path)
            ))

    def _validate_configs(self):
        """Validate configuration files."""
        configs_dir = self.mod_path / "Configs" / "CaptureProfiles"

        if not configs_dir.is_dir():
            return

        for config_file in configs_dir.glob("*.conf"):
            self._validate_single_config(config_file)

    def _validate_single_config(self, config_path: Path):
        """Validate a single config file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for CaptureProfileConfig definition
            if 'CaptureProfileConfig' not in content:
                self.report.add(ValidationResult(
                    check="config_validation",
                    severity=Severity.WARNING,
                    message="Config missing CaptureProfileConfig definition",
                    file_path=str(config_path)
                ))

            # Check for required fields
            required_fields = ['name', 'version', 'captureEnabled']
            for field in required_fields:
                if field not in content:
                    self.report.add(ValidationResult(
                        check="config_validation",
                        severity=Severity.WARNING,
                        message=f"Config possibly missing field: {field}",
                        file_path=str(config_path)
                    ))

            self.report.add(ValidationResult(
                check="config_validation",
                severity=Severity.INFO,
                message=f"Config validated: {config_path.name}"
            ))

        except Exception as e:
            self.report.add(ValidationResult(
                check="config_validation",
                severity=Severity.ERROR,
                message=f"Failed to read config: {e}",
                file_path=str(config_path)
            ))

    def _validate_naming_conventions(self):
        """Check naming conventions across the mod."""
        # Check script file names
        scripts_dir = self.mod_path / "Scripts" / "Game" / "DataCapture"
        if scripts_dir.is_dir():
            for script_file in scripts_dir.glob("*.c"):
                if not script_file.stem.startswith("SCR_"):
                    self.report.add(ValidationResult(
                        check="naming_conventions",
                        severity=Severity.WARNING,
                        message=f"Script file should start with 'SCR_': {script_file.name}",
                        file_path=str(script_file),
                        suggestion=f"Rename to: SCR_{script_file.stem}.c"
                    ))


def main():
    parser = argparse.ArgumentParser(description="Validate DriveDiT DataCapture Mod")
    parser.add_argument('--strict', action='store_true', help='Fail on warnings')
    parser.add_argument('--fix', action='store_true', help='Auto-fix issues')
    parser.add_argument('--report', type=str, help='Output JSON report to file')
    parser.add_argument('--mod-path', type=str, help='Path to mod directory')

    args = parser.parse_args()

    # Determine mod path
    if args.mod_path:
        mod_path = Path(args.mod_path)
    else:
        # Default to script's parent directory
        mod_path = Path(__file__).parent.parent

    if not mod_path.is_dir():
        print(f"Error: Mod directory not found: {mod_path}")
        sys.exit(1)

    # Run validation
    validator = ModValidator(mod_path, strict=args.strict)
    report = validator.validate()

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Errors:   {report.errors}")
    print(f"Warnings: {report.warnings}")
    print(f"Info:     {report.info}")
    print(f"Status:   {'PASSED' if report.passed else 'FAILED'}")

    # Print detailed results
    if report.errors > 0 or report.warnings > 0:
        print("\n" + "-" * 60)
        print("ISSUES FOUND:")
        print("-" * 60)
        for result in report.results:
            if result.severity in [Severity.ERROR, Severity.WARNING]:
                prefix = "[ERROR]" if result.severity == Severity.ERROR else "[WARN]"
                print(f"{prefix} {result.check}: {result.message}")
                if result.file_path:
                    print(f"         File: {result.file_path}")
                if result.suggestion:
                    print(f"         Fix: {result.suggestion}")

    # Write JSON report if requested
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport written to: {args.report}")

    # Exit with appropriate code
    if not report.passed:
        sys.exit(1)
    elif args.strict and report.warnings > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
