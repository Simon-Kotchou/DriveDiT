#!/usr/bin/env python3
"""
DriveDiT DataCapture Mod Build Script

Automates the build process for the Enfusion mod.
Integrates with Workbench CLI when available.

Usage:
    python scripts/build.py [--platform PC] [--config release] [--clean]

Options:
    --platform    Target platform (PC, PC_WB, HEADLESS)
    --config      Build configuration (debug, release)
    --clean       Clean build artifacts before building
    --validate    Run validation before build
    --package     Create release package after build
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import validation if available
try:
    from validate import ModValidator, ValidationReport
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False


class ModBuilder:
    """Handles build automation for Enfusion mods."""

    # Build directories
    BUILD_DIR = "build"
    OUTPUT_DIR = "output"
    PACKAGE_DIR = "packages"

    # Platform configurations
    PLATFORMS = {
        "PC": {"suffix": "_pc", "defines": ["PLATFORM_PC"]},
        "PC_WB": {"suffix": "_wb", "defines": ["PLATFORM_PC", "WORKBENCH"]},
        "HEADLESS": {"suffix": "_headless", "defines": ["PLATFORM_PC", "HEADLESS"]}
    }

    # Build configurations
    CONFIGS = {
        "debug": {"defines": ["DEBUG"], "optimize": False},
        "release": {"defines": ["RELEASE", "NDEBUG"], "optimize": True}
    }

    def __init__(self, mod_path: Path, platform: str = "PC", config: str = "release"):
        self.mod_path = mod_path
        self.platform = platform
        self.config = config
        self.build_path = mod_path / self.BUILD_DIR
        self.output_path = mod_path / self.OUTPUT_DIR
        self.package_path = mod_path / self.PACKAGE_DIR

        # Load project configuration
        self.gproj = self._load_gproj()

    def _load_gproj(self) -> Dict[str, Any]:
        """Load and parse the .gproj file."""
        gproj_path = self.mod_path / "DriveDiT_DataCapture.gproj"
        if not gproj_path.exists():
            raise FileNotFoundError(f".gproj file not found: {gproj_path}")

        with open(gproj_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean(self):
        """Clean build artifacts."""
        print("Cleaning build artifacts...")

        dirs_to_clean = [self.build_path, self.output_path]
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  Removed: {dir_path}")

        print("Clean complete.")

    def validate(self) -> bool:
        """Run validation before build."""
        if not HAS_VALIDATOR:
            print("Warning: Validator not available, skipping validation")
            return True

        print("Running pre-build validation...")
        validator = ModValidator(self.mod_path)
        report = validator.validate()

        if not report.passed:
            print(f"Validation failed with {report.errors} errors")
            return False

        print(f"Validation passed ({report.warnings} warnings)")
        return True

    def prepare_build(self):
        """Prepare build directories."""
        print("Preparing build directories...")

        # Create build directories
        self.build_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create platform-specific output directory
        platform_output = self.output_path / f"{self.gproj['name']}_{self.platform.lower()}"
        platform_output.mkdir(parents=True, exist_ok=True)

        return platform_output

    def copy_resources(self, output_dir: Path):
        """Copy mod resources to build directory."""
        print("Copying resources...")

        # Directories to copy
        resource_dirs = [
            "Scripts",
            "Prefabs",
            "Configs"
        ]

        for dir_name in resource_dirs:
            src_dir = self.mod_path / dir_name
            if src_dir.exists():
                dst_dir = output_dir / dir_name
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                print(f"  Copied: {dir_name}")

        # Copy .gproj file
        gproj_src = self.mod_path / f"{self.gproj['name']}.gproj"
        gproj_dst = output_dir / f"{self.gproj['name']}.gproj"
        if gproj_src.exists():
            shutil.copy2(gproj_src, gproj_dst)
            print("  Copied: .gproj")

    def generate_version_file(self, output_dir: Path):
        """Generate version information file."""
        version_info = {
            "name": self.gproj.get("name", "DriveDiT_DataCapture"),
            "title": self.gproj.get("title", "DriveDiT Data Capture"),
            "version": self.gproj.get("version", "1.0.0"),
            "guid": self.gproj.get("guid", ""),
            "build_date": datetime.now().isoformat(),
            "platform": self.platform,
            "config": self.config,
            "git_commit": self._get_git_commit()
        }

        version_file = output_dir / "version.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2)

        print(f"  Generated: version.json")

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.mod_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def build_scripts(self, output_dir: Path) -> bool:
        """Compile scripts (if Workbench CLI available)."""
        print("Building scripts...")

        # Check for Workbench CLI
        workbench_cli = self._find_workbench_cli()
        if workbench_cli:
            return self._build_with_workbench(workbench_cli, output_dir)
        else:
            print("  Workbench CLI not found, scripts will be compiled at runtime")
            return True

    def _find_workbench_cli(self) -> Optional[Path]:
        """Find Workbench CLI executable."""
        # Common installation paths
        search_paths = [
            Path(os.environ.get("ARMA_REFORGER_TOOLS", "")) / "ArmaReforgerWorkbenchSteam.exe",
            Path("C:/Program Files (x86)/Steam/steamapps/common/Arma Reforger Tools/ArmaReforgerWorkbenchSteam.exe"),
            Path("D:/SteamLibrary/steamapps/common/Arma Reforger Tools/ArmaReforgerWorkbenchSteam.exe"),
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _build_with_workbench(self, workbench: Path, output_dir: Path) -> bool:
        """Build using Workbench CLI."""
        print(f"  Using Workbench: {workbench}")

        # Build command
        cmd = [
            str(workbench),
            "-buildProject",
            str(self.mod_path / f"{self.gproj['name']}.gproj"),
            "-buildOutput",
            str(output_dir),
            "-platform",
            self.platform
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  Build failed: {result.stderr}")
                return False
            print("  Build successful")
            return True
        except subprocess.TimeoutExpired:
            print("  Build timed out")
            return False
        except Exception as e:
            print(f"  Build error: {e}")
            return False

    def build(self, clean: bool = False, validate: bool = True) -> bool:
        """Run the complete build process."""
        print("=" * 60)
        print(f"Building: {self.gproj['name']} v{self.gproj.get('version', '?')}")
        print(f"Platform: {self.platform}, Config: {self.config}")
        print("=" * 60)

        # Clean if requested
        if clean:
            self.clean()

        # Validate if requested
        if validate:
            if not self.validate():
                return False

        # Prepare build
        output_dir = self.prepare_build()

        # Copy resources
        self.copy_resources(output_dir)

        # Generate version file
        self.generate_version_file(output_dir)

        # Build scripts
        if not self.build_scripts(output_dir):
            return False

        print("=" * 60)
        print(f"Build complete: {output_dir}")
        print("=" * 60)

        return True

    def package(self, include_source: bool = False) -> Optional[Path]:
        """Create a release package."""
        print("Creating release package...")

        # Ensure package directory exists
        self.package_path.mkdir(parents=True, exist_ok=True)

        # Package name
        version = self.gproj.get("version", "1.0.0")
        timestamp = datetime.now().strftime("%Y%m%d")
        package_name = f"{self.gproj['name']}_v{version}_{self.platform.lower()}_{timestamp}"

        # Source directory (built output)
        source_dir = self.output_path / f"{self.gproj['name']}_{self.platform.lower()}"
        if not source_dir.exists():
            print(f"Error: Build output not found: {source_dir}")
            print("Run build first.")
            return None

        # Create archive
        archive_path = self.package_path / package_name
        shutil.make_archive(
            str(archive_path),
            'zip',
            source_dir.parent,
            source_dir.name
        )

        final_path = Path(str(archive_path) + ".zip")
        print(f"Package created: {final_path}")
        print(f"Size: {final_path.stat().st_size / 1024 / 1024:.2f} MB")

        return final_path


def main():
    parser = argparse.ArgumentParser(description="Build DriveDiT DataCapture Mod")
    parser.add_argument('--platform', type=str, default='PC',
                        choices=['PC', 'PC_WB', 'HEADLESS'],
                        help='Target platform')
    parser.add_argument('--config', type=str, default='release',
                        choices=['debug', 'release'],
                        help='Build configuration')
    parser.add_argument('--clean', action='store_true',
                        help='Clean build artifacts before building')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Run validation before build')
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                        help='Skip validation')
    parser.add_argument('--package', action='store_true',
                        help='Create release package after build')
    parser.add_argument('--mod-path', type=str,
                        help='Path to mod directory')

    args = parser.parse_args()

    # Determine mod path
    if args.mod_path:
        mod_path = Path(args.mod_path)
    else:
        mod_path = Path(__file__).parent.parent

    if not mod_path.is_dir():
        print(f"Error: Mod directory not found: {mod_path}")
        sys.exit(1)

    try:
        # Create builder
        builder = ModBuilder(mod_path, args.platform, args.config)

        # Run build
        if not builder.build(clean=args.clean, validate=args.validate):
            print("Build failed!")
            sys.exit(1)

        # Create package if requested
        if args.package:
            package_path = builder.package()
            if not package_path:
                print("Packaging failed!")
                sys.exit(1)

        print("\nBuild completed successfully!")

    except Exception as e:
        print(f"Build error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
