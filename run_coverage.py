"""Build Debug and run OpenCppCoverage on monti_tests.

Produces: coverage/ (HTML) and coverage.xml (Cobertura) at repo root.

Usage:
    python run_coverage.py [--build-dir build] [--config Debug]
"""

import argparse
import os
import shutil
import subprocess
import sys


def _find_opencppcoverage() -> str:
    """Locate OpenCppCoverage executable."""
    occ = shutil.which("OpenCppCoverage")
    if occ:
        return occ

    default = r"C:\Program Files\OpenCppCoverage\OpenCppCoverage.exe"
    if os.path.isfile(default):
        return default

    print("Error: OpenCppCoverage not found on PATH or in default location.",
          file=sys.stderr)
    print("Install from https://github.com/OpenCppCoverage/OpenCppCoverage/releases",
          file=sys.stderr)
    sys.exit(1)


def run_coverage(build_dir: str, config: str) -> None:
    """Build tests and run coverage."""
    repo_root = os.path.dirname(os.path.abspath(__file__))

    occ_exe = _find_opencppcoverage()
    print(f"Using: {occ_exe}")

    # Build
    print(f"=== Building {config} configuration ===")
    build_path = os.path.join(repo_root, build_dir)
    result = subprocess.run(
        ["cmake", "--build", build_path, "--config", config, "--target", "monti_tests"])
    if result.returncode != 0:
        print("Error: Build failed.", file=sys.stderr)
        sys.exit(1)

    # Locate test executable
    test_exe = os.path.join(build_path, config, "monti_tests.exe")
    if not os.path.isfile(test_exe):
        print(f"Error: Test executable not found at {test_exe}", file=sys.stderr)
        sys.exit(1)

    # Source directories to include in coverage
    source_dirs = ["renderer", "scene", "denoise", "capture", "app"]

    # Build OpenCppCoverage arguments
    occ_args = [occ_exe]
    for d in source_dirs:
        occ_args.extend(["--sources", os.path.join(repo_root, d)])

    # Exclude test code, build artifacts, and third-party/fetched dependencies
    for d in ["tests", "build", "external", "deps", "_deps"]:
        occ_args.extend(["--excluded_sources", os.path.join(repo_root, d)])

    # HTML report
    html_dir = os.path.join(repo_root, "coverage")
    occ_args.extend(["--export_type", f"html:{html_dir}"])

    # Cobertura XML report
    xml_file = os.path.join(repo_root, "coverage.xml")
    occ_args.extend(["--export_type", f"cobertura:{xml_file}"])

    # Test executable
    occ_args.extend(["--", test_exe])

    print("=== Running OpenCppCoverage ===")
    result = subprocess.run(occ_args)
    if result.returncode != 0:
        print("Error: OpenCppCoverage failed.", file=sys.stderr)
        sys.exit(1)

    print()
    print("=== Coverage complete ===")
    print(f"  HTML report : {html_dir}{os.sep}index.html")
    print(f"  Cobertura   : {xml_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build and run C++ coverage with OpenCppCoverage")
    parser.add_argument("--build-dir", default="build",
                        help="CMake build directory (default: build)")
    parser.add_argument("--config", default="Debug",
                        help="Build configuration (default: Debug)")
    args = parser.parse_args()

    run_coverage(args.build_dir, args.config)


if __name__ == "__main__":
    main()
