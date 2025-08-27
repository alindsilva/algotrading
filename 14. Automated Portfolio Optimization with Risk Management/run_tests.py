#!/usr/bin/env python3
"""
Test runner script for the IBKR trading application.
Provides different test execution modes and coverage reporting.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, **kwargs):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode == 0


def run_tests(test_type="all", coverage=True, verbose=False, fail_fast=False):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test paths based on type
    if test_type == "unit":
        cmd.extend([
            "tests/test_core",
            "tests/test_contracts", 
            "tests/test_analytics",
            "tests/test_data",
            "-m", "unit"
        ])
    elif test_type == "integration":
        cmd.extend([
            "tests/test_app",
            "-m", "integration"
        ])
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}")
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing", 
            "--cov-report=html:htmlcov",
            "--cov-fail-under=80"
        ])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add fail fast
    if fail_fast:
        cmd.append("-x")
    
    # Add asyncio mode
    cmd.extend(["--asyncio-mode=auto"])
    
    # Run the tests
    return run_command(cmd)


def run_linting():
    """Run code linting checks."""
    print("Running linting checks...")
    
    success = True
    
    # Run flake8
    print("\n=== Running flake8 ===")
    if not run_command(["python", "-m", "flake8", "src/", "tests/", "--max-line-length=100"]):
        success = False
    
    # Run black check
    print("\n=== Running black check ===")
    if not run_command(["python", "-m", "black", "--check", "src/", "tests/"]):
        success = False
    
    return success


def run_type_checking():
    """Run type checking with mypy."""
    print("\n=== Running mypy type checking ===")
    return run_command([
        "python", "-m", "mypy", "src/", 
        "--ignore-missing-imports",
        "--disallow-untyped-defs"
    ])


def format_code():
    """Format code with black."""
    print("Formatting code with black...")
    return run_command(["python", "-m", "black", "src/", "tests/"])


def clean_cache():
    """Clean pytest and coverage cache files."""
    print("Cleaning cache files...")
    
    cache_dirs = [
        ".pytest_cache",
        "__pycache__",
        "htmlcov",
        ".coverage"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            if os.path.isfile(cache_dir):
                os.remove(cache_dir)
            else:
                import shutil
                shutil.rmtree(cache_dir)
    
    # Remove __pycache__ directories recursively
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                import shutil
                shutil.rmtree(cache_path)
    
    print("Cache cleaned!")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="IBKR Trading Application Test Runner")
    
    parser.add_argument(
        "action",
        choices=["test", "lint", "type-check", "format", "clean", "all"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true", 
        help="Stop on first test failure"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = True
    
    if args.action == "test":
        success = run_tests(
            test_type=args.test_type,
            coverage=not args.no_coverage,
            verbose=args.verbose,
            fail_fast=args.fail_fast
        )
        
    elif args.action == "lint":
        success = run_linting()
        
    elif args.action == "type-check":
        success = run_type_checking()
        
    elif args.action == "format":
        success = format_code()
        
    elif args.action == "clean":
        clean_cache()
        
    elif args.action == "all":
        print("Running complete test suite...")
        
        # Clean first
        clean_cache()
        
        # Format code
        print("\n" + "="*60)
        print("FORMATTING CODE")
        print("="*60)
        format_success = format_code()
        
        # Run linting
        print("\n" + "="*60) 
        print("LINTING")
        print("="*60)
        lint_success = run_linting()
        
        # Run type checking
        print("\n" + "="*60)
        print("TYPE CHECKING")  
        print("="*60)
        type_success = run_type_checking()
        
        # Run tests
        print("\n" + "="*60)
        print("UNIT TESTS")
        print("="*60)
        unit_success = run_tests(
            test_type="unit",
            coverage=not args.no_coverage,
            verbose=args.verbose
        )
        
        print("\n" + "="*60)
        print("INTEGRATION TESTS")
        print("="*60)
        integration_success = run_tests(
            test_type="integration", 
            coverage=False,  # Coverage already done in unit tests
            verbose=args.verbose
        )
        
        # Overall success
        success = all([
            format_success, lint_success, type_success,
            unit_success, integration_success
        ])
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Formatting: {'✓ PASS' if format_success else '✗ FAIL'}")
        print(f"Linting: {'✓ PASS' if lint_success else '✗ FAIL'}")
        print(f"Type Checking: {'✓ PASS' if type_success else '✗ FAIL'}")
        print(f"Unit Tests: {'✓ PASS' if unit_success else '✗ FAIL'}")
        print(f"Integration Tests: {'✓ PASS' if integration_success else '✗ FAIL'}")
        print(f"\nOverall: {'✓ SUCCESS' if success else '✗ FAILURE'}")
    
    if success:
        print("\n🎉 All checks passed!")
        return 0
    else:
        print("\n❌ Some checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
