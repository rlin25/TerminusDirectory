#!/usr/bin/env python3
"""
Comprehensive test runner for the rental ML system.

This script provides various test running options and generates detailed reports.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional


class TestRunner:
    """Comprehensive test runner with reporting capabilities."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_dir = self.project_root / "tests"
        
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run all unit tests."""
        print("🔬 Running unit tests...")
        
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v" if verbose else "-q"]
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
        
        cmd.extend(["-m", "unit"])
        
        return self._run_command(cmd)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run all integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = ["python", "-m", "pytest", "tests/integration/", "-v" if verbose else "-q"]
        cmd.extend(["-m", "integration"])
        
        return self._run_command(cmd)
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        
        cmd = ["python", "-m", "pytest", "tests/performance/", "-v" if verbose else "-q"]
        cmd.extend(["-m", "performance", "--durations=10"])
        
        return self._run_command(cmd)
    
    def run_ml_tests(self, verbose: bool = False) -> int:
        """Run ML-specific tests."""
        print("🤖 Running ML tests...")
        
        cmd = ["python", "-m", "pytest", "-v" if verbose else "-q"]
        cmd.extend(["-m", "ml", "--durations=10"])
        
        return self._run_command(cmd)
    
    def run_api_tests(self, verbose: bool = False) -> int:
        """Run API tests."""
        print("🌐 Running API tests...")
        
        cmd = ["python", "-m", "pytest", "-v" if verbose else "-q"]
        cmd.extend(["-m", "api"])
        
        return self._run_command(cmd)
    
    def run_smoke_tests(self, verbose: bool = False) -> int:
        """Run smoke tests for basic functionality."""
        print("💨 Running smoke tests...")
        
        cmd = ["python", "-m", "pytest", "-v" if verbose else "-q"]
        cmd.extend(["-m", "smoke", "-x"])  # Stop on first failure
        
        return self._run_command(cmd)
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True, 
                      exclude_slow: bool = False) -> Dict[str, int]:
        """Run all test suites."""
        print("🚀 Running full test suite...")
        
        results = {}
        
        # Unit tests with coverage
        results['unit'] = self.run_unit_tests(verbose, coverage)
        
        # Integration tests
        results['integration'] = self.run_integration_tests(verbose)
        
        # ML tests (excluding slow ones if requested)
        if not exclude_slow:
            results['ml'] = self.run_ml_tests(verbose)
        
        # API tests
        results['api'] = self.run_api_tests(verbose)
        
        # Performance tests (excluding slow ones if requested)
        if not exclude_slow:
            results['performance'] = self.run_performance_tests(verbose)
        
        return results
    
    def run_tests_by_marker(self, marker: str, verbose: bool = False) -> int:
        """Run tests with specific marker."""
        print(f"🏷️  Running tests with marker: {marker}")
        
        cmd = ["python", "-m", "pytest", "-v" if verbose else "-q"]
        cmd.extend(["-m", marker])
        
        return self._run_command(cmd)
    
    def run_tests_by_path(self, path: str, verbose: bool = False) -> int:
        """Run tests in specific path."""
        print(f"📁 Running tests in: {path}")
        
        cmd = ["python", "-m", "pytest", path, "-v" if verbose else "-q"]
        
        return self._run_command(cmd)
    
    def run_parallel_tests(self, num_workers: int = 4, verbose: bool = False) -> int:
        """Run tests in parallel using pytest-xdist."""
        print(f"⚡ Running tests in parallel with {num_workers} workers...")
        
        cmd = ["python", "-m", "pytest", "-v" if verbose else "-q"]
        cmd.extend(["-n", str(num_workers)])
        
        return self._run_command(cmd)
    
    def generate_coverage_report(self) -> int:
        """Generate detailed coverage report."""
        print("📊 Generating coverage report...")
        
        # Run tests with coverage
        cmd = ["python", "-m", "pytest", "--cov=src", 
               "--cov-report=html:htmlcov", "--cov-report=xml:coverage.xml",
               "--cov-report=term-missing", "--cov-fail-under=80"]
        
        result = self._run_command(cmd)
        
        if result == 0:
            print("✅ Coverage report generated:")
            print(f"   HTML report: {self.project_root}/htmlcov/index.html")
            print(f"   XML report: {self.project_root}/coverage.xml")
        
        return result
    
    def lint_and_format(self) -> int:
        """Run linting and formatting checks."""
        print("🧹 Running linting and formatting checks...")
        
        results = []
        
        # Black formatting check
        print("  Checking code formatting with black...")
        results.append(self._run_command(["black", "--check", "src/", "tests/"]))
        
        # isort import sorting check
        print("  Checking import sorting with isort...")
        results.append(self._run_command(["isort", "--check-only", "src/", "tests/"]))
        
        # flake8 linting
        print("  Running flake8 linting...")
        results.append(self._run_command(["flake8", "src/", "tests/"]))
        
        # mypy type checking
        print("  Running mypy type checking...")
        results.append(self._run_command(["mypy", "src/"]))
        
        return max(results) if results else 0
    
    def check_dependencies(self) -> int:
        """Check for dependency issues."""
        print("📦 Checking dependencies...")
        
        # Check for security vulnerabilities
        print("  Checking for security vulnerabilities...")
        safety_result = self._run_command(["safety", "check"], check=False)
        
        # Check for outdated packages
        print("  Checking for outdated packages...")
        pip_result = self._run_command(["pip", "list", "--outdated"], check=False)
        
        return 0  # These are informational
    
    def run_pre_commit_checks(self) -> int:
        """Run pre-commit checks."""
        print("🔍 Running pre-commit checks...")
        
        return self._run_command(["pre-commit", "run", "--all-files"])
    
    def _run_command(self, cmd: List[str], check: bool = True) -> int:
        """Run a command and return exit code."""
        try:
            print(f"Running: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(cmd, cwd=self.project_root, check=check)
            
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f}s")
            
            return result.returncode
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed with exit code {e.returncode}")
            return e.returncode
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            return 1
    
    def print_summary(self, results: Dict[str, int]):
        """Print test results summary."""
        print("\n" + "="*60)
        print("📋 TEST RESULTS SUMMARY")
        print("="*60)
        
        total_suites = len(results)
        passed_suites = sum(1 for code in results.values() if code == 0)
        
        for suite, exit_code in results.items():
            status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
            print(f"{suite.upper():15} {status}")
        
        print("-"*60)
        print(f"Total: {passed_suites}/{total_suites} test suites passed")
        
        if passed_suites == total_suites:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Check output above for details.")
        
        print("="*60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for rental ML system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --all                    # Run all tests
  python scripts/run_tests.py --unit --coverage        # Run unit tests with coverage
  python scripts/run_tests.py --ml --verbose           # Run ML tests with verbose output
  python scripts/run_tests.py --marker slow            # Run tests marked as 'slow'
  python scripts/run_tests.py --path tests/unit/       # Run tests in specific path
  python scripts/run_tests.py --parallel 8             # Run tests in parallel
  python scripts/run_tests.py --smoke                  # Run smoke tests only
  python scripts/run_tests.py --lint                   # Run linting checks
        """
    )
    
    # Test suite options
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--ml", action="store_true", help="Run ML tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    
    # Test execution options
    parser.add_argument("--marker", help="Run tests with specific marker")
    parser.add_argument("--path", help="Run tests in specific path")
    parser.add_argument("--parallel", type=int, help="Run tests in parallel with N workers")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", default=True, help="Generate coverage report")
    parser.add_argument("--no-coverage", dest="coverage", action="store_false", help="Skip coverage report")
    parser.add_argument("--exclude-slow", action="store_true", help="Exclude slow tests")
    
    # Quality checks
    parser.add_argument("--lint", action="store_true", help="Run linting and formatting checks")
    parser.add_argument("--deps", action="store_true", help="Check dependencies")
    parser.add_argument("--pre-commit", action="store_true", help="Run pre-commit checks")
    
    args = parser.parse_args()
    
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    runner = TestRunner(str(project_root))
    
    results = {}
    
    try:
        if args.lint:
            results['lint'] = runner.lint_and_format()
        
        if args.deps:
            results['deps'] = runner.check_dependencies()
        
        if args.pre_commit:
            results['pre-commit'] = runner.run_pre_commit_checks()
        
        if args.all:
            results.update(runner.run_all_tests(
                verbose=args.verbose,
                coverage=args.coverage,
                exclude_slow=args.exclude_slow
            ))
        
        elif args.unit:
            results['unit'] = runner.run_unit_tests(args.verbose, args.coverage)
        
        elif args.integration:
            results['integration'] = runner.run_integration_tests(args.verbose)
        
        elif args.performance:
            results['performance'] = runner.run_performance_tests(args.verbose)
        
        elif args.ml:
            results['ml'] = runner.run_ml_tests(args.verbose)
        
        elif args.api:
            results['api'] = runner.run_api_tests(args.verbose)
        
        elif args.smoke:
            results['smoke'] = runner.run_smoke_tests(args.verbose)
        
        elif args.marker:
            results[f'marker-{args.marker}'] = runner.run_tests_by_marker(args.marker, args.verbose)
        
        elif args.path:
            results[f'path-{args.path}'] = runner.run_tests_by_path(args.path, args.verbose)
        
        elif args.parallel:
            results['parallel'] = runner.run_parallel_tests(args.parallel, args.verbose)
        
        else:
            # Default to running smoke tests
            results['smoke'] = runner.run_smoke_tests(args.verbose)
        
        # Print summary if multiple test suites were run
        if len(results) > 1:
            runner.print_summary(results)
        
        # Exit with non-zero code if any tests failed
        exit_code = max(results.values()) if results else 0
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()