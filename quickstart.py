#!/usr/bin/env python3
"""
Quick Start Guide - Verify NEXUS-Ω Installation
Runs basic smoke tests to ensure system is working correctly.
"""

import sys
import subprocess


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    try:
        import numpy
        import nexus_seed
        import nexus_reflection_loop
        import nexus_spawn
        import nexus_tokenizer
        import nexus_omega
        import victor_monolith
        import asi_core
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def run_tests():
    """Run the test suite."""
    print("\nRunning test suite...")
    result = subprocess.run(
        [sys.executable, "test_nexus.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ All tests passed")
        return True
    else:
        print(f"❌ Tests failed:\n{result.stdout}\n{result.stderr}")
        return False


def run_demo():
    """Run a quick demo."""
    print("\nRunning quick demo (3 iterations)...")
    result = subprocess.run(
        [sys.executable, "demo_nexus.py", "3"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("✅ Demo completed successfully")
        # Show last few lines of output
        lines = result.stdout.strip().split('\n')
        print("\nDemo output (last 10 lines):")
        for line in lines[-10:]:
            print(f"  {line}")
        return True
    else:
        print(f"❌ Demo failed:\n{result.stdout}\n{result.stderr}")
        return False


def main():
    """Run all quick start checks."""
    print("=" * 70)
    print("NEXUS-Ω Quick Start - Installation Verification")
    print("=" * 70)
    print()
    
    results = {
        "Imports": test_imports(),
        "Tests": run_tests(),
        "Demo": run_demo()
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 System is ready! All checks passed.")
        print("\nNext steps:")
        print("  1. Run full demo: python3 demo_nexus.py 10")
        print("  2. Test Victor Monolith: python3 victor_monolith.py")
        print("  3. Start autonomous operation: python3 nexus_omega.py")
        print("     (Use Ctrl+C to stop gracefully)")
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
