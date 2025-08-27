#!/usr/bin/env python3
"""
Script to resolve dependency conflicts with vectorbtpro and openbb packages
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("✅ Success!")
            # Only show first few lines to avoid spam
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                for line in lines[:5]:
                    print(f"   {line}")
                print(f"   ... ({len(lines)-10} more lines)")
                for line in lines[-5:]:
                    print(f"   {line}")
            else:
                for line in lines:
                    print(f"   {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    print("🔧 Fixing dependency conflicts for IBKR Trading Application")
    print("=" * 60)
    
    # Step 1: Downgrade numpy to compatible version
    print("\n📦 Step 1: Installing compatible numpy version")
    if not run_command([
        sys.executable, "-m", "pip", "install", "numpy>=1.22.4,<2.0.0", "--force-reinstall"
    ], "Downgrading numpy to <2.0.0 for compatibility"):
        return False
    
    # Step 2: Downgrade numba to compatible version for vectorbtpro
    print("\n📦 Step 2: Installing compatible numba version")
    if not run_command([
        sys.executable, "-m", "pip", "install", "numba>=0.56.0,<0.57.0", "--force-reinstall"
    ], "Downgrading numba for vectorbtpro compatibility"):
        return False
    
    # Step 3: Install/upgrade other requirements
    print("\n📦 Step 3: Installing other requirements")
    if not run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], "Installing requirements from requirements.txt"):
        return False
    
    # Step 4: Verify installations
    print("\n🔍 Step 4: Verifying installations")
    packages_to_check = [
        "numpy", "numba", "pandas", "aiosqlite", "pytest", 
        "python-dotenv", "ibapi", "vectorbtpro"
    ]
    
    for package in packages_to_check:
        if not run_command([
            sys.executable, "-c", f"import {package.replace('-', '_')}; print(f'{package}: OK')"
        ], f"Checking {package}"):
            print(f"⚠️  {package} might not be properly installed")
    
    # Step 5: Show final package versions
    print("\n📋 Final package versions:")
    if not run_command([
        sys.executable, "-m", "pip", "list"
    ], "Listing installed packages"):
        return False
    
    # Step 6: Run a quick test
    print("\n🧪 Running quick compatibility test...")
    test_script = '''
import numpy as np
import numba
import pandas as pd
import aiosqlite
print(f"numpy version: {np.__version__}")
print(f"numba version: {numba.__version__}")
print(f"pandas version: {pd.__version__}")
print("✅ All critical packages imported successfully!")
'''
    
    if run_command([
        sys.executable, "-c", test_script
    ], "Testing package imports"):
        print("\n🎉 Dependency resolution completed successfully!")
        print("\nYou can now run:")
        print("  python -m pytest tests/ -v")
        print("  python main.py")
        return True
    else:
        print("\n❌ Some packages failed to import. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
