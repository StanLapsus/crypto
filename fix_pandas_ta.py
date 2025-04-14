#!/usr/bin/env python3
"""
Fix the NaN import issue in pandas_ta.
This script patches the pandas_ta library to replace imports of NaN from numpy
with the correct np.nan usage.
"""

import os
import sys
import importlib
import re

def find_pandas_ta_path():
    """Find the path where pandas_ta is installed"""
    try:
        import pandas_ta
        return os.path.dirname(pandas_ta.__file__)
    except ImportError:
        print("Error: pandas_ta is not installed.")
        return None

def patch_file(file_path):
    """Patch a file to replace 'from numpy import NaN' with correct numpy imports"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        # Read file content
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Replace problematic imports
        new_content = re.sub(
            r'from numpy import NaN(.*)',
            r'import numpy as np\1  # Fixed NaN import',
            content
        )
        
        # Replace NaN usages with np.nan
        new_content = new_content.replace(' NaN', ' np.nan')
        
        # Write back to file if changes were made
        if new_content != content:
            with open(file_path, 'w') as file:
                file.write(new_content)
            print(f"✅ Patched: {file_path}")
            return True
        else:
            print(f"No changes needed for: {file_path}")
            return False
    except Exception as e:
        print(f"Error patching {file_path}: {e}")
        return False

def main():
    # Find pandas_ta path
    pandas_ta_path = find_pandas_ta_path()
    if not pandas_ta_path:
        sys.exit(1)
    
    print(f"Found pandas_ta at: {pandas_ta_path}")
    
    # Files known to have NaN import issues
    problem_files = [
        "momentum/squeeze_pro.py",
        # Add more files here if needed
    ]
    
    # Patch each file
    patched_count = 0
    for rel_path in problem_files:
        full_path = os.path.join(pandas_ta_path, rel_path)
        if patch_file(full_path):
            patched_count += 1
    
    # Also try to find and patch other files that might import NaN
    for root, _, files in os.walk(pandas_ta_path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                # Skip already patched files
                if any(filepath.endswith(rel_path) for rel_path in problem_files):
                    continue
                
                # Check if file contains the problematic import
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if 'from numpy import NaN' in content:
                            if patch_file(filepath):
                                patched_count += 1
                except Exception:
                    continue
    
    print(f"\nPatched {patched_count} files in pandas_ta")
    print("Please restart your application for changes to take effect.")
    
    if patched_count > 0:
        print("\nAlternatively, consider downgrading numpy to a version compatible with pandas_ta:")
        print("pip install numpy==1.23.5")

if __name__ == "__main__":
    main()
