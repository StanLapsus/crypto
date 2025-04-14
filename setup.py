#!/usr/bin/env python3
import sys
import subprocess
import os
import time
import re

def print_step(message):
    """Print a step message with formatting"""
    print(f"\n{'='*80}\n{message}\n{'='*80}")

def run_command(command, check=True):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def check_module_installed(module_name):
    """Check if a Python module is installed"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def patch_pandas_ta():
    """Patch pandas_ta library to fix NaN import issue"""
    print_step("Patching pandas_ta library")
    
    try:
        import pandas_ta
        pandas_ta_path = os.path.dirname(pandas_ta.__file__)
        print(f"Found pandas_ta at: {pandas_ta_path}")
        
        # Check for squeeze_pro.py which has the NaN import issue
        squeeze_pro_path = os.path.join(pandas_ta_path, "momentum", "squeeze_pro.py")
        if os.path.exists(squeeze_pro_path):
            with open(squeeze_pro_path, 'r') as f:
                content = f.read()
                
            if 'from numpy import NaN' in content:
                print(f"Found problematic import in {squeeze_pro_path}")
                
                # Replace the import
                content = content.replace('from numpy import NaN', 'import numpy as np')
                content = content.replace(' NaN', ' np.nan')
                
                with open(squeeze_pro_path, 'w') as f:
                    f.write(content)
                    
                print("✅ Successfully patched pandas_ta")
                return True
        
        print("No issues found in pandas_ta or patch already applied")
        return True
        
    except Exception as e:
        print(f"Error patching pandas_ta: {e}")
        return False

def main():
    print_step("Crypto Trading System Setup and Diagnostics")
    
    # Check Python version
    print_step("Checking Python version")
    print(f"Python version: {sys.version}")
    
    # Install requirements
    print_step("Installing requirements")
    run_command("pip install -r requirements.txt")
    
    # Test critical imports
    print_step("Testing critical imports")
    critical_modules = ['polars', 'numpy', 'requests', 'matplotlib']  # Updated list - removed pandas_ta, pandas, vaex
    
    all_modules_ok = True
    for module in critical_modules:
        if check_module_installed(module):
            print(f"✅ {module} is installed")
        else:
            print(f"❌ {module} is NOT installed")
            all_modules_ok = False
            
            # Try to install it directly
            print(f"  Attempting to install {module}...")
            if run_command(f"pip install {module}", check=False):
                if check_module_installed(module):
                    print(f"  ✅ {module} successfully installed")
                    all_modules_ok = True
                else:
                    print(f"  ❌ Failed to install {module}")
    
    # Test API connectivity
    print_step("Testing API connectivity")
    try:
        import requests
        
        # Test CoinGecko API
        print("Testing CoinGecko API...")
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            print("✅ CoinGecko API is accessible")
        else:
            print(f"❌ CoinGecko API returned status code {response.status_code}")
            
        # Test other APIs
        print("Testing Binance API...")
        response = requests.get("https://api.binance.com/api/v3/time", timeout=10)
        if response.status_code == 200:
            print("✅ Binance API is accessible")
        else:
            print(f"❌ Binance API returned status code {response.status_code}")
            
    except Exception as e:
        print(f"Error testing API connectivity: {e}")
    
    # Create necessary directories
    print_step("Creating necessary directories")
    directories = ['data', 'logs', 'models', 'data/feedback', 'data/api_cache']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    # Create a minimal test run
    print_step("Running a minimal test")
    try:
        print("Importing and instantiating CryptoAPIUtils...")
        from api_utils import CryptoAPIUtils
        api = CryptoAPIUtils()
        
        print("Testing price history fetch...")
        result = api.get_crypto_price_history('BTC/USD', days=1, interval='hourly')
        if result and 'prices' in result:
            print(f"✅ Successfully fetched {len(result['prices'])} price points from {result['source']}")
            
            # Test polars dataframe creation
            import polars as pl
            prices = result['prices']
            df = pl.DataFrame({
                'timestamp': [p[0] for p in prices],
                'price': [p[1] for p in prices]
            })
            df = df.with_column(pl.col('timestamp').cast(pl.Datetime).dt.with_time_unit('ms'))
            print(f"✅ Successfully created Polars DataFrame with {df.height} rows")
        else:
            print("❌ Failed to fetch price history")
            
    except Exception as e:
        print(f"Error in minimal test: {e}")
        import traceback
        traceback.print_exc()
    
    # Final status
    print_step("Setup Complete")
    if all_modules_ok:
        print("✅ All critical modules are installed")
        print("You can now run the system with: python main.py")
    else:
        print("⚠️ Some modules may not be installed correctly")
        print("Please fix the issues above before running the system")

if __name__ == "__main__":
    main()
