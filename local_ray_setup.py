#!/usr/bin/env python3
"""
Local environment Ray setup script
This script attempts to install and configure Ray for your local development environment
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_ray_installation():
    """Check current Ray installation status"""
    print("\n=== Checking Ray Installation ===")
    
    # Check if Ray is installed
    try:
        import ray
        print(f"✅ Ray is installed: version {ray.__version__ if hasattr(ray, '__version__') else 'unknown'}")
        
        # Try to initialize Ray
        try:
            if not ray.is_initialized():
                ray.init(local_mode=True)
            print("✅ Ray can be initialized successfully")
            ray.shutdown()
            return True
        except Exception as e:
            print(f"❌ Ray cannot be initialized: {e}")
            return False
            
    except ImportError:
        print("❌ Ray is not installed")
        return False

def install_ray():
    """Install Ray with RLlib"""
    print("\n=== Installing Ray ===")
    
    commands = [
        ("pip uninstall -y ray", "Removing existing Ray installation"),
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install 'ray[rllib]==2.9.0'", "Installing Ray 2.9.0 with RLlib"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True

def test_ray_functionality():
    """Test Ray functionality with our training system"""
    print("\n=== Testing Ray Functionality ===")
    
    try:
        # Add current directory to path
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        
        from server.services.ray_fallback import RayFallbackSystem
        
        fallback = RayFallbackSystem()
        
        print(f"Ray Available: {fallback.ray_available}")
        if fallback.ray_error:
            print(f"Ray Error: {fallback.ray_error}")
            
        if fallback.ray_available:
            print("✅ Ray is working with our training system!")
            
            # Test a quick training run
            test_config = {
                'experiment_id': 888,
                'experiment_name': 'Ray Setup Test',
                'total_episodes': 2,
                'use_ray': True
            }
            
            print("🔄 Running quick Ray training test...")
            result = fallback.start_training(test_config)
            
            if result.get('success') and result.get('ray_available'):
                print("✅ Ray training test successful!")
                return True
            else:
                print(f"❌ Ray training test failed: {result.get('message', 'Unknown error')}")
                return False
        else:
            print("❌ Ray is not available for our training system")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ray functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup process"""
    print("=== Ray Local Environment Setup ===")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Step 1: Check current installation
    if check_ray_installation():
        print("\n✅ Ray is already working! No setup needed.")
        return True
    
    # Step 2: Install Ray
    if not install_ray():
        print("\n❌ Ray installation failed. Check error messages above.")
        return False
    
    # Step 3: Test functionality
    if test_ray_functionality():
        print("\n🎉 SUCCESS: Ray is now set up and working with your training system!")
        print("\nNext steps:")
        print("1. Restart your web application")
        print("2. Try Ray training from the web interface")
        print("3. Check server logs for 'Ray RLlib is available' message")
        return True
    else:
        print("\n⚠️  Ray installed but not fully functional.")
        print("The fallback training system will still work perfectly.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)