
#!/usr/bin/env python3
"""
Test Ray worker import issues
"""

import os
import sys

def test_ray_worker_imports():
    """Test that Ray workers can import our modules"""
    
    print("Testing Ray worker import configuration...")
    
    # Add paths like Ray workers would need
    server_dir = os.path.join(os.getcwd(), 'server')
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    try:
        # Test importing services modules
        from services.communication_types import Message, MessageType
        from services.marl_framework import PheromoneAttentionNetwork
        
        print("✅ SUCCESS: All service modules can be imported")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: Import error - {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error - {e}")
        return False

def test_ray_initialization():
    """Test Ray initialization with proper runtime environment"""
    
    try:
        import ray
        
        # Configure like our updated Ray integration
        runtime_env = {
            "py_modules": [os.path.join(os.getcwd(), 'server')],
            "env_vars": {
                "PYTHONPATH": f"{os.getcwd()}:{os.path.join(os.getcwd(), 'server')}"
            }
        }
        
        if not ray.is_initialized():
            ray.init(
                local_mode=True,
                runtime_env=runtime_env,
                ignore_reinit_error=True
            )
        
        print("✅ SUCCESS: Ray initialized with runtime environment")
        
        # Test a simple Ray task
        @ray.remote
        def test_import_task():
            import sys
            import os
            
            # Paths should be set by runtime_env
            try:
                from services.communication_types import Message
                return {"success": True, "message": "Imports work in Ray worker"}
            except ImportError as e:
                return {"success": False, "error": str(e)}
        
        result = ray.get(test_import_task.remote())
        
        if result["success"]:
            print("✅ SUCCESS: Ray worker can import services modules")
        else:
            print(f"❌ FAILED: Ray worker import failed - {result['error']}")
        
        ray.shutdown()
        return result["success"]
        
    except Exception as e:
        print(f"❌ FAILED: Ray test failed - {e}")
        return False

if __name__ == "__main__":
    print("=== Ray Worker Import Test ===")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python Path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Test 1: Direct imports
    import_success = test_ray_worker_imports()
    
    # Test 2: Ray worker imports
    ray_success = test_ray_initialization()
    
    if import_success and ray_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("Ray workers should now be able to import your services modules.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("The fallback training system will still work.")
    
    print("\nNext steps:")
    print("1. Run this test: python test_ray_worker_imports.py")
    print("2. Try Ray training from the web interface")
    print("3. Check server logs for improved Ray worker status")
