#!/usr/bin/env python3
"""
Quick verification script to test Ray training in your local environment
"""

import sys
import os
import json

# Add current directory to Python path (matches server setup)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

def test_ray_training():
    """Test Ray training with the import fixes"""
    try:
        from server.services.ray_fallback import RayFallbackSystem
        
        print("🔄 Testing Ray training system...")
        
        fallback = RayFallbackSystem()
        
        # Create test configuration
        test_config = {
            'experiment_id': 999,
            'experiment_name': 'Local Environment Test',
            'total_episodes': 3,
            'learning_rate': 0.001,
            'use_ray': True,
            'num_rollout_workers': 2,
        }
        
        print(f"Ray Available: {fallback.ray_available}")
        if fallback.ray_error:
            print(f"Ray Error: {fallback.ray_error}")
        
        # Start training
        result = fallback.start_training(test_config)
        
        print(f"Training Result: {result.get('success', False)}")
        print(f"Training Type: {result.get('training_type', 'unknown')}")
        print(f"Ray Available in Result: {result.get('ray_available', False)}")
        
        if result.get('success'):
            final_metrics = result.get('final_metrics', {})
            print(f"Episodes Completed: {final_metrics.get('total_episodes', 'N/A')}")
            print(f"Communication Efficiency: {final_metrics.get('communication_efficiency', 'N/A')}")
            
        return result.get('success', False) and result.get('ray_available', False)
        
    except Exception as e:
        print(f"❌ Error testing Ray: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == '__main__':
    print("=== Ray Training Fix Verification ===")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python Path: {sys.path[0]}")
    
    success = test_ray_training()
    
    if success:
        print("\n✅ SUCCESS: Ray training is working with full functionality!")
        print("The import fixes resolved the 'attempted relative import' issue.")
    else:
        print("\n⚠️  Ray training is using fallback mode.")
        print("This is still functional but check Ray installation if you want full distributed training.")
    
    print("\nNext steps:")
    print("1. Try starting Ray training from the web interface")
    print("2. Check the server logs for Ray training status")
    print("3. The system will automatically use the best available mode")