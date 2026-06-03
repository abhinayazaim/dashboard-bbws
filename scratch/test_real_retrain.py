import os
import sys
import time

# Robust sys.path configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT_DIR)

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from dashboard.models import ModelRegistry
from dashboard.ml_engine import MLEngine

def main():
    print("=== MODEL REGISTRY BEFORE TRAINING ===")
    initial_active = None
    for m in ModelRegistry.objects.all().order_by('-training_date'):
        print(f" - Version: {m.version_name} | Active: {m.is_active} | Val Loss: {m.val_loss} | RMSE: {m.rmse}")
        if m.is_active:
            initial_active = m

    engine = MLEngine()
    
    # Force loading of current artifacts to ensure engine is initialized
    print(f"Active model version from MLEngine metadata: {engine.metadata.get('model_version') if engine.metadata else 'None'}")
    
    print("\nTriggering train_candidate_model()...")
    msg = engine.train_candidate_model()
    print("Response message:", msg)
    
    print("Waiting for training thread to finish (polling database)...")
    start_time = time.time()
    timeout = 600  # 10 minutes max for training
    promoted = False
    
    while time.time() - start_time < timeout:
        time.sleep(10)
        # Reload/query registry
        django.db.close_old_connections()
        current_active = ModelRegistry.objects.filter(is_active=True).first()
        
        # Check if the active model has changed
        if current_active:
            if not initial_active or current_active.id != initial_active.id:
                print(f"\n[PROMOTED] New active model detected: {current_active.version_name}")
                print(f" - Val Loss: {current_active.val_loss:.6f} (Initial: {initial_active.val_loss if initial_active else 'None'})")
                print(f" - RMSE: {current_active.rmse:.4f}")
                promoted = True
                break
        
        # Print progress check
        print(".", end="", flush=True)
        
    if not promoted:
        print("\nRetraining completed. No new model was promoted (either validation loss did not improve, or timeout reached).")
        print("\n=== MODEL REGISTRY AFTER RETRAINING ===")
        for m in ModelRegistry.objects.all().order_by('-training_date'):
            print(f" - Version: {m.version_name} | Active: {m.is_active} | Val Loss: {m.val_loss} | RMSE: {m.rmse}")

if __name__ == '__main__':
    main()
