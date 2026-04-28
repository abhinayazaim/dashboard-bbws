import os, json, zipfile, shutil
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import h5py

# Extract to read weights
temp_dir = 'models/_temp_inspect'
with zipfile.ZipFile('models/best_model.keras', 'r') as z:
    z.extractall(temp_dir)

weights_path = os.path.join(temp_dir, 'model.weights.h5')

def print_h5(f, prefix=''):
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            print_h5(item, prefix + '  ')
        else:
            print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")

with h5py.File(weights_path, 'r') as f:
    print("=== H5 Weight Structure ===")
    print_h5(f)

shutil.rmtree(temp_dir, ignore_errors=True)
