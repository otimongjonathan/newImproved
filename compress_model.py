import torch
import os

model_path = 'models/best_model.pth'
compressed_path = 'models/best_model_compressed.pth'

print(f"Loading model from {model_path}")
model_data = torch.load(model_path, map_location='cpu')

print(f"Compressing and saving to {compressed_path}")
torch.save(model_data, compressed_path, _use_new_zipfile_serialization=True)

original = os.path.getsize(model_path) / (1024**2)
compressed = os.path.getsize(compressed_path) / (1024**2)

print(f"Original: {original:.2f} MB")
print(f"Compressed: {compressed:.2f} MB")
print(f"Saved: {original - compressed:.2f} MB ({((original-compressed)/original*100):.1f}%)")

# Replace original with compressed
os.remove(model_path)
os.rename(compressed_path, model_path)
print("Model compressed successfully!")
