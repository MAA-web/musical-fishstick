import json
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import torch
import numpy as np

# === Config ===
MODEL_DIR = Path("./FLUX-1.dev/transformer")
INDEX_PATH = MODEL_DIR / "diffusion_pytorch_model.safetensors.index.json"
SHARDS_DIR = MODEL_DIR
EXPORT_DIR = MODEL_DIR / "selected_tensors"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# === Load index.json ===
with open(INDEX_PATH, "r") as f:
    index = json.load(f)

weight_map = index["weight_map"]
tensor_keys = sorted(weight_map.keys())

# === Show menu ===
print("📦 Tensors available in the model:\n")
for i, key in enumerate(tensor_keys):
    print(f"[{i}] {key}")
print("\nEnter the index or name of the tensor you want to inspect/export:")

# === Get user selection ===
selection = input("→ ").strip()

# Determine which key was selected
if selection.isdigit():
    selection = int(selection)
    if selection < 0 or selection >= len(tensor_keys):
        print("❌ Invalid index.")
        exit()
    tensor_key = tensor_keys[selection]
else:
    if selection not in tensor_keys:
        print("❌ Tensor name not found.")
        exit()
    tensor_key = selection

# === Load tensor ===
shard_filename = weight_map[tensor_key]
shard_path = SHARDS_DIR / shard_filename

print(f"\n📥 Loading '{tensor_key}' from shard: {shard_filename}")

with safe_open(shard_path, framework="pt", device="cpu") as f:
    tensor = f.get_tensor(tensor_key).cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

# === Show tensor info ===
print(f"\n🧠 Tensor: {tensor_key}")
print(f"• Shape: {list(tensor.shape)}")
print(f"• Dtype: {tensor.dtype}")
print(f"• Preview (flattened): {tensor.flatten()[:10].tolist()}")

# === Ask for export ===
export_choice = input("\nExport this tensor? [s = .safetensors, t = .txt, b = both, n = no]: ").lower().strip()

safe_key = tensor_key.replace('.', '_').replace('/', '_')

if export_choice in {"s", "b"}:
    out_path = EXPORT_DIR / f"{safe_key}.safetensors"
    save_file({tensor_key: tensor}, out_path)
    print(f"✅ Saved as: {out_path}")

if export_choice in {"t", "b"}:
    out_path = EXPORT_DIR / f"{safe_key}.txt"
    with open(out_path, "w") as f_txt:
        f_txt.write(f"# shape: {list(tensor.shape)}\n")
        f_txt.write(f"# dtype: {str(tensor.dtype)}\n")
        np_tensor = tensor.numpy()
        np.savetxt(f_txt, np_tensor.reshape(-1, np_tensor.shape[-1]) if np_tensor.ndim > 1 else np_tensor, fmt="%.9g")
    print(f"✅ Saved as: {out_path}")

if export_choice == "n":
    print("🛑 No file exported.")

print("\n🎯 Done.")
