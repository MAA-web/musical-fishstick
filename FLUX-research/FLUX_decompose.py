from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import torch

# Path to your FLUX transformer model directory
MODEL_DIR = Path("./FLUX-1.dev./transformer")

# Locate .safetensors file
tensor_files = list(MODEL_DIR.glob("*.safetensors"))
assert tensor_files, "No .safetensors files found."

tensor_file = tensor_files[0]
print(f"Loading: {tensor_file}")

# Output directory for decomposed safetensors
output_dir = MODEL_DIR / "decomposed_safetensors"
output_dir.mkdir(parents=True, exist_ok=True)

counter = 0

# with safe_open(tensor_file, framework="pt", device="cpu") as f:
#     for key in f.keys():
#         tensor = f.get_tensor(key).cpu()

#         if tensor.dtype == torch.bfloat16:
#             tensor = tensor.to(torch.float32)  # optionally convert for portability

#         safe_key = key.replace('.', '_').replace('/', '__')
#         out_path = output_dir / f"{counter}__{safe_key}.safetensors"

#         # Save as a single-entry safetensors file
#         save_file({key: tensor}, out_path)
#         print(f"✅ Saved: {key} → {out_path.name}")
#         counter += 1

print(f"\n✅ All tensors saved individually in: {output_dir}")
