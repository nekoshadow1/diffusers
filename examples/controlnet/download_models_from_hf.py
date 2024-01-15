import os
from shutil import move
from huggingface_hub import hf_hub_download

syncdreamer_ckpt_path = "SyncDreamer/ckpt/"
controlnet_path = 'trained_model/'

if not os.path.exists(syncdreamer_ckpt_path):
    os.mkdir(syncdreamer_ckpt_path)

if not os.path.exists(controlnet_path):
    os.mkdir(controlnet_path)
    
filenames = ["syncdreamer-pretrain.ckpt", "ViT-L-14.pt", 'diffusion_pytorch_model.safetensors', 'config.json']
for filename in filenames[:2]:
    hf_hub_download(repo_id="jianfuzhang233/controlnet_syncdreamer", filename=filename, repo_type="dataset", local_dir=syncdreamer_ckpt_path)
for filename in filenames[2:]:
    hf_hub_download(repo_id="jianfuzhang233/controlnet_syncdreamer", filename=filename, local_dir=controlnet_path)

print("All files are downloaded!")
