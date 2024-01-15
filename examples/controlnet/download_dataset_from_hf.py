import os
from shutil import move
from huggingface_hub import hf_hub_download

syncdreamer_ckpt_path = "SyncDreamer/ckpt/"
if not os.path.exists(syncdreamer_ckpt_path):
    os.mkdir(syncdreamer_ckpt_path)
    
filenames = ["renderings-v1-220000-230000.tar.gz", "syncdreamer-pretrain.ckpt", "ViT-L-14.pt"]
for filename in filenames[:1]:
    hf_hub_download(repo_id="jianfuzhang233/controlnet_syncdreamer", filename=filename, repo_type="dataset", local_dir='.')

for filename in filenames[1:]:
    hf_hub_download(repo_id="jianfuzhang233/controlnet_syncdreamer", filename=filename, repo_type="dataset", local_dir=syncdreamer_ckpt_path)
    
print("All files are downloaded!")