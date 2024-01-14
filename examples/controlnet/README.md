# ControlNet + SyncDreamer training example

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

This example is based on the [training example in the original ControlNet repository](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md). It trains a ControlNet to fill circles using a [small synthetic dataset](https://huggingface.co/datasets/fusing/fill50k).

## System requirements

CUDA 11.3

conda

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
conda create -n controlnet python=3.8
conda activate controlnet
git clone https://github.com/nekoshadow1/diffusers.git
cd diffusers
pip install -e .
```

Then cd in the example folder and run

```bash
cd examples/controlnet
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

## Download training dataset and pretrained SyncDreamer weights

I reused the training dataset of Syncdreamer. Due to hardware and time limit, I only utilized the smallest split (renderings-v1-220000-230000.tar.gz) of the training data. You may download the same part as me using the following commands:

If the download script does not work, it is because HF limits large file downloads probably. You can download from [here](https://connecthkuhk-my.sharepoint.com/personal/yuanly_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSyncDreamerData&ga=1) instead.
Also download `ViT-L-14.ckpt` and `syncdreamer-pretrain.ckpt` [here](https://connecthkuhk-my.sharepoint.com/personal/yuanly_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSyncDreamer&ga=1) and put the model files in `SyncDreamer/ckpt/`. 

```bash
#If the script does not work, you need to download the data file from: 
python download_from_hf.py
tar -xf renderings-v1-220000-230000.tar.gz

mkdir '/home/jupyter/data/'
export DATA_PATH='/home/jupyter/data/syncdreamer/'
mkdir $DATA_PATH
python create_dataset.py --DATA_PATH=$DATA_PATH

rm renderings-v1-220000-230000.tar.gz
sudo rm -r renderings-v1
```

You may download other splits [here](https://connecthkuhk-my.sharepoint.com/personal/yuanly_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSyncDreamerData&ga=1).

(Important) Also remember to put the following script (syncdreamer.py) in DATA_PATH, and modify three variables `METADATA_URL`, `IMAGES_URL`, `CONDITIONING_IMAGES_URL` in line 23-25 accordingly:

```bash
cp syncdreamer.py $DATA_PATH
```

## Training

```bash
mkdir models_syncdreamer
export DATA_PATH='/home/jupyter/data/syncdreamer/'
export OUTPUT_DIR="/home/jupyter/diffusers/examples/controlnet/models_syncdreamer"

accelerate launch train_controlnet_syncdreamer.py \
  --output_dir=$OUTPUT_DIR \
  --train_data_dir=$DATA_PATH \
  --resolution=256 \
  --learning_rate=1e-5 \
  --train_batch_size=4
```

## Inference