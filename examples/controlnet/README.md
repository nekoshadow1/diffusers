# ControlNet + SyncDreamer

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

[SyncDreamer: Generating Multiview-consistent Images from a Single-view Image](https://arxiv.org/abs/2309.03453) by Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang.

This example is based on the [training example in the original ControlNet repository](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md).


## [Live Demo](https://huggingface.co/spaces/jianfuzhang233/controlnet)

## System requirements

Linux

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

If the download script fails (the downloaded files are extremely small), it is because HF limits large file downloads probably. You can download renderings-v1-220000-230000.tar.gz [here](https://huggingface.co/datasets/jianfuzhang233/controlnet_syncdreamer/tree/main) instead.
Also download `ViT-L-14.ckpt` and `syncdreamer-pretrain.ckpt` [here](https://huggingface.co/datasets/jianfuzhang233/controlnet_syncdreamer/tree/main) and put the model files in `SyncDreamer/ckpt/`.

```bash
python download_dataset_from_hf.py
tar -xf renderings-v1-220000-230000.tar.gz

mkdir '/home/jupyter/data/'
export DATA_PATH='/home/jupyter/data/syncdreamer/'
mkdir $DATA_PATH
python create_dataset.py --DATA_PATH=$DATA_PATH

rm renderings-v1-220000-230000.tar.gz
sudo rm -r renderings-v1
```

You may download other splits [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EqrCp4rcFOFBuCatr88bkL0Bl3qKIEU1gSPS7TQ2KGb7Yg?e=eSESKA).

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

You can download my trained ControlNet [here](https://huggingface.co/jianfuzhang233/controlnet_syncdreamer/tree/main) or run the following commands.

If the download script fails (the downloaded files are extremely small), it is because HF limits large file downloads probably. You can download diffusion_pytorch_model.safetensors and config.json [here](https://huggingface.co/jianfuzhang233/controlnet_syncdreamer) instead, and put them in `trained_model`. Also download `ViT-L-14.ckpt` and `syncdreamer-pretrain.ckpt` [here](https://huggingface.co/datasets/jianfuzhang233/controlnet_syncdreamer/tree/main) and put them in `SyncDreamer/ckpt/`.

```bash
mkdir trained_model
export MODEL_PATH='/home/jupyter/diffusers/examples/controlnet/trained_model/'
python download_models_from_hf.py
```

You can run the inference in inference.ipynb or run the following script.

```bash
python inference.py --MODEL_PATH trained_model --INPUT_PATH testset/aircraft.png --AZIMUTH 90
```

## Citation
If you find this repository useful in your project, please cite the following papers. :)
```
@article{liu2023syncdreamer,
  title={SyncDreamer: Generating Multiview-consistent Images from a Single-view Image},
  author={Liu, Yuan and Lin, Cheng and Zeng, Zijiao and Long, Xiaoxiao and Liu, Lingjie and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2309.03453},
  year={2023}
}
@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
  author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
  booktitle={IEEE International Conference on Computer Vision (ICCV)}
  year={2023},
}
```