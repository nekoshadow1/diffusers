import pandas as pd
from huggingface_hub import hf_hub_url
import datasets
import os

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "input_image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "target_index": datasets.Value("uint8"),
        "target_images": datasets.Value("string"),
    },
)

METADATA_URL = '/home/jupyter/data/controlnet_syncdreamer/train.jsonl'
IMAGES_URL = '/home/jupyter/data/controlnet_syncdreamer/'
INPUT_IMAGES_URL = '/home/jupyter/data/controlnet_syncdreamer/'
CONDITIONING_IMAGES_URL = '/home/jupyter/data/controlnet_syncdreamer/'

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class SyncDreamer(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = METADATA_URL
        images_dir = IMAGES_URL
        input_images_dir = INPUT_IMAGES_URL
        conditioning_images_dir = CONDITIONING_IMAGES_URL

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "input_images_dir": input_images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, input_images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            target_index = row["target_index"]

            image_path = row["image"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()
            
            input_image_path = row["input_image"]
            input_image_path = os.path.join(input_images_dir, input_image_path)
            input_image = open(input_image_path, "rb").read()

            target_images_paths = row['target_images']
            
            conditioning_image_path = row["conditioning_image"]
            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["conditioning_image"]
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["image"], {
                "target_index": target_index,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "input_image": {
                    "path": input_image_path,
                    "bytes": input_image,
                },
                "target_images": target_images_paths,
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }