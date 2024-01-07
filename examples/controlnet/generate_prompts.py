import argparse
import random
from tqdm import tqdm
import os
from shutil import copy

# Prompt template
templates = [
'Adjust the image orientation by 30 degrees.',
'Turn the image clockwise by 30 degrees.',
'Rotate the picture 30 degrees to the right.',
'Apply a 30-degree clockwise rotation to the image.',
'Change the image angle by 30 degrees.',
'Shift the image orientation 30 degrees to the right.',
'Spin the image 30 degrees in a clockwise direction.',
'Tilt the image by 30 degrees.',
'Perform a 30-degree rotation on the image.',
'Give the image a 30-degree clockwise twist.',
'Swing the image to the right by 30 degrees.',
'Twist the picture 30 degrees in a clockwise manner.',
'Modify the image orientation with a 30-degree turn.',
'Adjust the image angle by 30 degrees to the right.',
'Execute a 30-degree clockwise rotation on the image.',
'Tweak the image positioning with a 30-degree turn.',
'Pivot the image by 30 degrees.',
'Apply a 30-degree rotation in a clockwise fashion to the image.',
'Alter the image orientation by 30 degrees to the right.',
'Give the image a 30-degree clockwise spin.',
]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to create training data for controlnet+syncdreamer.")
    parser.add_argument(
        "--DATA_PATH",
        type=str,
        default=None,
        required=True,
        help="Path to unpreprocessed data.",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

def generate_prompt(templates, degree):
    return random.choice(templates).replace('30', str(degree))

args = parse_args()

ROOT_PATH = './renderings-v1/'
object_ids = os.listdir(ROOT_PATH)

with open(os.path.join(args.DATA_PATH, 'train.jsonl'), 'w') as file:
    for j in tqdm(range(len(object_ids))):
        object_id = object_ids[j]
        cond_img_name = object_id + '_000.png'
        img_path = ROOT_PATH + object_id + '/'
        
        # copy conditioning images
        src = ROOT_PATH + object_id + '/000.png'
        dst = args.DATA_PATH + 'conditioning_images/' + cond_img_name
        dst_path = args.DATA_PATH + 'conditioning_images/'
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        copy(src, dst)
        
        cond_img_path = 'conditioning_images/' + cond_img_name
        
        for i in range(16):
            src_img_name = str(i).zfill(3) + '.png'
            dst_img_name = object_id + '_' + str(i).zfill(3) + '.png'
            
            # copy target images
            src = ROOT_PATH + object_id + '/' + src_img_name
            dst = args.DATA_PATH + 'images/' + dst_img_name
            dst_path = args.DATA_PATH + 'images/'
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            copy(src, dst)
            
            degree = i * 22.5
            prompt = generate_prompt(templates, degree)
            line = '"text": "{}", "image": "{}", "conditioning_image": "{}", "target_index":{}'.format(prompt, 'images/' + dst_img_name, cond_img_path, i)
            line = '{' + line + '}\n'
            file.write(line)
            
file.close()