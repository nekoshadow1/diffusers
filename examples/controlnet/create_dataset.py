import argparse
from tqdm import tqdm
import os
from shutil import copy

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
        
        target_images = []
        for i in range(16):
            src_img_name = str(i).zfill(3) + ".png"
            dst_img_name = object_id + "_" + str(i).zfill(3) + ".png"
            target_images.append("images/" + dst_img_name)
        target_images = str(target_images)
        
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
            line = '"image": "{}", "conditioning_image": "{}", "target_index":{}, "target_images": "{}"'.format('images/' + dst_img_name, cond_img_path, i, target_images)
            line = '{' + line + '}\n'
            file.write(line)
            
file.close()