import json
import subprocess
import random
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Read arguments from JSON file
    # output_dir = '/home/jupyter/data/controlnet_syncdreamer_dataset/offset'
    output_dir = args.output_dir
    
    with open('/home/jupyter/dev/objects.json', 'r') as file:
        data = json.load(file)
    file.close()

    # Loop through the arguments and run the main Python script
    for object_uid in list(data.keys())[:10]:
        object_path = data[object_uid]
        offset = args.output_dir.split('/')[-1]
        print('offset')
        if offset == 'offset':
            y_offset = random_number = round(random.uniform(-0.3, 0.3), 2)
            z_offset = random_number = round(random.uniform(-0.3, 0.3), 2)
        else:
            y_offset = 0
            z_offset = 0
        
        subprocess.run(['blender', '--background', '--python', '/home/jupyter/dev/SyncDreamer/blender_script.py', '--', 
                        '--object_path', object_path, '--output_dir', args.output_dir, '--camera_type', 'fixed', '--y_offset', str(y_offset), '--z_offset', str(z_offset)])

if __name__ == "__main__":
    main()