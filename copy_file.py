import os
import shutil
import argparse

import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('folder', type=str, help='Input string to process')
    args = parser.parse_args()

    src_folder_path = "/LOCAL2/anguyen/faic/vdan/grasping/robotic-grasping/data/grasp-anything-old/{}".format(args.folder)
    dest_folder_path = "/LOCAL2/anguyen/faic/vdan/grasping/robotic-grasping/data/grasp-anything++/{}".format(args.folder)

    with open('need.txt', 'r') as f:
        needed_files_str = f.readlines()
        needed_files = []
        for file in needed_files_str:
            needed_files.append(file.strip('\n'))
    
    copy_files_with_prefix(src_folder_path, dest_folder_path, needed_files)


def copy_files_with_prefix(src_folder, dest_folder, idx_list):
    for filename in os.listdir(src_folder):
        # Extract idx from the filename
        idx = filename.split('_')[0]

        # Check if the idx is in the list of interested idxs
        if idx in idx_list:
            # Construct source and destination paths
            src_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename)

            # Copy the file to the destination folder
            shutil.copyfile(src_path, dest_path)
            print(f"Copied {filename} to {dest_folder}")

if __name__ == '__main__':
    main()