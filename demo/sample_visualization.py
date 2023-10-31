import os
import pickle
import numpy as np
import torch
import cv2

def draw_multi_box(img, box_coordinates):
    point_color1 = (255, 255, 0)  # BGR
    point_color2 = (255, 0, 255)  # BGR
    thickness = 2
    lineType = 4
    for i in range(len(box_coordinates)):
        center = (box_coordinates[i, 1].item(), box_coordinates[i, 2].item())
        size = (box_coordinates[i, 3].item(), box_coordinates[i, 4].item())
        angle = box_coordinates[i, 5].item()
        box = cv2.boxPoints((center, size, angle))
        box = np.int64(box)
        cv2.line(img, box[0], box[3], point_color1, thickness, lineType)
        cv2.line(img, box[3], box[2], point_color2, thickness, lineType)
        cv2.line(img, box[2], box[1], point_color1, thickness, lineType)
        cv2.line(img, box[1], box[0], point_color2, thickness, lineType)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sample_dir = 'data/grasp-anything/image'
prompt_dir = 'data/grasp-anything/prompt'
grasp_dir = 'data/grasp-anything/positive_grasp'

files = os.listdir(sample_dir)

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(os.path.join(sample_dir, file))
    handle = file.split('.')[0]
    print(file)
    with open(os.path.join(prompt_dir, handle + '.pkl'), 'rb') as f:
        prompt, queries = pickle.load(f)
        print(prompt, queries)
    all_grasp = []
    count = 0
    for idx, obj in enumerate(queries):
        try:
            with open(os.path.join(grasp_dir, handle + '_{}.pt'.format(idx)), 'rb') as f:
                grasp = torch.load(f)
                grasp = grasp.tolist()
                grasp = [grasp[-1]]
                all_grasp += grasp
            count += 1
        except:
            pass
    
    print(count)
    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
