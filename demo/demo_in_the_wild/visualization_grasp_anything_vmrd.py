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


files = ['demo/demo_in_the_wild/in_the_wild_vmrd.jpg']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            198.6877,
            282.1250, 
            48.3628,
            24.5811,
            -4.4553],
            [0.9100020527839661, 
            197.6877,
            256.1250, 
            47.3628,
            25.5811,
            -5.4553],
            [0.9100020527839661, 
            178.6877,
            152.1250, 
            145.3628,
            35.5811,
            145.4553],
            [0.9100020527839661, 
            289.6877,
            175.1250, 
            103.3628,
            25.5811,
            143.4553],]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
