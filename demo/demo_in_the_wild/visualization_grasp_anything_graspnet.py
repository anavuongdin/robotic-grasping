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


files = ['demo/demo_in_the_wild/in_the_wild_graspnet.png']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            198.6877,
            155.1250, 
            35.3628,
            10.5811,
            -30.4553],
            [0.9100020527839661, 
            238.6877,
            153.1250, 
            30.3628,
            10.5811,
            1.4553],
            [0.9100020527839661, 
            216.6877,
            221.1250, 
            10.3628,
            5.5811,
            40.4553],
            [0.9100020527839661, 
            252.6877,
            225.1250, 
            38.3628,
            13.5811,
            4.4553],
            [0.9100020527839661, 
            318.6877,
            197.1250, 
            70.3628,
            30.5811,
            140.4553],
            [0.9100020527839661, 
            279.6877,
            152.1250, 
            29.3628,
            9.5811,
            -5.4553],]
    
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
