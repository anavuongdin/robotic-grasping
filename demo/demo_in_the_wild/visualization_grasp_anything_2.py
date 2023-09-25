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


files = ['demo/demo_in_the_wild/in_the_wild_2.png']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            275.6877,
            261.1250, 
            95.3628,
            24.5811,
            -10.4553],
            [0.9100020527839661, 
            267.6877,
            235.1250, 
            90.3628,
            25.5811,
            -11.4553],
            [0.9100020527839661, 
            100.6877,
            217.1250, 
            30.3628,
            14.5811,
            70.8553],
            [0.9100020527839661, 
            189.6877,
            389.1250, 
            120.3628,
            20.5811,
            1.8553],
            [0.9100020527839661, 
            309.6877,
            346.1250, 
            60.3628,
            12.5811,
            -3.8553],
            [0.9100020527839661, 
            317.6877,
            180.1250, 
            75.3628,
            25.5811,
            4.9553],
            [0.9100020527839661, 
            318.6877,
            207.1250, 
            74.3628,
            26.5811,
            3.9553]]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
