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


files = ['demo/demo_in_the_wild/in_the_wild_1.png']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            143.6877,
            157.1250, 
            71.3628,
            16.5811,
            1.4553],
            [0.9100020527839661, 
            142.6877,
            169.1250, 
            70.3628,
            16.5811,
            1.7553],
            [0.9100020527839661, 
            145.6877,
            139.1250, 
            71.3628,
            16.5811,
            1.8553],
            [0.9100020527839661, 
            266.6877,
            156.1250, 
            74.3628,
            18.5811,
            1.9553],
            [0.9100020527839661, 
            264.6877,
            179.1250, 
            73.3628,
            18.5811,
            1.8553],
            [0.9100020527839661, 
            265.6877,
            135.1250, 
            74.3628,
            19.5811,
            1.9553]]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
