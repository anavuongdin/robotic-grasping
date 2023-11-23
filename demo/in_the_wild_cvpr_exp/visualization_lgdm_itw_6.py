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


files = ['demo/in_the_wild_cvpr_exp/itw_6.jpg']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            58.6877,
            151.1250, 
            20.3628,
            10.5811,
            -20.4553],
            [0.9100020527839661, 
            39.6877,
            157.1250, 
            20.3628,
            10.5811,
            -15.4553],
            [0.9100020527839661, 
            89.6877,
            302.1250, 
            50.3628,
            12.5811,
            -15.4553],
            [0.9100020527839661, 
            327.6877,
            258.1250, 
            45.3628,
            15.5811,
            15.4553],
            [0.9100020527839661, 
            283.6877,
            81.1250, 
            20.3628,
            10.5811,
            145.4553],
            [0.9100020527839661, 
            268.6877,
            161.1250, 
            22.3628,
            12.5811,
            151.4553],
            # [0.9100020527839661, 
            # 200.6877,
            # 129.1250, 
            # 165.3628,
            # 55.5811,
            # 7.4553],
            # [0.9100020527839661, 
            # 202.6877,
            # 138.1250, 
            # 165.3628,
            # 55.5811,
            # 3.4553],
            ]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
