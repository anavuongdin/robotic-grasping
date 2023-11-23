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


files = ['demo/in_the_wild_cvpr_exp/itw_4.jpeg']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            203.6877,
            94.1250, 
            75.3628,
            20.5811,
            1.4553],
            [0.9100020527839661, 
            37.6877,
            149.1250, 
            80.3628,
            25.5811,
            -20.4553],
            [0.9100020527839661, 
            162.6877,
            283.1250, 
            40.3628,
            15.5811,
            40.4553],
            [0.9100020527839661, 
            306.6877,
            332.1250, 
            20.3628,
            5.5811,
            130.4553],
            [0.9100020527839661, 
            328.6877,
            237.1250, 
            30.3628,
            12.5811,
            80.4553],
            [0.9100020527839661, 
            146.6877,
            334.1250, 
            30.3628,
            12.5811,
            120.4553],
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
