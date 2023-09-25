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


files = ['demo/demo_in_the_wild/in_the_wild_4.png']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            377.6877,
            277.1250, 
            16.3628,
            10.5811,
            1.4553],
            [0.9100020527839661, 
            334.6877,
            308.1250, 
            26.3628,
            12.5811,
            1.4553],
            [0.9100020527839661, 
            335.6877,
            293.1250, 
            26.3628,
            12.5811,
            1.4553],
            [0.9100020527839661, 
            33.6877,
            261.1250, 
            45.3628,
            22.5811,
            -1.4553],
            [0.9100020527839661, 
            32.6877,
            197.1250, 
            45.3628,
            22.5811,
            -1.4553],
            [0.9100020527839661, 
            30.6877,
            136.1250, 
            45.3628,
            22.5811,
            -1.4553],
            [0.9100020527839661, 
            169.6877,
            63.1250, 
            25.3628,
            11.5811,
            90.4553],
            [0.9100020527839661, 
            155.6877,
            63.1250, 
            24.3628,
            12.5811,
            91.4553],
            [0.9100020527839661, 
            310.6877,
            102.1250, 
            78.3628,
            32.5811,
            1.4553],
            [0.9100020527839661, 
            135.6877,
            210.1250, 
            110.3628,
            40.5811,
            -1.4553],
            [0.9100020527839661, 
            234.6877,
            352.1250, 
            48.3628,
            12.5811,
            120.4553],
            [0.9100020527839661, 
            288.6877,
            354.1250, 
            38.3628,
            10.5811,
            125.4553],
            
            [0.9100020527839661, 
            36.6877,
            354.1250, 
            34.3628,
            9.5811,
            118.4553],]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
