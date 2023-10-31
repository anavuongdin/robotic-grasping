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


files = ['/home/anvd2aic/Desktop/robotic-grasping/in-the-wild-2.jpg']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [[0.9100020527839661, 
            196, 241,
            82.3628,
            16.5811,
            -1.4404297],
            [0.9100020527839661, 
            195, 224,
            80.3628,
            15.5811,
            -1.4404297],
            [0.9100020527839661, 
            195, 210,
            83.3628,
            14.5811,
            -1.4404297],
            [0.9100020527839661, 
            193, 193,
            85.3628,
            15.5811,
            -1.4404297],
             [0.9990,
             270.5548,
             205.6628,
             50.7518,
             20.5932,
             95.0580],
             [0.9990,
             391.5548,
             245.6628,
             23.7518,
             15.5932,
             90.0580],
             ]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
