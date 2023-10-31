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
            243.6877,
            201.1250, 
            71.3628,
            16.5811,
            97.4553],
            [0.9990,
             260.5548,
             204.6628,
             67.7518,
             16.5932,
             97.0580],
            [0.9990,
             172.5548,
             134.6628,
             87.7518,
             24.5932,
             75.0580],
            [0.9990,
             175.5548,
             135.6628,
             90.7518,
             25.5932,
             60.0580],
             [0.9990,
             168.5548,
             136.6628,
             102.7518,
             30.5932,
             110.0580],
             [0.9990,
             274.5548,
             296.6628,
             125.7518,
             40.5932,
             110.0580],
             [0.9100020527839661, 
            80.6877,
            133.1250, 
            51.3628,
            14.5811,
            120.4553],
            [0.9100020527839661, 
            67.6877,
            122.1250, 
            51.3628,
            14.5811,
            120.4553],]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
