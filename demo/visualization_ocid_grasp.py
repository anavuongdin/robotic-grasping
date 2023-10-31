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

    grasp = [[0.9990,
             170.5548,
             135.6628,
             100.7518,
             25.5932,
             60.0580],
             [0.9990,
             261.5548,
             202.6628,
             50.7518,
             20.5932,
             70.0580],
             [0.9990,
             261.5548,
             202.6628,
             50.7518,
             20.5932,
             100.0580],
             [0.9990,
             105.5548,
             156.6628,
             50.7518,
             20.5932,
             110.0580],
             [0.9990,
             96.5548,
             148.6628,
             52.7518,
             20.5932,
             110.0580],
             ]
    print(grasp)
    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
