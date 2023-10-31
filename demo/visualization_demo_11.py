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


files = ['demo/demo_1.png']

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    img = cv2.imread(file)

    all_grasp = []

    grasp = [
 [0.9999024868011475,
  142.9915771484375,
  179.2072296142578,
  71.79671478271484,
  18.357772827148438,
  162.19931030273438],

 [0.9999604225158691,
  160.90321350097656,
  197.59173583984375,
  71.29438018798828,
  17.661863327026367,
  152.91561889648438],
 [0.9964454770088196,
  172.29425048828125,
  220.1477813720703,
  66.09976959228516,
  17.28848648071289,
  144.6192169189453],
  [0.9975957274436951,
  198.91294860839844,
  259.81231689453125,
  71.59678649902344,
  17.453445434570312,
  158.4920196533203],
 [0.9968709349632263,
  208.15078735351562,
  277.2290344238281,
  67.6906967163086,
  16.730676651000977,
  149.80006408691406],
  [0.9998051524162292,
  183.82037353515625,
  234.48043823242188,
  71.80750274658203,
  17.496152877807617,
  147.93019104003906],
  [0.9987971782684326,
  227.8133544921875,
  299.2830505371094,
  69.871337890625,
  16.954769134521484,
  151.38926696777344],]


    all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
