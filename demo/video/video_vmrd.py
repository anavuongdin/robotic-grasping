import os
import pickle
import numpy as np
import torch
import cv2


def convert_to_list(script: str):
    grasps = script.strip('\n').split('|')
    grasps = list(map(lambda x: x.split(','), grasps))
    grasps = list(map(lambda grasp: list(map(float, grasp)), grasps))
    return grasps


def random_grasp(grasp, adj):
    _, x, y, w, h, angle = grasp

    x -= np.random.random() - 0.5
    y -= np.random.random() - 0.5
    w += np.random.random() - 0.5
    h += np.random.random() - 0.5
    angle += (adj + np.random.random())/10

    return [_, x, y, w, h, angle]

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
    return img


dataset = 'vmrd'
path = '/home/anvd2aic/Desktop/supplementary_video/IMG-1533'
script_file = 'demo/video/config/{}.txt'.format(dataset)
save_path = '/home/anvd2aic/Desktop/supplementary_video/{}'.format(dataset)
images = sorted(os.listdir(path))

default_grasp = [[0.97, 83.78, 85.01, 72.63, 18.55, 23.16]]
diff = 15
with open(script_file, 'r') as f:
    scripts = f.readlines()

for idx, image_file in enumerate(images):
    image_file = os.path.join(path, image_file)
    img = cv2.imread(image_file)

    counter = idx // diff
    adjust = idx % diff
    grasp = convert_to_list(scripts[counter])

    grasp = list(map(lambda x: random_grasp(x, adjust), grasp))
    grasp = torch.tensor(grasp)

    img = draw_multi_box(img, grasp)
    cv2.imwrite(os.path.join(save_path, str(idx) + '.jpg'), img)


    if idx % diff == 0:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()