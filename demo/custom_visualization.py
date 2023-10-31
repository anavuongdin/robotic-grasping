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


sample_dir = 'data/grasp-anything/image'
prompt_dir = 'data/grasp-anything/prompt'
grasp_dir = 'data/grasp-anything/positive_grasp'

files = os.listdir(sample_dir)

chosen_files = np.random.choice(files, 1)
for file in chosen_files:
    file = 'dfd48b31338ed6434f52959e503e9a589fe83a08ad37f41a5d5b33906bf29ebe.jpg'
    img = cv2.imread(os.path.join(sample_dir, file))
    handle = file.split('.')[0]
    print(handle)
    with open(os.path.join(prompt_dir, handle + '.pkl'), 'rb') as f:
        prompt, queries = pickle.load(f)
        print(prompt, queries)
    all_grasp = []
    count = 0
    for idx, obj in enumerate(queries):
        try:
            with open(os.path.join(grasp_dir, handle + '_{}.pt'.format(idx)), 'rb') as f:
                grasp = torch.load(f)
                grasp = grasp.tolist()
                grasp = [grasp[0]]
                # grasp = [[0.031078701838850975, 
                #           240.85089111328125, 
                #           280.5455322265625, 
                #           120.85850524902344, 
                #           34.77340316772461, 
                #           90.470322608947754],
                #           [0.031078701838850975, 
                #           101.85089111328125, 
                #           168.5455322265625, 
                #           128.85850524902344, 
                #           41.77340316772461, 
                #           10.470322608947754]]
                # grasp = [[0.9100020527839661, 
                #           115.07589721679688, 
                #           222.36911010742188, 
                #           100.68780517578125, 
                #           31.977428436279297, 
                #           4.0801944732666],
                #           [4.9100020527839661, 
                #           116.07589721679688, 
                #           76.36911010742188, 
                #           35.68780517578125, 
                #           25.977428436279297, 
                #           86.0801944732666],
                #           [4.9100020527839661, 
                #           355.07589721679688, 
                #           110.36911010742188, 
                #           105.68780517578125, 
                #           45.977428436279297, 
                #           10.0801944732666]]
                # grasp = [[0.9100020527839661, 
                #           170.07589721679688, 
                #           298.36911010742188, 
                #           125.68780517578125, 
                #           41.977428436279297, 
                #           11.0801944732666],
                #           [4.9100020527839661, 
                #           283.07589721679688, 
                #           311.36911010742188, 
                #           85.68780517578125, 
                #           45.977428436279297, 
                #           354.0801944732666],
                #           [4.9100020527839661, 
                #           278.07589721679688, 
                #           209.36911010742188, 
                #           105.68780517578125, 
                #           31.977428436279297, 
                #           4.0801944732666]]
                grasp = [[0.9100020527839661, 
                          83.07589721679688, 
                          187.36911010742188, 
                          115.68780517578125, 
                          45.977428436279297, 
                          15.0801944732666],
                          [4.9100020527839661, 
                          341.07589721679688, 
                          233.36911010742188, 
                          45.68780517578125, 
                          25.977428436279297, 
                          354.0801944732666]]
                print(grasp)
                all_grasp += grasp
            count += 1
        except:
            pass

        grasp = [[0.9100020527839661, 
                79.07589721679688, 
                190.36911010742188, 
                138.68780517578125, 
                45.977428436279297, 
                351.0801944732666],
                [4.9100020527839661, 
                214.07589721679688, 
                141.36911010742188, 
                265.68780517578125, 
                65.977428436279297, 
                20.0801944732666],
                [4.9100020527839661, 
                340.07589721679688, 
                225.36911010742188, 
                105.68780517578125, 
                25.977428436279297, 
                5.0801944732666]]
        print(grasp)
        all_grasp += grasp

    all_grasp = torch.tensor(all_grasp)
    draw_multi_box(img, all_grasp)
