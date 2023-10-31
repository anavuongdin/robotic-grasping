import torch
import numpy as np
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing import image

network_path = 'weights/model_cornell'

rgb_img = image.Image.from_file('in-the-wild-2.jpg')
x = rgb_img
model = torch.load(network_path).cuda()

# Predict the grasp pose using the saved model
with torch.no_grad():
    xc = torch.from_numpy(x.astype(np.float32))
    xc = xc.unsqueeze(0)
    xc = xc.permute(0, 3, 1, 2).cuda()
    pred = model.predict(xc)

q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
grasps = detect_grasps(q_img, ang_img, width_img)

for grasp in grasps:
    print(grasp.center, grasp.angle, grasp.length, grasp.width)