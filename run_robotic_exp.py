import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='lgrconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--weight', type=str, default='weights/llm/lgrconvnet',
                        help='Network weight in inference mode')
    parser.add_argument('--diffusion', type=bool, default=False,
                        help='Using diffusion model or not')


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    diffusion_flag = args.diffusion

    input_channels = 1 * args.use_depth + 3 * args.use_rgb

    # Setup network
    net = torch.load(args.weight)

    # Setup device
    device = get_device(args.force_cpu)
    net = net.to(device)

    # Sample input
    x = torch.rand(1, 3, 224, 224).cuda()
    prompt = "" # Default
    query = "Grasp me the bottle at its neck"

    if not diffusion_flag:
        pos_pred, cos_pred, sin_pred, width_pred = net(x, prompt, query)
    else:
        alpha = 0.4
        idx = torch.ones(x.shape[0]).to(device)
        pos_pred, cos_pred, sin_pred, width_pred = net(None, x, None, query, alpha, idx)
    print(pos_pred.shape, cos_pred.shape, sin_pred.shape, width_pred.shape)

main()