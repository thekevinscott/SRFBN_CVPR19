import argparse, time, os
from networks import create_model, define_net
import numpy as np
import torch

import imageio
from data import common

import options.options as option
from utils import util
from solvers import SRSolver
from data import create_dataloader
from data import create_dataset


networks= {
    "which_model": "SRFBN",
    "num_features": 64,
    "in_channels": 3,
    "out_channels": 3,
    "num_steps": 4,
    "num_groups": 6
    }

def run(pretrained_path, output_path, model_name='SRFBN', scale=4, degrad='BI', opt='options/test/test_SRFBN_example.json'):
    opt = option.parse(opt)
    opt = option.dict_to_nonedict(opt)
    # model = create_model(opt)
    model = define_net({
        "scale": scale,
        "which_model": "SRFBN",
        "num_features": 64,
        "in_channels": 3,
        "out_channels": 3,
        "num_steps": 4,
        "num_groups": 6
    })

    img = common.read_img('./results/LR/MyImage/chip.png', 'img')

    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    lr_tensor = torch.unsqueeze(tensor, 0)

    checkpoint = torch.load(pretrained_path)
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    load_func = model.load_state_dict
    load_func(checkpoint)
    torch.save(model, './model.pt')


    with torch.no_grad():
        SR = model(lr_tensor)[0]
    # visuals = np.transpose(SR.data[0].float().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    visuals = np.transpose(SR.data[0].float().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    imageio.imwrite(output_path, visuals)
