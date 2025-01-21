import argparse
import logging
import os
from os import listdir

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.edgedata_loading import EdgeDataset as BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from pathlib import Path



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = full_img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    print(img.shape)

    with torch.no_grad():
        output = net(img).cpu()
        #output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()



def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


import time


if __name__ == '__main__':
    #for image in os.listdir(folder):
    model = 'checkpoints/checkpoint_epoch1.pth'
    output_file = 'predict_output/output2.jpg'
    scale = 0.5
    mask_threshold = 0.5

    net = UNet(n_channels=4, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    crackdir_img = Path('./data/crack/imgs')
    crackdir_mask = Path('./data/crack/masks')
    crackdir_edge = Path('data/crack/edges')
    img_scale = 0.5

    dataset = BasicDataset(crackdir_img, crackdir_mask, crackdir_edge, img_scale)

    


    data = dataset[0]
    img = data['image']
    name = data['name']

    print(name)

    mask = predict_img(net=net,
                    full_img=img,
                    scale_factor=scale,
                    out_threshold=mask_threshold,
                    device=device)


    out_filename = output_file
    result = mask_to_image(mask, mask_values)
    result.save(out_filename)
    logging.info(f'Mask saved to {out_filename}')

    
