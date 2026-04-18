import cv2
import math
import numpy as np
import os
import torch
from torch.nn import functional as F
from basicsr.utils import img2tensor, tensor2img

class RealESRGANer():
    def __init__(self,
                 scale,
                 model_path,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if model is None:
             raise ValueError("Model must be provided for minimal vendored version.")
        
        self.model = model
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.model.load_state_dict(loadnet[keyname], strict=True)
        self.model.eval()
        self.model.to(self.device)
        if self.half:
            self.model.half()

    def pre_process(self, img):
        img = img2tensor(img, bgr2rgb=True, float32=True)
        img = img.unsqueeze(0).to(self.device)
        if self.half:
            img = img.half()
        if self.pre_pad != 0:
            img = F.pad(img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        return img

    def process(self):
        self.output = self.model(self.img)

    def post_process(self):
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h, w, _ = img.shape
        self.img = self.pre_process(img)

        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()

        output_img = self.post_process()
        output_img = tensor2img(output_img, rgb2bgr=True, min_max=(0, 1))

        if outscale is not None and outscale != float(self.scale):
            output_img = cv2.resize(output_img, (int(w * outscale), int(h * outscale)), interpolation=cv2.INTER_LANCZOS4)

        return output_img, 'fixed'

    def tile_process(self):
        """Modified tile process for memory efficiency."""
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                size_x = min(self.tile_size, width - ofs_x)
                size_y = min(self.tile_size, height - ofs_y)

                tile_x_start = max(ofs_x - self.tile_pad, 0)
                tile_x_end = min(ofs_x + size_x + self.tile_pad, width)
                tile_y_start = max(ofs_y - self.tile_pad, 0)
                tile_y_end = min(ofs_y + size_y + self.tile_pad, height)
                tile_width = tile_x_end - tile_x_start
                tile_height = tile_y_end - tile_y_start

                input_tile = self.img[:, :, tile_y_start:tile_y_end, tile_x_start:tile_x_end]

                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except Exception as e:
                    print(f"Error in tile processing: {e}")
                    raise e

                output_tile_y_start = (ofs_y - tile_y_start) * self.scale
                output_tile_y_end = output_tile_y_start + size_y * self.scale
                output_tile_x_start = (ofs_x - tile_x_start) * self.scale
                output_tile_x_end = output_tile_x_start + size_x * self.scale

                self.output[:, :, ofs_y * self.scale:(ofs_y + size_y) * self.scale,
                            ofs_x * self.scale:(ofs_x + size_x) * self.scale] = output_tile[:, :, output_tile_y_start:output_tile_y_end,
                                                                                           output_tile_x_start:output_tile_x_end]
