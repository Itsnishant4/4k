import torch
import cv2
import numpy as np
from PIL import Image
import os
import requests
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class ImageUpscaler:
    def __init__(self, model_name='RealESRGAN_x4plus', device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.upsampler = None
        self.load_model()

    def load_model(self):
        if self.model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
        elif self.model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            model_path = os.path.join('weights', 'RealESRGAN_x2plus.pth')
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Download weight if not exists
        if not os.path.exists(model_path):
            os.makedirs('weights', exist_ok=True)
            self.download_weight(self.model_name, model_path)

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=400, # Tiling for 4K stability
            tile_pad=10,
            pre_pad=0,
            half=True if self.device.type != 'cpu' else False, # Use half precision on GPU
            device=self.device
        )

    def download_weight(self, model_name, path):
        print(f"Downloading {model_name}...")
        urls = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        }
        url = urls.get(model_name)
        response = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    def upscale(self, pil_image):
        # Convert PIL to CV2
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Inference
        output, _ = self.upsampler.enhance(img, outscale=4)
        
        # Convert back to PIL
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output)

# Singleton instance
_upscaler_instance = None
def get_upscaler():
    global _upscaler_instance
    if _upscaler_instance is None:
        _upscaler_instance = ImageUpscaler()
    return _upscaler_instance
