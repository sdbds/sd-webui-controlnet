import os
import torch
import torch.nn.functional as F
import numpy as np

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from einops import rearrange
from modules import devices
from annotator.annotator_path import models_path
import torchvision.transforms as transforms
import metric3d.mono.configs as configs
from metric3d.mono.model.monodepth_model import get_configured_monodepth_model
from scripts.utils import resize_image_with_pad


class Metric3DDetector:
    model_dir = os.path.join(models_path, "metric3d")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth"
        modelpath = os.path.join(self.model_dir, "metric_depth_vit_giant2_800k.pth")
        if not os.path.exists(modelpath):
            from scripts.utils import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        cfg = Config.fromfile(configs.__path__[0] + '/HourglassDecoder/vit.raft5.large.py')
        model = get_configured_monodepth_model(cfg)
        model.eval()
        self.model = model.to(self.device)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image, new_fov=60.0, resulotion=512):
        if self.model is None:
            self.load_model()

        self.model.to(self.device)
        intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
        orig_H, orig_W = input_image.shape[:2]
        input_size = (616, 1064)
        scale = min(input_size[0] / orig_H, input_size[1] / orig_W)
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        
        input_image, remove_pad = resize_image_with_pad(input_image, resulotion)
        
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(self.device)
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            image_depth = self.norm(image_depth)
            
            depth = self.model(image_depth)

            depth = ((depth + 1) * 0.5).clip(0, 1)

            depth = rearrange(depth[0], 'c h w -> h w c').cpu().numpy()
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return remove_pad(depth_image)