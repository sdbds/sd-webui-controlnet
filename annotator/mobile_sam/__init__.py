from __future__ import print_function

import os
import numpy as np
from PIL import Image
from typing import Union

from modules import devices
from annotator.util import load_model
from annotator.annotator_path import models_path

from controlnet_aux import SamDetector
from controlnet_aux.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SamDetector_Aux(SamDetector):

    model_dir = os.path.join(models_path, "mobile_sam")

    def __init__(self, mask_generator: SamAutomaticMaskGenerator):
        super().__init__(mask_generator)

        self.device = devices.device
        self.model = SamDetector_Aux().to(self.device).eval()
        self.from_pretrained(model_type="vit_t")

    @classmethod
    def from_pretrained(cls, model_type="vit_t"):
        """
        Possible model_type : vit_h, vit_l, vit_b, vit_t
        download weights from https://huggingface.co/dhkim2810/MobileSAM
        """
        remote_url = os.environ.get(
            "CONTROLNET_MOBILE_SAM_MODEL_URL",
            "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt",
        )
        model_path = load_model(
            "mobile_sam.pt", remote_url=remote_url, model_dir=cls.model_dir
        ) 

        sam = sam_model_registry[model_type](checkpoint=model_path)

        mask_generator = SamAutomaticMaskGenerator(sam)

        return cls(mask_generator)

    def __call__(self, input_image: Union[np.ndarray, Image.Image]=None, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs) -> np.ndarray:
        self.model.to(self.device)
        super().__call__(image=input_image, detect_resolution=detect_resolution, image_resolution=image_resolution, output_type=output_type, **kwargs)