from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class KolorsPipelineOutput(BaseOutput):
    """
    Output class for Kolors pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]

#copied from src/diffusers/pipelines/ledits_pp/pipeline_output.py
@dataclass
class KolorsLEditsPPInversionPipelineOutput(BaseOutput):
    """
    Output class for LEdits++ Diffusion pipelines.

    Args:
        input_images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of the cropped and resized input images as PIL images of length `batch_size` or NumPy array of shape `
            (batch_size, height, width, num_channels)`.
        vae_reconstruction_images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of VAE reconstruction of all input images as PIL images of length `batch_size` or NumPy array of shape
            ` (batch_size, height, width, num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    vae_reconstruction_images: Union[List[PIL.Image.Image], np.ndarray]