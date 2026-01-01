from __future__ import annotations

import base64
import io
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ImageDomain = Literal["photo", "sketch"]

_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def load_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def detect_domain(image: Image.Image) -> ImageDomain:
    """Heuristic classifier to distinguish photo vs sketch inputs."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2, keepdims=True)

    chroma = float(np.linalg.norm(rgb - gray, axis=2).mean())

    if chroma >= 0.025:
        return "photo"

    gray2d = gray[..., 0]
    dx = np.abs(np.diff(gray2d, axis=1)).mean()
    dy = np.abs(np.diff(gray2d, axis=0)).mean()
    edge_density = (dx + dy) * 0.5

    if edge_density >= 0.03:
        return "sketch"

    return "photo"


def to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL image to normalized batch tensor."""
    tensor = _transform(image).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert model output tensor to a PIL Image."""
    tensor = tensor.squeeze(0).detach().cpu().mul(0.5).add(0.5).clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize tensor output into PNG bytes."""
    image = tensor_to_image(tensor)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert output tensor back to PNG base64 string."""
    return base64.b64encode(tensor_to_bytes(tensor)).decode("utf-8")
