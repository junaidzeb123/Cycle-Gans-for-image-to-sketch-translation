from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from models.generator import Generator
from utils import detect_domain, load_image, tensor_to_bytes, to_tensor

BASE_DIR = Path(__file__).resolve().parent
WEIGHT_DIR = BASE_DIR / "weight"

app = FastAPI(title="CycleGAN Deployment API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_photo_to_sketch: Optional[Generator] = None
_sketch_to_photo: Optional[Generator] = None


def _load_generator(filename: str) -> Generator:
    weights_path = WEIGHT_DIR / filename
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    model = Generator().to(_device)
    state_dict = torch.load(weights_path, map_location=_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event() -> None:
    global _photo_to_sketch, _sketch_to_photo
    _photo_to_sketch = _load_generator("G_photo_to_sketch.pth")
    _sketch_to_photo = _load_generator("G_sketch_to_photo.pth")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _photo_to_sketch, _sketch_to_photo
    _photo_to_sketch = None
    _sketch_to_photo = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "device": str(_device)}


@app.post("/translate", tags=["inference"])
async def translate_image(file: UploadFile = File(...)) -> dict[str, str]:
    if file.content_type is None or not file.content_type.startswith("image"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    try:
        image = load_image(data)
    except Exception as exc:  # pragma: no cover - PIL specific errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode image payload.",
        ) from exc

    domain = detect_domain(image)
    generator = _photo_to_sketch if domain == "photo" else _sketch_to_photo
    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Requested generator is not available.",
        )

    input_tensor = to_tensor(image, _device)
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    image_bytes = tensor_to_bytes(output_tensor)
    headers = {
        "X-Detected-Domain": domain,
        "X-Generated-Domain": "sketch" if domain == "photo" else "photo",
    }
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


