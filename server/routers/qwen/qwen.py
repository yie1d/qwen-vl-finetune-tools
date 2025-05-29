import io
from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form

from .model import Qwen2_5_VL_7B

router = APIRouter(
    prefix="/qwen",
    tags=["routers"]
)


@router.post('/infer', response_model=dict[str, list[int]])
async def infer(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
):
    bbox = Qwen2_5_VL_7B().inference(
        io.BytesIO(await image.read()),
        prompt
    )
    return {
        'bbox': bbox
    }
