# app.py
from typing import List
import time
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import os
from pathlib import Path

from config import settings
from storage import GalleryStorage
from ssan_engine import SSANEngine  # 改成你的 engine 文件名
from typing import Optional

# 使用 lifespan 替代 on_event
@asynccontextmanager
async def lifespan(app):
    global engine, store
    store = GalleryStorage(settings.artifact_dir)
    engine = SSANEngine(
        ckpt_path=settings.ckpt_path,
        vocab_path=settings.vocab_path,
        device=settings.device
    )
    if not store.exists():
        raise RuntimeError(
            f"Gallery artifacts not found in '{settings.artifact_dir}'. "
            f"Run build_gallery.py first."
        )
    g = store.load()
    engine.set_gallery(g.features, g.paths)
    yield

app = FastAPI(title="SSAN Text-to-Person Search API", version="1.0.0", lifespan=lifespan)

engine: Optional[SSANEngine] = None
store: Optional[GalleryStorage] = None


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topk: int = Field(default=settings.default_topk, ge=1, le=100)


class SearchItem(BaseModel):
    path: str
    score: float


class SearchResponse(BaseModel):
    text: str
    topk: int
    ms: float
    results: List[SearchItem]


## 已由 lifespan 替代


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
def stats():
    if engine is None or engine.gallery_features is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {
        "gallery_size": int(engine.gallery_features.shape[0]),
        "feature_dim": int(engine.gallery_features.shape[1]),
        "device": settings.device,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if engine is None or engine.gallery_features is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    t0 = time.time()
    results = engine.search(req.text, topk=req.topk)
    ms = (time.time() - t0) * 1000.0

    return SearchResponse(
        text=req.text,
        topk=req.topk,
        ms=ms,
        results=[SearchItem(path=p, score=s) for p, s in results],
    )


class UploadResponse(BaseModel):
    success: bool
    message: str
    uploaded_files: List[str]


# 新增的代码（`upload_gallery`接口）
@app.post("/upload_gallery", response_model=UploadResponse)
async def upload_gallery(files: List[UploadFile] = File(...)):
    """
    接收图片文件并提取特征，保存到图库
    """
    if engine is None or store is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    temp_dir = Path("temp_images")
    temp_dir.mkdir(parents=True, exist_ok=True)

    new_img_paths = []
    for file in files:
        safe_name = os.path.basename(file.filename)
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename")

        img_path = str(temp_dir / safe_name)
        new_img_paths.append(img_path)

        with open(img_path, "wb") as f:
            f.write(await file.read())

    # 1) 提取新图片特征
    new_features = engine.extract_image_features(new_img_paths, batch_size=32)

    # 2) 加载现有图库并更新
    existing_gallery = store.load()
    combined_features = np.concatenate((existing_gallery.features, new_features), axis=0)
    combined_paths = existing_gallery.paths + new_img_paths

    # 3) 保存更新后的图库
    store.save(combined_features, combined_paths)
    engine.set_gallery(combined_features, combined_paths)

    return UploadResponse(success=True, message="Gallery updated", uploaded_files=new_img_paths)
