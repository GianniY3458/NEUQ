from typing import List, Optional
import time
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from main.config import settings
from main.storage import GalleryStorage
from main.ssan_engine import SSANEngine

# ---------------------------------------------------------------------------
# 目录初始化（确保静态目录存在，否则 StaticFiles 会报错）
# ---------------------------------------------------------------------------
GALLERY_DIR = Path(settings.gallery_dir)
TEMP_IMAGES_DIR = Path("temp_images")
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
TEMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 路径 → 前端可访问的 URL
# ---------------------------------------------------------------------------
def path_to_image_url(file_path: str) -> str:
    """将本地文件路径转为前端可访问的 URL"""
    normalized = file_path.replace("\\", "/")
    basename = os.path.basename(file_path)
    if "temp_images" in normalized:
        return f"/static/uploads/{basename}"
    else:
        return f"/static/gallery/{basename}"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    global engine, store
    store = GalleryStorage(settings.artifact_dir)
    engine = SSANEngine(
        ckpt_path=settings.ckpt_path,
        vocab_path=settings.vocab_path,
        device=settings.device,
    )
    if not store.exists():
        raise RuntimeError(
            f"Gallery artifacts not found in '{settings.artifact_dir}'. "
            f"Run build_gallery.py first."
        )
    g = store.load()
    engine.set_gallery(g.features, g.paths)
    yield


# ---------------------------------------------------------------------------
# App 实例
# ---------------------------------------------------------------------------
SSAN = FastAPI(
    title="SSAN Text-to-Person Search API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS —— 放在最前面，确保 OPTIONS 预检请求也能通过
SSAN.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[SSANEngine] = None
store: Optional[GalleryStorage] = None


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topk: int = Field(default=settings.default_topk, ge=1, le=100)


class SearchItem(BaseModel):
    path: str        # 原始本地路径（调试用）
    image_url: str   # 前端可直接使用的图片 URL
    score: float


class SearchResponse(BaseModel):
    text: str
    topk: int
    ms: float
    results: List[SearchItem]


class UploadResponse(BaseModel):
    success: bool
    message: str
    uploaded_files: List[str]   # 上传后的图片 URL 列表


class StatsResponse(BaseModel):
    gallery_size: int
    feature_dim: int
    device: str


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@SSAN.get("/health")
def health():
    return {"ok": True}


@SSAN.get("/stats", response_model=StatsResponse)
def stats():
    if engine is None or engine.gallery_features is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return StatsResponse(
        gallery_size=int(engine.gallery_features.shape[0]),
        feature_dim=int(engine.gallery_features.shape[1]),
        device=settings.device,
    )


@SSAN.post("/search", response_model=SearchResponse)
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
        results=[
            SearchItem(
                path=p,
                image_url=path_to_image_url(p),
                score=s,
            )
            for p, s in results
        ],
    )


@SSAN.post("/upload_gallery", response_model=UploadResponse)
async def upload_gallery(files: List[UploadFile] = File(...)):
    """接收图片文件，提取特征并更新图库"""
    if engine is None or store is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    TEMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    new_img_paths = []
    for file in files:
        safe_name = os.path.basename(file.filename or "")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        img_path = str(TEMP_IMAGES_DIR / safe_name)
        new_img_paths.append(img_path)
        with open(img_path, "wb") as f:
            f.write(await file.read())

    # 1) 提取新图片特征
    new_features = engine.extract_image_features(new_img_paths, batch_size=32)

    # 2) 合并现有图库
    existing_gallery = store.load()
    combined_features = np.concatenate(
        (existing_gallery.features, new_features), axis=0
    )
    combined_paths = existing_gallery.paths + new_img_paths

    # 3) 保存 & 刷新内存
    store.save(combined_features, combined_paths)
    engine.set_gallery(combined_features, combined_paths)

    uploaded_urls = [path_to_image_url(p) for p in new_img_paths]
    return UploadResponse(
        success=True,
        message=f"已上传 {len(new_img_paths)} 张图片，图库已更新",
        uploaded_files=uploaded_urls,
    )


@SSAN.post("/clear_gallery")
def clear_gallery():
    """清空图库（特征 + 图片文件）"""
    global engine, store
    if store is None or engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    # 获取当前特征维度（如果有的话），兜底用 1024
    feat_dim = 1024
    if engine.gallery_features is not None and engine.gallery_features.ndim == 2:
        feat_dim = engine.gallery_features.shape[1]

    # 清空特征
    store.save(np.empty((0, feat_dim)), [])
    engine.set_gallery(np.empty((0, feat_dim)), [])

    # 删除 temp_images 下所有文件
    if TEMP_IMAGES_DIR.exists():
        for f in TEMP_IMAGES_DIR.iterdir():
            if f.is_file():
                f.unlink()

    return {"success": True, "message": "图库已清空"}


# ---------------------------------------------------------------------------
# 静态文件挂载（让前端可以通过 URL 访问图片）
# ---------------------------------------------------------------------------
SSAN.mount("/static/gallery", StaticFiles(directory=str(GALLERY_DIR)), name="gallery_static")
SSAN.mount("/static/uploads", StaticFiles(directory=str(TEMP_IMAGES_DIR)), name="uploads_static")
