# config.py
from pydantic import BaseModel

class Settings(BaseModel):
    ckpt_path: str = "./model/best.pth.tar"
    vocab_path: str = "./model/ind2word.pkl"
    gallery_dir: str = "./gallery"
    artifact_dir: str = "./artifacts"
    device: str = "cuda:0"
    default_topk: int = 5

settings = Settings()