import os
import argparse
from typing import List

from main.storage import GalleryStorage
from main.ssan_engine import SSANEngine

from main.config import settings

def load_images_from_folder(folder: str, recursive: bool = True,
                            exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    paths = []
    if recursive:
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(folder, fn))
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=settings.ckpt_path)
    ap.add_argument("--vocab", default=settings.vocab_path)
    ap.add_argument("--gallery_dir", default=settings.gallery_dir)
    ap.add_argument("--artifact_dir", default=settings.artifact_dir)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--device", default=settings.device)
    args = ap.parse_args()

    img_paths = load_images_from_folder(args.gallery_dir, recursive=args.recursive)
    print(f"Found {len(img_paths)} images")

    engine = SSANEngine(ckpt_path=args.ckpt, vocab_path=args.vocab, device=args.device)
    feats = engine.extract_image_features(img_paths, batch_size=args.batch_size)
    print("Features:", feats.shape)

    store = GalleryStorage(args.artifact_dir)
    store.save(feats, img_paths)
    print("Saved gallery to:", args.artifact_dir)

if __name__ == "__main__":
    main()