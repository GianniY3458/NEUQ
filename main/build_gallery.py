# build_gallery.py
import os
import argparse
from typing import List

from storage import GalleryStorage
from ssan_engine import SSANEngine  # 改成你的 engine 文件名


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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--gallery_dir", required=True)
    ap.add_argument("--artifact_dir", default="artifacts")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--device", default="cuda:0")
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