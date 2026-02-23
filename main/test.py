"""Quick smoke test – run from project root:  python -m main.smoke_test"""
import os
from main.ssan_engine import SSANEngine
from main.config import settings


def load_images_from_folder(folder_path, recursive=True,
                            exts=(".jpg", ".jpeg", ".png", ".bmp")):
    img_paths = []
    walker = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
    for root, _, files in walker:
        for f in files:
            if f.lower().endswith(exts):
                img_paths.append(os.path.join(root, f))
    img_paths.sort()
    return img_paths


def main():
    gallery_folder = settings.gallery_dir
    img_paths = load_images_from_folder(gallery_folder)
    if not img_paths:
        print(f"No images found in {gallery_folder}")
        return

    engine = SSANEngine(
        ckpt_path=settings.ckpt_path,
        vocab_path=settings.vocab_path,
        device=settings.device,
    )

    gallery_features = engine.extract_image_features(img_paths, batch_size=32)
    engine.set_gallery(gallery_features, img_paths)

    query = ("A tall, thin girl in her mid 20's with shoulder-length, fringed, "
             "straight, black hair is wearing a grey-colored long overcoat.")
    result = engine.search(query, topk=5)
    print(result)


if __name__ == "__main__":
    main()