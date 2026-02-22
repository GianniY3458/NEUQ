import os
from ssan_engine import SSANEngine  # 改成你的文件名

import os

def load_images_from_folder(folder_path,
                            recursive=True,
                            exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    读取文件夹内所有图片路径
    """
    img_paths = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(exts):
                    img_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            if file.lower().endswith(exts):
                img_paths.append(os.path.join(folder_path, file))

    img_paths.sort()  # 保证顺序固定
    return img_paths
def main():
    ckpt_path = "../model/best.pth.tar"   # 改成你的
    vocab_path = "../model/ind2word.pkl"      # 改成你的
    gallery_folder = "../test"  # 改成你的
    img_paths = load_images_from_folder(gallery_folder)
    engine = SSANEngine(ckpt_path=ckpt_path, vocab_path=vocab_path)

    gallery_features = engine.extract_image_features(img_paths, batch_size=32)
    engine.set_gallery(gallery_features, img_paths)

    # 3) 文本特征（验证 shape + L2 norm）
    q = "A tall, thin girl in her mid 20's with shoulder-length, fringed, straight, black hair is wearing a grey-colored long overcoat paired with black colored skinny jeggings and a white-colored high neck inner. She is also wearing a pair of black ankle-length boots."

    # 4) 检索（验证不会报错）
    result = engine.search(q, topk=10)
    print(result)

if __name__ == "__main__":
    main()