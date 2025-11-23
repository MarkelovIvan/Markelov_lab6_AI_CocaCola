# prepare_data.py
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

RANDOM_SEED = 42

RAW_DIR = "../data/raw"
OUT_DIR = "../data/processed"

AUG_OPS = ['flip', 'rotate', 'blur']


# Create directories
def make_dirs(base, splits=('train','val','test'), classes=('logo','not_logo')):
    for s in splits:
        for c in classes:
            Path(base, s, c).mkdir(parents=True, exist_ok=True)


# Split raw data
def split_and_copy(raw_dir, out_dir, ratios=(0.7,0.15,0.15)):
    random.seed(RANDOM_SEED)

    classes = [d.name for d in Path(raw_dir).iterdir() if d.is_dir()]
    make_dirs(out_dir, classes=classes)

    for cls in classes:
        files = list(Path(raw_dir, cls).glob('*.*'))
        random.shuffle(files)
        n = len(files)

        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)

        parts = {
            'train': files[:n_train],
            'val': files[n_train:n_train+n_val],
            'test': files[n_train+n_val:]
        }

        for split, flist in parts.items():
            print(f"[INFO] Copying {len(flist)} {cls} images to {split}/")
            for f in flist:
                dest = Path(out_dir, split, cls, f.name)
                shutil.copy(f, dest)


def augment_train_set(out_dir):
    train_dir = Path(out_dir, "train")
    print("[INFO] Starting augmentation...")

    for cls_dir in train_dir.iterdir():
        if not cls_dir.is_dir():
            continue

        for img_path in list(cls_dir.glob("*.*")):
            for op in AUG_OPS:
                img = Image.open(img_path).convert("RGB")
                aug_img = img.copy()

                if op == "flip":
                    aug_img = ImageOps.mirror(aug_img)
                elif op == "rotate":
                    aug_img = aug_img.rotate(random.choice([15, -15, 5, -5]))
                elif op == "blur":
                    aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=1))

                out_name = img_path.stem + f"_{op}" + img_path.suffix
                aug_img.save(cls_dir / out_name)

    print("[INFO] Augmentation finished.")


# MAIN
def main():
    raw_path = Path(RAW_DIR)
    out_path = Path(OUT_DIR)

    print("[INFO] Splitting dataset...")
    split_and_copy(raw_path, out_path)

    print("[INFO] Augmenting training data...")
    augment_train_set(out_path)

    print("[INFO] DONE! Processed dataset saved to:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
