import os
import shutil
import zipfile
import requests

DATA_DIR = "dataset"
FOOD101_ZIP = "food-101.zip"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
FOOD101_TAR = "food-101.tar.gz"

def download_food101():
    if not os.path.exists(FOOD101_TAR):
        print("Downloading Food-101...")
        r = requests.get(FOOD101_URL, stream=True)
        with open(FOOD101_TAR, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download completed.")
    else:
        print("Food-101 TAR already exists.")

def extract_food101():
    if not os.path.exists("food-101"):
        print("Extracting Food-101...")
        import tarfile
        with tarfile.open(FOOD101_TAR) as tar:
            tar.extractall()
        print("Extraction complete.")
    else:
        print("food-101 folder already exists.")

def prepare_dataset():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    images_dir = "food-101/images"
    meta_dir = "food-101/meta"

    # Load train/test file lists
    with open(os.path.join(meta_dir, "train.txt"), "r") as f:
        train_items = [line.strip() for line in f]

    with open(os.path.join(meta_dir, "test.txt"), "r") as f:
        val_items = [line.strip() for line in f]

    print("Creating dataset/train and dataset/val folders...")

    # Helper: copy a list of images into a destination folder
    def copy_split(items, dest_root):
        for item in items:
            cls = item.split("/")[0]
            src_img = os.path.join(images_dir, item + ".jpg")
            dest_cls_folder = os.path.join(dest_root, cls)
            os.makedirs(dest_cls_folder, exist_ok=True)
            shutil.copy(src_img, dest_cls_folder)

    print("Populating train/ ...")
    copy_split(train_items, train_dir)

    print("Populating val/ ...")
    copy_split(val_items, val_dir)

    print("\nFood-101 dataset is ready in /dataset/train and /dataset/val")

if __name__ == "__main__":
    download_food101()
    extract_food101()
    prepare_dataset()
