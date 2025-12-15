import os
import shutil
import tarfile
import requests
from sklearn.model_selection import train_test_split

DATA_DIR = "dataset"
FOOD101_TAR = "food-101.tar.gz"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EXTRACTED_DIR = "food-101/images"

def download_food101():
    if not os.path.exists(FOOD101_TAR):
        print("Downloading Food-101...")
        r = requests.get(FOOD101_URL, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(FOOD101_TAR, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\rDownloaded {downloaded/1024/1024:.2f}/{total_size/1024/1024:.2f} MB", end="")
        print("\nDownload completed.")
    else:
        print("Food-101 TAR already exists.")

def safe_extract(tar_path, extract_path="."):
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            member_path = os.path.join(extract_path, member.name)
            if not os.path.commonprefix([os.path.abspath(extract_path), os.path.abspath(member_path)]) == os.path.abspath(extract_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(extract_path)
    print("Extraction complete.")

def extract_food101():
    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting Food-101...")
        safe_extract(FOOD101_TAR)
    else:
        print("food-101/images folder already exists.")

def prepare_dataset(test_size=0.2, random_state=42):
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    categories = [cat for cat in os.listdir(EXTRACTED_DIR) if os.path.isdir(os.path.join(EXTRACTED_DIR, cat))]

    print("Creating train/val dataset...")

    for category in categories:
        cat_path = os.path.join(EXTRACTED_DIR, category)
        images = [f for f in os.listdir(cat_path) if f.endswith(".jpg")]

        train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=random_state)

        # Copy train images
        train_cat_dir = os.path.join(train_dir, category)
        os.makedirs(train_cat_dir, exist_ok=True)
        for img in train_imgs:
            shutil.copy(os.path.join(cat_path, img), train_cat_dir)

        # Copy val images
        val_cat_dir = os.path.join(val_dir, category)
        os.makedirs(val_cat_dir, exist_ok=True)
        for img in val_imgs:
            shutil.copy(os.path.join(cat_path, img), val_cat_dir)

    print("\nFood-101 dataset is ready in /dataset/train and /dataset/val")

if __name__ == "__main__":
    download_food101()
    extract_food101()
    prepare_dataset()
