from argparse import ArgumentParser
from hashlib import md5
from multiprocessing import Pool
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm


def Crop(image_path:Path):
    try:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        result = Image.fromarray(image)
        name = md5(result.tobytes()).hexdigest()
        result.save(f"cropped/{name}.png")
    except Exception as e:
        print(image_path)
        raise e

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    total = [image_path for image_path in path.iterdir()][:4000]
    with Pool() as p:
        results = list(tqdm(p.imap_unordered(Crop, total), total=len(total), desc=path.name))