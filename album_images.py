from argparse import ArgumentParser
from hashlib import md5
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
from PIL import Image

AFFINE = A.Affine(rotate=[-20, 20], shear=[-20, 20])
CROP = A.RandomCrop(width=450, height=450)
AUGMENTOR = A.Compose([
    AFFINE, 
    # CROP
    ], bbox_params=A.BboxParams(format='yolo'))

def augment(image_path:Path):
    global AUGMENTOR, RESULT

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label_path = image_path.parents[1]/"labels"/f"{image_path.stem}.txt"
    labels = []
    with label_path.open("r") as f:
        for line in f.readlines():
            index, center_x, center_y, width, height = line.split()
            labels.append((float(center_x), float(center_y), float(width), float(height), int(index)))

    transformed = AUGMENTOR(image=image, bboxes=labels)
    transformed_image = transformed['image']
    transformed_labels = transformed['bboxes']

    img = Image.fromarray(transformed_image)
    name = md5(img.tobytes()).hexdigest()
    img.save(RESULT/"images"/f"{name}.png")

    with (RESULT/"labels"/f"{name}.txt").open("w") as f:
        for center_x, center_y, width, height, index in transformed_labels:
            f.write(f"{index} {center_x} {center_y} {width} {height}\n")

    print(image_path.name, f"{name}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("yolo_dir")
    args = parser.parse_args()

    images = Path(args.yolo_dir)/"images"

    RESULT = Path("result")
    (RESULT/"images").mkdir(parents=True, exist_ok=True)
    (RESULT/"labels").mkdir(parents=True, exist_ok=True)

    image_paths = [i for i in images.iterdir()]

    with Pool() as p:
        result = list(p.imap_unordered(augment, image_paths))

    # for image in image_paths:
    #     augment(image)