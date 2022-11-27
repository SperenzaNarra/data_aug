from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from random import choice
from typing import Dict, Tuple

import cv2
import imgaug as ia
from PIL import Image
from tqdm import tqdm

from Cache import cache


def get_label(name:str, x:int, y:int, n:int):
    global name_to_index
    assert x < n and y < n, f"invalid value x {x} or invalid value y {y}, which suppose to smaller than {n}"
    index = name_to_index[name]
    width = 1 / n
    center = width / 2
    return f"{index} {center+width*x} {center+width*y} {width} {width}\n"

def generate(args:Tuple[Path, int]):
    global CHOICES, IMAGES, LABELS, MODEL, WIDTH, HEIGHT
    
    save_path, n = args
    save_image = IMAGES/f"{save_path}.png"
    save_label = LABELS/f"{save_path}.txt"
    images = []
    
    with save_label.open("w") as f:
        for x in range(n):
            for y in range(n):
                image_type = choice(CHOICES)
                images.append(choice(MODEL[image_type]))
                f.write(get_label(image_type, x, y, n))
    
    image = ia.draw_grid(images, cols=n, rows=n)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    im = Image.fromarray(image)
    im.save(str(save_image))
            
if __name__ == "__main__":
    parser = ArgumentParser(description="reading images generated by prepare.py and generate images")
    parser.add_argument("-p", "--path", type=str, default="augmented", help="path of images generated by prepare.py")
    parser.add_argument("-t", "--type", type=int, default=0, help="select image type, 0 means all")
    parser.add_argument("--width",      type=int, default=640, help="width of generated image")
    parser.add_argument("--height",     type=int, default=640, help="height of generated image")
    parser.add_argument("-n",           type=int, default=2, help="generate images from 2x2 to nxn")
    parser.add_argument("-T", "--total-per-type", type=int, default=100, help="numbers for each nxn images")
    parser.add_argument("--save",                           default="yolo", help="directory to save images and lables")
    args = parser.parse_args()
    
    
    
    model = cache(Path(args.path))
    CHOICES = list(model.keys())
    name_to_index:Dict[str, int] = {name:i for i, name in enumerate(CHOICES)}
    assert len(CHOICES) >= args.type, f"invalid type, you need to choose a number between 0 and {len(CHOICES)} for {CHOICES}"
    
    if args.type != 0:
        CHOICES = [CHOICES[args.type - 1]]
        print("You select image", CHOICES[0])
    else:
        print("You select all images")

    MODEL = defaultdict(list)
    for image_type in CHOICES:
        for image in model[image_type]:
            MODEL[image_type].append(model[image_type][image])

    # global variable
    N = args.n
    WIDTH = args.width
    HEIGHT = args.height
    TOTAL = args.total_per_type
    
    IMAGES = Path(args.save)/"images"
    LABELS = Path(args.save)/"lables"
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(exist_ok=True)
    
    for n in range(2, N+1):
        images = []
        for i in range(TOTAL):
            save_path = f"{n}-{i:05}"
            if not (IMAGES/f"{save_path}.png").exists():
                images.append((save_path, n))
        if images:
            # for opts in tqdm(images, desc=f"{n}x{n}"):
            #     generate(opts)
            with Pool() as p:
                results = list(tqdm(p.imap_unordered(generate, images), total=len(images), desc=f"{n}x{n}"))
        else:
            print(f"{n}x{n}")