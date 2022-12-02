from argparse import ArgumentParser
from itertools import combinations
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import List, Tuple

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from numpy import asarray, ndarray
from PIL import Image
from tqdm import tqdm

import data_aug

name_to_fn = {
    "rotate"   : data_aug.rotate,
    "shear"    : data_aug.shear,
    "gaussian" : data_aug.gaussian,
    "motion"   : data_aug.motion,
    "addictive": data_aug.addictive,
    "multiply" : data_aug.multiply,
}

tags = list(name_to_fn.keys())

def getCombinations(l):
    for r in range(1, len(l)):
        for combination in combinations(l, r):
            yield combination
            
def getImage(path:str, width:int, height=int):
    image = asarray(Image.open(path).convert("RGB"))
    image = cv2.resize(image, (width, height))
    return Path(path).stem, image

def augment(save_paths:List[Path], affine:iaa.Affine, images:List[ndarray]):
    ia.seed(current_process().pid)
    images_aug = affine(images=images)
    buffer = []
    for path, image in tqdm(zip(save_paths, images_aug), total=len(images_aug), desc="generating"):
        buffer.append((path, image))
    return buffer

def saveImg(args:Tuple[Path, ndarray]):
    path, img = args
    im = Image.fromarray(img)
    im.save(str(path))

def generate(idx:int, total:int, save_path:Path, image:ndarray, combination:List[str], args):
    total_per_type:int = args.total_per_type
    
    if "spatter" in combination:
        combination.remove("spatter")
        sequence = data_aug.spatter_generator(*[name_to_fn[affine] for affine in combination])
    else:
        sequence = iaa.Sequential([name_to_fn[affine] for affine in combination])
        
    # init path
    length = len(str(total))
    idx = str(idx)
    idx = "0"*(length-len(idx))+idx
    print(f"{idx}/{total} {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)
 
    # start
    images = []
    paths = []
    for i in range(total_per_type):
        save = save_path/f"{i:05}.png"
        if not save.exists():
            images.append(image)
            paths.append(save)
            
    
    if images:
        with Pool() as p:
            buffer = augment(paths, sequence, images)
            buffer = list(tqdm(p.imap_unordered(saveImg, buffer), total=len(buffer), desc="save"))

if __name__ == "__main__":
    parser = ArgumentParser(description="generate augmented images")
    parser.add_argument("-img", "--images", required=True, nargs='+')
    parser.add_argument("--save", default="augmented")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("-t", "--total-per-type", type=int, default=100)
    parser.add_argument("--ignore", nargs="+", help=",".join(tags))
    parser.add_argument("--include", nargs="+", help=",".join(tags))
    args = parser.parse_args()
    
    raw_paths = args.images
    raws = [getImage(image, args.width, args.height) for image in raw_paths]
    
    save_path = Path(args.save)
    save_path.mkdir(exist_ok=True)

    ignores = args.ignore
    includes = args.include
    if not includes:
        includes = tags
    
    total = len([0 for _ in getCombinations(tags)]) * len(raws) - 1
    i = 0
    
    for combination in getCombinations(tags):
        if ignores:
            for ignore in ignores:
                if ignore in combination:
                    continue
        
        should_skip = True
        for include in includes:
            if include in combination:
                should_skip = False
        if should_skip:
            continue

        for name, image in raws:
            save = save_path/"_".join(combination)/name
            generate(i, total, save, image, list(combination), args)
            i += 1