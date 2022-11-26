from argparse import ArgumentParser
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from random import choice
from time import time, time_ns

import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from numpy import asarray
from PIL import Image
from tqdm import tqdm

rotate          = iaa.Affine(rotate=(-180, 180))
shear           = iaa.Affine(shear=(-20, 20))
gaussian        = iaa.GaussianBlur(sigma=(0.0, 4.0))
motion          = iaa.MotionBlur(k=15)
addictive       = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
multiply        = iaa.Multiply((0.5, 1.5), per_channel=0.5)
spatter_orig    = [iaa.imgcorruptlike.Spatter(severity=i) for i in range(1, 4)]

class spatter_generator:
    def __init__(self, *args) -> None:
        self.fn = [iaa.Sequential([spatter_orig[i], *args]) for i in range(len(spatter_orig))]
    def __call__(self, image=None, images=None):
        if not images is None:
            result = []
            for image in images:
                aug = choice(self.fn)
                result.append(aug(image=image))
            return result
        elif not image is None:
            aug = choice(self.fn)
            return aug(image=image)

def save(args):
    path:Path = args[0]
    img:np.ndarray = args[1]
    im = Image.fromarray(img)
    im.save(str(path))
    
def augment(args):
    save_path, affine, images = args
    images_aug = affine(image=images)
    return save_path, images_aug

    # images_aug = affine(images=images)
    # return save_path, ia.draw_grid(images_aug, cols=4, rows=4)
    

def generate(idx:int, total:int, path:Path, img_fn, affine):
    global total_per_type
    
    ia.seed(time_ns())
    length = len(str(total))
    idx = str(idx)
    idx = "0"*(length-len(idx))+idx    
    print(f"{idx}/{total} {path}")
    
    # init
    path.mkdir(parents=True, exist_ok=True)
    
    sample:np.ndarray = img_fn()
    shape = sample.shape
    
    # images = [img_fn() for _ in range(16)]
    images = img_fn()
    # images = []
    # for i in range(4):
    #     for j in range(4):
    #         if i%2 == 0 and j%2==1 or i%2==1 and j%2==0:
    #             images.append(img_fn())
    #         else:
    #             images.append(np.zeros(shape, dtype=sample.dtype))
    
    
    results = []
    buffers = []
    
    for i in range(total_per_type):
        save_path = path/f"{i:08}.png"
        if save_path.exists():
            continue
        buffers.append((save_path, affine, images))
        
    # if len(buffers):
    #     # if multiprocess, all images will be same due to pickling the randomizer, therefore it has to be single processing
    #     for buffer in tqdm(buffers, desc="generate"):
    #         results.append(augment(buffer))
        
    # if there exists tasks
    if len(buffers):
        results = [augment(buffer) for buffer in tqdm(buffers, desc="generate")]
        # results = [save(result) for result in tqdm(results, desc="save")]
    
    if len(buffers):
        with Pool(8) as p:
            # results = list(tqdm(p.imap_unordered(augment, buffers), total=len(buffers), desc=f"generate"))
            results = list(tqdm(p.imap_unordered(save, results), total=len(results), desc=f"save"))
            
if __name__ == "__main__":
    global toal_per_type
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    
    total_per_type = args.n
    
    # path
    RED   = "red.png"
    GREEN = "green.jpg"

    # images
    red   = asarray(Image.open(RED))
    green = asarray(Image.open(GREEN))

    red   = cv2.cvtColor(red, cv2.COLOR_RGBA2RGB)
    red   = cv2.resize(red, (256, 256))
    green = cv2.resize(green, (256, 256))

    Image.fromarray(red).save("template_red.png")
    Image.fromarray(green).save("template_green.png")
    
    
    name_to_fn = {
        "rotate"   : rotate,
        "shear"    : shear,
        "gaussian" : gaussian,
        "motion"   : motion,
        "addictive": addictive,
        "multiply" : multiply,
    }
    
    tags = ["spatter","gaussian","motion","addictive","multiply","shear", "rotate"]
    
    
    
    def getCombinations(l):
        for r in range(1, len(l)):
            for combination in combinations(l, r):
                yield combination
    
    start = time()
    # run
    total = len([c for c in getCombinations(tags)]) * 3 - 1
    for i, combination in enumerate(getCombinations(tags)):
        i = i * 3
        path = Path("augmented")/Path("_".join(combination))
        if "spatter" in combination:
            combination = combination[1:]
            aug = spatter_generator(*[name_to_fn[name] for name in combination])
        else:            
            aug =  iaa.Sequential([name_to_fn[name] for name in combination])
        generate(i, total, path/"red", lambda:red, aug)
        generate(i+1, total, path/"green", lambda:green, aug)
        generate(i+2, total, path/"red_green", lambda:choice([red, green]), aug)
        
    # end
    total = time() - start
    hours = int(total // 60 // 60)
    minutes = int(total // 60 - hours * 60)
    seconds = int(total - minutes * 60 - hours * 3600)
    print("total used", hours, "hours", minutes, "minutes", seconds, "seconds")
