import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from time import time
import cv2
from tqdm import tqdm

def crawler(path:Path):
    total = len([0 for _ in path.iterdir()])
    for images in tqdm(path.iterdir(), total=total, desc=str(path)):
        for images_type in images.iterdir():
            for image in images_type.iterdir():
                yield images_type.name, image

def readCache(model:defaultdict, save:Path):
    print("reading", save)
    f = open(str(save), "rb")
    while True:
        try:
            image_type, name, img = pickle.load(f)
            model[image_type][name]=img
        except EOFError:
            break
        except Exception as e:
            raise e
    f.close()

def cache(path:Path,save:Path):
    start = time()
    model = defaultdict(dict)
    if save.exists():
        readCache(model, save)
        print("read cache used", "%.2f"%(time()-start), "seconds")
        
        f = open(str(save), "ab")
    else:
        f = open(str(save), "wb")
        
    for image_type, image in crawler(path):
        if str(image) in model[image_type]:
            continue
        name = str(image)
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model[image_type][name] = img
        pickle.dump((image_type, name, img), f)
        
    print("total cache used", "%.2f"%(time()-start), "seconds")
    f.close()
    return model

if __name__ == "__main__":
    parser = ArgumentParser("used to save image, generated by prepare.py, into images.cache")
    parser.add_argument("path")
    parser.add_argument("save", default="image.cache")
    args = parser.parse_args()
    
    path = Path(args.path)
    save = Path(args.save)
    model = defaultdict(dict)
    
    model = cache(path, save)