import pickle
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import Dict
from multiprocessing import Pool
import cv2
from numpy import ndarray
from tqdm import tqdm
from multiprocessing import Pool


def crawler(path:Path):
    total = len([0 for _ in path.iterdir()])
    for image in tqdm(path.iterdir(), total=total, desc=str(path)):
        yield image
            
def readCache(model:dict, save:Path):
    print("reading", save)
    f = open(str(save), "rb")
    while True:
        try:
            name, img = pickle.load(f)
            model[name]=img
        except EOFError:
            break
        except Exception as e:
            raise e
    f.close()

def load(name:str):
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return name, img

def cache(path:Path,save:Path):
    start = time()
    model:Dict[str, ndarray] = {}
    if save.exists():
        readCache(model, save)
        print("read cache used", "%.2f"%(time()-start), "seconds")
        
        f = open(str(save), "ab")
    else:
        f = open(str(save), "wb")
        
    images = []
    for image in crawler(path):
        if not str(image) in model:
            images.append(str(image))
    
    if images:
        with Pool() as p:
            buffer = list(tqdm(p.imap_unordered(load, images), total=len(images), desc=str(path)))
        for name, img in buffer:
            model[name] = img
            pickle.dump((name, img), f)
        
    print("total cache used", "%.2f"%(time()-start), "seconds")
    f.close()
    return model

if __name__ == "__main__":
    parser = ArgumentParser(description="used to save unclassified images")
    parser.add_argument("path")
    parser.add_argument("save")
    args = parser.parse_args()
    
    path = Path(args.path)
    save = Path(args.save)
    model = {}
    
    model = cache(path, save)