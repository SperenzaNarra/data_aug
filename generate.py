import pickle
from argparse import ArgumentParser
from pathlib import Path
from time import time
from random import choice
from PIL import Image
from multiprocessing import Pool
from imgaug import draw_grid
import cv2
from tqdm import tqdm

def generate(args):
    global model
    
    selected, image_path, label_path, n = args
    
    images = []
    labels = []
    
    for i in range(n):
        for j in range(n):
            idx = int(i * n + j)
            name = selected[idx]
            index = model[name]["index"]
            images.append(model[name]["image"])
            labels.append(f"{index} {center+j*width} {center+i*width} {width} {width}")
            
    image = draw_grid(images, rows=n, cols=n)
    image = cv2.resize(image, (640, 640))
    image = Image.fromarray(image)
    image.save(str(image_path))
    lable = "\n".join(labels)
    with label_path.open("w") as f:
        f.write(lable)

if __name__ == "__main__":
    global model
    start = time()
    model = {}
    
    parser = ArgumentParser(description="generate nxn images")
    parser.add_argument("n", type=int)
    parser.add_argument("-t", "--total", type=int, default=1000)
    args = parser.parse_args()
    
    N     = args.n
    TOTAL = args.total
    IMAGES = Path("images")
    LABLES = Path("labels")
    IMAGES.mkdir(parents=True,exist_ok=True)
    LABLES.mkdir(parents=True,exist_ok=True)
    
    
    print("loading")
    saved_path = Path("images.cache")
    if saved_path.exists():
        model = pickle.load(saved_path.open("rb"))
        
    images_src = Path("cache/images")
    labels_src = Path("cache/labels")
    
    total = len([0 for _ in images_src.iterdir()])
    if total:
        for image in tqdm(images_src.iterdir(), total=total):
            if not image.name in model:
                label_path = labels_src/f"{image.stem}.txt"
                with label_path.open("r") as f:
                    line = f.readline()
                    index, _, _, _, _ = line.split()
                img = cv2.imread(str(image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                model[image.name] = {"image":img, "index":index}
    
    print("used", f"{time()-start:.04f}", "seconds")
    choices = tuple(model.keys())
    
    
    # Core
    for n in range(2, N+1):
        buffer = []
        width = 1/n
        center = width/2
        for i in range(TOTAL):
            image_path = IMAGES/f"{n}-{i:05}.png"
            label_path = LABLES/f"{n}-{i:05}.txt"
            
            if image_path.exists():
                continue
            selected = [choice(choices) for _ in range(int(N*N))]
            buffer.append((selected, image_path, label_path, n))
        if buffer:
            with Pool(8) as p:
                results = list(tqdm(p.imap_unordered(generate, buffer), total=len(buffer), desc=f"generate {n}x{n}"))
        
        
    print("caching")
    pickle.dump(model,saved_path.open("wb"))
    print("used", f"{time()-start:.04f}", "seconds")