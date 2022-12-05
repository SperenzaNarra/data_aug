from pathlib import Path
from argparse import ArgumentParser

def rm_dir(path:Path):
    for subpath in path.iterdir():
        if subpath.is_file():
            subpath.unlink()
        elif subpath.is_dir():
            print(subpath)
            rm_dir(subpath)
            subpath.rmdir()
    path.rmdir()
    

if __name__ == "__main__":
    parser = ArgumentParser(description="rm all directories in source which contain keyword")
    parser.add_argument("source")
    parser.add_argument("types", nargs="+")
    args = parser.parse_args()

    source = Path(args.source)
    types = args.types

    for path in source.iterdir():
        for subpath in path.iterdir():
            if subpath.name in types:
                rm_dir(subpath)