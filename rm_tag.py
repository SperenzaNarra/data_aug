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
    parser.add_argument("-k", "--keywords", nargs='+')
    parser.add_argument('-n', action="store_true", help="negate")
    args = parser.parse_args()

    source = Path(args.source)
    keywords = args.keywords
    negate = args.n

    if keywords:
        for subpath in source.iterdir():
            if subpath.is_dir():
                names = subpath.name.split("_")
                should_rm = not negate

                for keyword in keywords:
                    if not keyword in names:
                        should_rm = negate
                        break

                if should_rm:
                    rm_dir(subpath)
