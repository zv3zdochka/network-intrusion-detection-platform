from pathlib import Path

def print_tree(path: Path, prefix: str = ""):
    entries = [
        p for p in path.iterdir()
        if not p.name.startswith(".")
    ]
    entries.sort(key=lambda x: (x.is_file(), x.name.lower()))

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension)

if __name__ == "__main__":
    root = Path(".").resolve()
    print(root.name)
    print_tree(root)
