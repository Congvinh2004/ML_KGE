"""One-off: export code cells from a pipeline .ipynb to copy-paste .md (same layout as kaggle_*_pipeline.md)."""
import json
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python _ipynb_to_md.py path/to/notebook.ipynb")
        sys.exit(1)
    ipynb = sys.argv[1]
    base, _ = os.path.splitext(ipynb)
    out_path = base + ".md"
    with open(ipynb, encoding="utf-8") as f:
        nb = json.load(f)
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        n = len(parts) + 1
        parts.append("## Cell %d — code\n\n```python\n%s\n```\n" % (n, src.rstrip()))
    text = "\n".join(parts) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(out_path, "code cells:", len(parts))


if __name__ == "__main__":
    main()
