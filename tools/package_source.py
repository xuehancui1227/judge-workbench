"""Build a source-only handoff; never include user datasets, Key or runtime state."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
target = root.parent/"judge-workbench-source.zip"
skip = {"data", ".venv", "__pycache__", ".pytest_cache", ".git", "node_modules"}
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(root.rglob("*")):
        if file.is_file() and not any(part in skip for part in file.relative_to(root).parts):
            if file.name in {"secrets.toml", ".env"} or file.suffix in {".log", ".pyc"}:
                continue
            archive.write(file, Path(root.name)/file.relative_to(root))
print(target)
