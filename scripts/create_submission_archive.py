"""
Script to create the final submission archive:
D:\Reasearch\PHOENIX_FEDERATED_LEARNING_SUBMISSION.zip
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Reasearch\fl_benchmark")
ZIP_PATH = Path(r"D:\Reasearch\PHOENIX_FEDERATED_LEARNING_SUBMISSION.zip")

ALWAYS_EXCLUDE_DIRS = {
    ".git",
    "fl_env",
    ".pytest_cache",
    "__pycache__",
    ".matplotlib",
    "diagnostic_attempt1_hang_logs",
    "test_wandb",
    "phase5_recovery",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".tmp",
    ".log",
    ".DS_Store",
}

def should_include(path: Path) -> bool:
    rel_parts = path.relative_to(PROJECT_ROOT).parts
    
    # Always exclude cache and tool directories
    for part in rel_parts:
        if part in ALWAYS_EXCLUDE_DIRS:
            return False
            
    # Exclude historical pilot/smoke folders directly under results/
    if len(rel_parts) >= 2 and rel_parts[0] == "results":
        if rel_parts[1] not in ["primary_benchmark", "README.md"]:
            return False
            
    # Exclude extensions
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return False
        
    # Exclude temporary or socket files
    if "wandb" in rel_parts:
        return False
        
    # Exclude root temporary files if any
    if path.name.startswith(".tmp") or path.name.endswith(".tmp"):
        return False
        
    return True

def create_archive():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        
    print(f"Packaging submission archive to: {ZIP_PATH}")
    included_files = []
    total_uncompressed = 0
    
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file_path in sorted(PROJECT_ROOT.rglob("*")):
            if file_path.is_file() and should_include(file_path):
                rel_path = file_path.relative_to(PROJECT_ROOT)
                archive_name = Path("fl_benchmark") / rel_path
                zipf.write(file_path, arcname=str(archive_name))
                included_files.append((str(rel_path), file_path.stat().st_size))
                total_uncompressed += file_path.stat().st_size
                
    zip_size = ZIP_PATH.stat().st_size
    
    # Compute SHA-256 of ZIP
    import hashlib
    hasher = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest().upper()
    
    print("\n" + "=" * 80)
    print(f"SUBMISSION ARCHIVE CREATED SUCCESSFULLY!")
    print(f"Archive Path:        {ZIP_PATH}")
    print(f"Total Files Packed:  {len(included_files):,}")
    print(f"Uncompressed Size:   {total_uncompressed / (1024*1024):.2f} MB")
    print(f"Compressed ZIP Size: {zip_size / (1024*1024):.2f} MB ({zip_size:,} bytes)")
    print(f"SHA-256 Checksum:    {zip_sha256}")
    print("=" * 80)
    
    # Summary of major categories
    categories = {}
    for rel_str, sz in included_files:
        top = rel_str.split(os.sep)[0]
        categories[top] = categories.get(top, 0) + sz
        
    print("\nPackage Contents by Category:")
    for cat, sz in sorted(categories.items()):
        print(f"  - {cat:25s}: {sz / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_archive()
