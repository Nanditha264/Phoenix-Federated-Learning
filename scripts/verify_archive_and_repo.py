"""
Complete verification of the final submission archive and repository readiness.
"""

import os
import re
import zipfile
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Reasearch\fl_benchmark")
ZIP_PATH = Path(r"D:\Reasearch\PHOENIX_FEDERATED_LEARNING_SUBMISSION.zip")

def verify_all():
    print("=" * 80)
    print("FINAL LOCAL PACKAGE VERIFICATION")
    print("=" * 80)
    
    # 1. ZIP file integrity and hash
    assert ZIP_PATH.exists(), f"ZIP not found: {ZIP_PATH}"
    size_bytes = ZIP_PATH.stat().st_size
    hasher = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest().upper()
    print(f"Archive:       {ZIP_PATH.name}")
    print(f"Size:          {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
    print(f"SHA-256:       {zip_sha256}")
    
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"ERROR: Corrupted entry in ZIP: {bad_file}")
            return False
        print("Integrity:     100% OK (All CRC32 checksums verified)")
        namelist = zf.namelist()
        print(f"File Count:    {len(namelist)} total entries")
        
        # 2. Check essential files inside archive
        essential = [
            "fl_benchmark/README.md",
            "fl_benchmark/REPRODUCIBILITY.md",
            "fl_benchmark/RESEARCH_SUMMARY.pdf",
            "fl_benchmark/FINAL_SUBMISSION_READINESS_REPORT.md",
            "fl_benchmark/requirements.txt",
            "fl_benchmark/pyproject.toml",
            "fl_benchmark/run_benchmark.py",
            "fl_benchmark/figures/fig1_val_accuracy_vs_rounds.png",
            "fl_benchmark/figures/fig2_val_loss_vs_rounds.png",
            "fl_benchmark/figures/fig3_update_norm_shrinkage.png",
            "fl_benchmark/figures/fig4_update_cosine_alignment.png",
            "fl_benchmark/figures/fig5_pairwise_cosine_heatmaps.png",
            "fl_benchmark/figures/fig6_per_class_test_accuracy.png",
            "fl_benchmark/figures/figure_source_data_mapping.json",
            "fl_benchmark/results/primary_benchmark/matrix_execution_summary.json",
            "fl_benchmark/results/primary_benchmark/comparative_analysis.json",
            "fl_benchmark/results/primary_benchmark/detailed_table.json",
        ]
        
        for run_name in [
            "fedavg_dir0.3_noise0.4_seed42",
            "fedprox_mu0.01_dir0.3_noise0.4_seed42",
            "fedavg_dir0.3_noise0.4_seed101",
            "fedprox_mu0.01_dir0.3_noise0.4_seed101",
            "fedavg_dir0.3_noise0.4_seed2024",
            "fedprox_mu0.01_dir0.3_noise0.4_seed2024",
        ]:
            essential.append(f"fl_benchmark/results/primary_benchmark/{run_name}/summary.json")
            essential.append(f"fl_benchmark/results/primary_benchmark/{run_name}/metrics.jsonl")
            essential.append(f"fl_benchmark/results/primary_benchmark/{run_name}/checkpoints/best_checkpoint.pt")
            essential.append(f"fl_benchmark/results/primary_benchmark/{run_name}/checkpoints/initial_checkpoint.pt")
            essential.append(f"fl_benchmark/results/primary_benchmark/{run_name}/checkpoints/last_checkpoint.pt")
            
        missing_in_zip = [f for f in essential if f not in namelist]
        if missing_in_zip:
            print(f"Missing essential files in ZIP ({len(missing_in_zip)}):")
            for m in missing_in_zip:
                print(f"  - {m}")
            return False
        else:
            print(f"Essential Inventory: ALL {len(essential)} core research & provenance assets present in archive.")

    # 3. Verify README relative image links
    print("\n--- Image Link Verification in README.md ---")
    readme_path = PROJECT_ROOT / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    
    img_matches = re.findall(r'src=["\']([^"\']+)["\']', readme_text)
    for img_rel in img_matches:
        img_full = PROJECT_ROOT / img_rel
        if not img_full.exists():
            print(f"BROKEN LINK: {img_rel}")
            return False
        else:
            print(f"  Image OK: {img_rel} ({img_full.stat().st_size:,} bytes)")

    # 4. Secret Scan
    print("\n--- Secret & Credential Scan ---")
    secret_patterns = [
        re.compile(r"wandb_[a-zA-Z0-9]{30,}"),
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
        re.compile(r"""api[_-]?key\s*[:=]\s*["'][a-zA-Z0-9_\-]{16,}""", re.IGNORECASE),
        re.compile(r"""bearer\s+[a-zA-Z0-9_\-\.]{20,}""", re.IGNORECASE),
    ]
    secrets_found = []
    for p in PROJECT_ROOT.rglob("*"):
        if p.is_file() and p.suffix not in [".pt", ".npz", ".png", ".pdf", ".zip"]:
            if any(skip in p.parts for skip in ["fl_env", ".pytest_cache", "__pycache__", ".matplotlib"]):
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for pat in secret_patterns:
                    matches = pat.findall(content)
                    if matches:
                        secrets_found.append((p.relative_to(PROJECT_ROOT), matches))
            except Exception:
                pass
                
    if secrets_found:
        print(f"CRITICAL: Secrets found ({len(secrets_found)})!")
        for path, match in secrets_found:
            print(f"  {path}: {match}")
        return False
    else:
        print("Secrets scan: 0 secrets found across all repository files.")
        
    print("\n" + "=" * 80)
    print("ALL LOCAL PACKAGE VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    verify_all()
