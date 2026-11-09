import os
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
secret_patterns = [
    re.compile(r"wandb_[a-zA-Z0-9]{30,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    re.compile(r"""api[_-]?key\s*[:=]\s*["'][a-zA-Z0-9_\-]{16,}""", re.IGNORECASE),
    re.compile(r"""bearer\s+[a-zA-Z0-9_\-\.]{20,}""", re.IGNORECASE),
]

findings = []
for p in root.rglob("*"):
    if p.is_file() and p.suffix not in [".pt", ".npz", ".png", ".pdf"]:
        if any(skip in p.parts for skip in ["fl_env", ".pytest_cache", "__pycache__", ".matplotlib"]):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for pat in secret_patterns:
                    matches = pat.findall(content)
                    if matches:
                        findings.append((p.relative_to(root), matches))
        except Exception as e:
            pass

print(f"Secret scan complete. Potential secrets found: {len(findings)}")
for path, match in findings:
    print(f"  {path}: {match}")
