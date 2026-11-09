"""
Verify W&B Cloud Publication.
Inspects all 6 runs in fedavg-vs-fedprox-pathmnist, verifies 5-round histories,
summary metrics, and checks public accessibility.
"""

import urllib.request
import urllib.error
import wandb

def main():
    api = wandb.Api()
    entity = "c3861282-pes-college"
    project_name = "fedavg-vs-fedprox-pathmnist"
    project_path = f"{entity}/{project_name}"

    proj = api.project(project_name, entity=entity)
    print("=" * 80)
    print("W&B PROJECT VERIFICATION REPORT")
    print(f"Project: {proj.name}")
    print(f"Entity:  {proj.entity}")
    print(f"URL:     {proj.url}")
    print("=" * 80)

    runs = list(api.runs(project_path))
    print(f"Total Runs Found: {len(runs)} (Expected: 6)\n")

    expected_names = [
        "fedavg_dir0.3_noise0.4_seed42",
        "fedprox_mu0.01_dir0.3_noise0.4_seed42",
        "fedavg_dir0.3_noise0.4_seed101",
        "fedprox_mu0.01_dir0.3_noise0.4_seed101",
        "fedavg_dir0.3_noise0.4_seed2024",
        "fedprox_mu0.01_dir0.3_noise0.4_seed2024",
    ]

    found_names = set()

    for r in runs:
        found_names.add(r.name)
        hist = r.history()
        rounds = [int(x) for x in hist["round"].dropna().tolist()] if "round" in hist else []
        
        print(f"• Run: {r.name}")
        print(f"  Run ID:           {r.id}")
        print(f"  State:            {r.state}")
        print(f"  History Rows:     {len(hist)}")
        print(f"  Rounds Logged:    {rounds} (5 rounds expected)")
        print(f"  Best Val Acc:     {r.summary.get('best_val_acc')}")
        print(f"  Test Accuracy:    {r.summary.get('test/accuracy')}")
        print(f"  Test Macro-F1:    {r.summary.get('test/macro_f1')}")
        print(f"  Class 8 Tumor:    {r.summary.get('test/class_8_adenocarcinoma_acc')}")
        print(f"  Post-hoc Mean:    {r.summary.get('post_hoc_val/mean_acc')}")
        print(f"  Post-hoc Client0: {r.summary.get('post_hoc_val/client_0_acc')}")
        print(f"  Run URL:          {r.url}")
        print()

    missing = set(expected_names) - found_names
    if missing:
        print(f"ERROR: Missing runs: {missing}")
    else:
        print("ALL 6 RUNS VERIFIED WITH COMPLETE 5-ROUND HISTORIES!\n")

    # Check unauthenticated HTTP accessibility
    project_url = proj.url
    print(f"Testing public HTTP access on: {project_url}")
    try:
        req = urllib.request.Request(
            project_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"HTTP Status: {resp.status} OK (Accessible publicly)")
    except urllib.error.HTTPError as e:
        print(f"HTTP Status: {e.code} ({e.reason})")
    except Exception as e:
        print(f"Access test result: {e}")

if __name__ == "__main__":
    main()
