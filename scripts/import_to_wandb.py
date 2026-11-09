"""
Dedicated Post-Hoc Weights & Biases Telemetry Importer.
Imports verified empirical metrics from the 6 primary benchmark runs into W&B
without rerunning training or mutating existing local artifacts.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_DIR = PROJECT_ROOT / "results" / "primary_benchmark"

RUNS = [
    {
        "name": "fedavg_dir0.3_noise0.4_seed42",
        "algorithm": "fedavg",
        "seed": 42,
        "mu": 0.0,
    },
    {
        "name": "fedprox_mu0.01_dir0.3_noise0.4_seed42",
        "algorithm": "fedprox",
        "seed": 42,
        "mu": 0.01,
    },
    {
        "name": "fedavg_dir0.3_noise0.4_seed101",
        "algorithm": "fedavg",
        "seed": 101,
        "mu": 0.0,
    },
    {
        "name": "fedprox_mu0.01_dir0.3_noise0.4_seed101",
        "algorithm": "fedprox",
        "seed": 101,
        "mu": 0.01,
    },
    {
        "name": "fedavg_dir0.3_noise0.4_seed2024",
        "algorithm": "fedavg",
        "seed": 2024,
        "mu": 0.0,
    },
    {
        "name": "fedprox_mu0.01_dir0.3_noise0.4_seed2024",
        "algorithm": "fedprox",
        "seed": 2024,
        "mu": 0.01,
    },
]


def check_wandb_auth():
    """
    Check if W&B authentication is present.
    Returns (is_authenticated, viewer_or_error).
    """
    try:
        import wandb
        if os.environ.get("WANDB_API_KEY"):
            return True, "API key detected in environment"
        api = wandb.Api()
        viewer = api.viewer
        return True, getattr(viewer, "entity", "authenticated_user")
    except Exception as e:
        return False, str(e)


def check_existing_wandb_runs(project: str, entity: str = None):
    """
    Query W&B project to find existing runs and prevent duplicates.
    Returns dict of {run_name: run_object}.
    """
    import wandb
    existing = {}
    try:
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path)
        for r in runs:
            existing[r.name] = r
            existing[r.id] = r
    except Exception:
        pass
    return existing


def import_run(run_info: dict, project: str, entity: str = None, offline: bool = False):
    """
    Import a single benchmark run's telemetry into W&B.
    """
    import wandb

    run_name = run_info["name"]
    run_dir = PRIMARY_DIR / run_name
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"

    assert summary_path.exists(), f"Missing summary: {summary_path}"
    assert metrics_path.exists(), f"Missing metrics: {metrics_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Standard configuration metadata
    config = {
        "experiment/project": project,
        "experiment/name": run_name,
        "algorithm": run_info["algorithm"],
        "federated/algorithm": run_info["algorithm"],
        "federated/rounds": 5,
        "federated/clients": 5,
        "federated/sample_ratio": 1.0,
        "federated/local_epochs": 2,
        "federated/mu": run_info["mu"],
        "federated/learning_rate": 0.01,
        "federated/momentum": 0.9,
        "federated/batch_size": 32,
        "partition/dirichlet_alpha": 0.3,
        "partition/num_clients": 5,
        "partition/noise_rate": 0.4,
        "partition/noisy_client_id": 0,
        "experiment/seed": run_info["seed"],
        "dataset/name": "PathMNIST",
        "dataset/train_samples": 89996,
        "dataset/val_samples": 10004,
        "dataset/test_samples": 7180,
        "compute/device": "cpu",
        "compute/num_threads": 4,
    }

    mode = "offline" if offline else "online"
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        id=f"pathmnist_{run_name}",
        resume="allow",
        config=config,
        mode=mode,
        reinit=True,
    )

    print(f"\n[W&B Import] Run: {run_name} (ID: {run.id})")

    # Step-by-step telemetry replay
    for rec in records:
        step = rec.get("step")
        if "round" in rec and rec["round"] is not None:
            # Round-level telemetry
            flat_metrics = {
                "round": rec["round"],
                "train/avg_loss": rec.get("train/avg_loss"),
                "eval/val_loss": rec.get("eval/val_loss"),
                "eval/val_acc": rec.get("eval/val_acc"),
                "eval/val_macro_f1": rec.get("eval/val_macro_f1"),
                "eval/best_val_acc": rec.get("eval/best_val_acc"),
                "eval/client_mean_val_acc": rec.get("eval/client_mean_val_acc"),
                "eval/client_mean_val_loss": rec.get("eval/client_mean_val_loss"),
                "geometry/global_update_norm": rec.get("geometry/global_update_norm"),
                "system/process_rss_mb": rec.get("system/process_rss_mb"),
                "system/ram_available_mb": rec.get("system/ram_available_mb"),
                "system/cpu_percent": rec.get("system/cpu_percent"),
                "time/round_seconds": rec.get("time/round_seconds"),
            }
            # Flatten client update norms
            c_norms = rec.get("geometry/client_update_norms", [])
            for c_id, n_val in enumerate(c_norms):
                flat_metrics[f"geometry/client_update_norm/client_{c_id}"] = n_val
            # Flatten alignments
            c_aligns = rec.get("geometry/alignment_with_global", [])
            for c_id, a_val in enumerate(c_aligns):
                flat_metrics[f"geometry/alignment_with_global/client_{c_id}"] = a_val

            run.log(flat_metrics, step=step)
            print(f"  - Logged Round {rec['round']}: Val Acc = {rec.get('eval/val_acc')*100:.2f}%, Global Norm = {rec.get('geometry/global_update_norm'):.4f}")

        elif "test" in rec:
            # Final test split evaluation record
            test_data = rec["test"]
            test_metrics = {
                "test/accuracy": test_data.get("accuracy"),
                "test/loss": test_data.get("loss"),
                "test/macro_f1": test_data.get("macro_f1"),
                "test/total_samples": test_data.get("total_samples"),
            }
            for c_id, acc_val in test_data.get("per_class_accuracy", {}).items():
                test_metrics[f"test/per_class_accuracy/class_{c_id}"] = acc_val

            run.log(test_metrics, step=step)
            print(f"  - Logged Final Test Split: Accuracy = {test_data.get('accuracy')*100:.2f}%, Macro-F1 = {test_data.get('macro_f1'):.4f}")

    # Final summary registration
    te = summary_data.get("test_evaluation", {})
    run.summary["best_val_acc"] = summary_data.get("best_val_acc")
    run.summary["rounds_completed"] = summary_data.get("rounds_completed")
    run.summary["test/accuracy"] = te.get("accuracy")
    run.summary["test/loss"] = te.get("loss")
    run.summary["test/macro_f1"] = te.get("macro_f1")
    run.summary["test/class_8_adenocarcinoma_acc"] = te.get("per_class_accuracy", {}).get("8")

    run_url = getattr(run, "url", "offline_run")
    run.finish()
    print(f"  - Finished run: {run_url}")
    return run_url


def main():
    parser = argparse.ArgumentParser(description="Post-Hoc W&B Importer for PathMNIST FL Benchmark")
    parser.add_argument("--project", default="fedavg-vs-fedprox-pathmnist", help="W&B Project Name")
    parser.add_argument("--entity", default=None, help="W&B Entity / Team Name")
    parser.add_argument("--offline", action="store_true", help="Log runs locally in W&B offline directory")
    parser.add_argument("--force", action="store_true", help="Force re-import even if runs already exist in project")
    args = parser.parse_args()

    print("=" * 80)
    print("WEIGHTS & BIASES POST-HOC TELEMETRY IMPORTER")
    print(f"Target Project: {args.project}")
    print("=" * 80)

    is_auth, info = check_wandb_auth()
    if not is_auth and not args.offline:
        print("\n[AUTHENTICATION NOTICE]")
        print("Weights & Biases authentication credentials were not detected in this environment.")
        print("Reason:", info)
        print("\nTo authenticate and synchronize your results to the W&B cloud:")
        print("  1. In your external terminal, run: wandb login")
        print("  2. When prompted, paste your API key from: https://wandb.ai/authorize")
        print("  3. Then re-run this importer:")
        print(f"     python scripts/import_to_wandb.py --project {args.project}")
        print("\nAlternative (Offline Local W&B Directory):")
        print(f"     python scripts/import_to_wandb.py --project {args.project} --offline")
        print("=" * 80)
        sys.exit(2)

    print(f"Authentication verified: {info}")
    
    existing_runs = {}
    if not args.offline:
        existing_runs = check_existing_wandb_runs(project=args.project, entity=args.entity)
        if existing_runs:
            print(f"Found {len(existing_runs)} existing runs in project '{args.project}'.")

    run_urls = []
    for run_info in RUNS:
        run_name = run_info["name"]
        run_id = f"pathmnist_{run_name}"
        if not args.force and (run_name in existing_runs or run_id in existing_runs):
            print(f"[SKIP] Run '{run_name}' already exists in W&B project. Skipping to prevent duplicate records.")
            ex = existing_runs.get(run_name, existing_runs.get(run_id))
            run_urls.append((run_name, getattr(ex, "url", "existing_run")))
            continue
            
        url = import_run(run_info, project=args.project, entity=args.entity, offline=args.offline)
        run_urls.append((run_info["name"], url))

    print("\n" + "=" * 80)
    print("ALL 6 RUNS SUCCESSFULLY IMPORTED TO WEIGHTS & BIASES!")
    print("=" * 80)
    for name, url in run_urls:
        print(f"  {name:42s} -> {url}")


if __name__ == "__main__":
    main()
