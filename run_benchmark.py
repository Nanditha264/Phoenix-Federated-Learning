"""
Benchmark Execution Entry Point.
Coordinates end-to-end execution of FedAvg and FedProx experiments on PathMNIST.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root and workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fl_benchmark.utils.seed import set_seed
from fl_benchmark.utils.device import setup_device, get_cpu_info
from fl_benchmark.utils.configuration import load_yaml, deep_merge
from fl_benchmark.utils.resource_monitor import FederatedResourceMonitor
from fl_benchmark.utils.checkpoint import CheckpointManager
from fl_benchmark.data.download import ensure_pathmnist_downloaded
from fl_benchmark.data.pathmnist_dataset import load_pathmnist_raw, PathMNISTDataset
from fl_benchmark.data.partition import dirichlet_partition, compute_partition_stats
from fl_benchmark.data.noise import inject_symmetric_noise
from fl_benchmark.data.validation import partition_validation_set
from fl_benchmark.models.model_factory import create_model
from fl_benchmark.federated.fedavg import FedAvgClient
from fl_benchmark.federated.fedprox import FedProxClient
from fl_benchmark.federated.server import FederatedServer
from fl_benchmark.tracking.local_logger import LocalLogger
from fl_benchmark.tracking.wandb_logger import WandbLogger
from fl_benchmark.evaluation.evaluator import GlobalEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg vs FedProx on PathMNIST")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to base YAML config"
    )
    parser.add_argument(
        "--override_config", type=str, default=None, help="Path to algorithm override YAML"
    )
    parser.add_argument(
        "--algorithm", type=str, default=None, choices=["fedavg", "fedprox"], help="Algorithm override"
    )
    parser.add_argument(
        "--mu", type=float, default=None, help="FedProx mu regularizer coefficient"
    )
    parser.add_argument(
        "--rounds", type=int, default=None, help="Communication rounds override"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed override"
    )
    parser.add_argument(
        "--smoke_test", action="store_true", help="Run 1 communication round with minimal samples for testing"
    )
    parser.add_argument(
        "--eval_test", action="store_true", help="Evaluate final test split on best checkpoint"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Explicit run name override"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Base output directory for results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configurations
    base_cfg_path = PROJECT_ROOT / args.config
    base_config = load_yaml(base_cfg_path)

    if args.override_config:
        override_cfg_path = PROJECT_ROOT / args.override_config
        override_config = load_yaml(override_cfg_path)
        config = deep_merge(base_config, override_config)
    else:
        config = base_config

    # Apply CLI overrides
    if args.algorithm:
        config["federated"]["algorithm"] = args.algorithm.lower()
    if args.mu is not None:
        config["federated"]["fedprox_mu"] = float(args.mu)
    if args.rounds is not None:
        config["federated"]["rounds"] = int(args.rounds)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.output_dir:
        config["tracking"]["output_dir"] = args.output_dir

    if args.smoke_test:
        config["federated"]["rounds"] = 1
        config["federated"]["local_epochs"] = 1

    algorithm = config["federated"]["algorithm"].lower()
    seed = config["experiment"]["seed"]
    if args.run_name:
        run_name = args.run_name
    elif algorithm == "fedprox":
        mu_val = config["federated"].get("fedprox_mu", 0.01)
        run_name = f"fedprox_mu{mu_val}_dir{config['partition']['dirichlet_alpha']}_noise{config['noise']['noise_rate']}_seed{seed}"
    else:
        run_name = f"{algorithm}_dir{config['partition']['dirichlet_alpha']}_noise{config['noise']['noise_rate']}_seed{seed}"

    print(f"\n========================================================")
    print(f"Federated Learning Benchmark: FedAvg vs FedProx on PathMNIST")
    print(f"Algorithm: {algorithm.upper()} | Seed: {seed} | Run: {run_name}")
    print(f"========================================================\n")

    # 2. Setup Device and Reproducibility
    set_seed(seed=seed, deterministic_algorithms=config["experiment"].get("deterministic", False))
    device = setup_device(
        num_threads=config["compute"]["num_threads"],
        num_interop_threads=config["compute"]["num_interop_threads"],
    )
    cpu_info = get_cpu_info()
    print(f"[Compute] Active Threads: {cpu_info['torch_num_threads']} (Intra-op), Device: {device}")

    # 3. Resource Monitor
    resource_monitor = FederatedResourceMonitor(
        warning_threshold_mb=config["safety"]["warning_threshold_mb"],
        client_pause_threshold_mb=config["safety"]["client_pause_threshold_mb"],
        emergency_stop_threshold_mb=config["safety"]["emergency_stop_threshold_mb"],
    )

    # 4. Data Pipeline
    cache_dir = config["dataset"]["cache_dir"]
    npz_path, md5_hash = ensure_pathmnist_downloaded(cache_dir)
    print(f"[Dataset] Verified PathMNIST MD5: {md5_hash}")

    raw_data = load_pathmnist_raw(npz_path)
    train_images = raw_data["train_images"]
    train_labels = raw_data["train_labels"]
    val_images = raw_data["val_images"]
    val_labels = raw_data["val_labels"]
    test_images = raw_data["test_images"]
    test_labels = raw_data["test_labels"]

    # Dirichlet Partition & Noise (Canonical File or Dynamic Generation)
    num_clients = config["partition"]["num_clients"]
    alpha = config["partition"]["dirichlet_alpha"]
    partition_file = PROJECT_ROOT / "data" / "partitions" / f"partition_dir{alpha}_clients{num_clients}_seed{seed}.npz"
    canonical_noise_labels = None

    if partition_file.exists():
        print(f"[Partition] Loading verified canonical partition: {partition_file.name}")
        part_data = np.load(partition_file)
        client_indices = [part_data[f"client_{i}"] for i in range(num_clients)]
        canonical_noise_labels = part_data["c0_noisy_labels"]
    else:
        client_indices = dirichlet_partition(
            labels=train_labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_client=config["partition"]["min_samples_per_client"],
        )

    p_stats = compute_partition_stats(train_labels, client_indices)
    print(f"[Partition] Client Sample Allocations: {p_stats['client_sample_counts']}")

    # If smoke test, subsample partitioned indices to allow instant verification
    if args.smoke_test:
        print("[Smoke Test] Subsampling to 128 samples per client for rapid end-to-end verification...")
        client_indices = [c_idxs[:128] for c_idxs in client_indices]

    # Controlled Label Noise for Client 0
    noisy_cid = config["noise"]["noisy_client_id"]
    noise_rate = config["noise"]["noise_rate"]
    client_datasets = []

    for cid in range(num_clients):
        c_idxs = client_indices[cid]
        c_labels = train_labels[c_idxs]

        if cid == noisy_cid and noise_rate > 0.0:
            if canonical_noise_labels is not None and not args.smoke_test:
                c_noisy = canonical_noise_labels
                n_corrupt = int(round(noise_rate * len(c_labels)))
                print(f"[Noise] Client {cid}: Loaded canonical noise ({n_corrupt}/{len(c_labels)} samples, {noise_rate*100:.2f}%)")
            else:
                c_noisy, c_clean, mask, audit = inject_symmetric_noise(
                    labels=c_labels,
                    noise_rate=noise_rate,
                    num_classes=config["dataset"]["num_classes"],
                    seed=config["noise"]["seed"] + cid,
                )
                print(
                    f"[Noise] Client {cid}: Corrupted {audit['corrupted_samples']}/{audit['total_samples']} "
                    f"samples ({audit['actual_noise_pct']}%)"
                )
            c_ds = PathMNISTDataset(
                images=train_images,
                labels=c_labels,
                indices=c_idxs,
                custom_labels=c_noisy,
            )
        else:
            c_ds = PathMNISTDataset(
                images=train_images,
                labels=c_labels,
                indices=c_idxs,
            )
        client_datasets.append(c_ds)

    # Validation Splitting (5 disjoint subsets)
    val_client_indices = partition_validation_set(val_labels, num_clients=num_clients, seed=seed)
    if args.smoke_test:
        client_val_loaders = [
            DataLoader(
                PathMNISTDataset(images=val_images, labels=val_labels, indices=idxs[:64]),
                batch_size=64,
                shuffle=False,
            )
            for idxs in val_client_indices
        ]
        global_val_loader = DataLoader(
            PathMNISTDataset(images=val_images, labels=val_labels, indices=np.arange(256)),
            batch_size=64,
            shuffle=False,
        )
    else:
        client_val_loaders = [
            DataLoader(
                PathMNISTDataset(images=val_images, labels=val_labels, indices=idxs),
                batch_size=64,
                shuffle=False,
            )
            for idxs in val_client_indices
        ]
        global_val_loader = DataLoader(
            PathMNISTDataset(images=val_images, labels=val_labels),
            batch_size=64,
            shuffle=False,
        )

    # 5. Model Initialization (Bitwise Canonical for paired comparisons)
    init_model_dir = PROJECT_ROOT / "data" / "initial_models"
    init_model_dir.mkdir(parents=True, exist_ok=True)
    canonical_model_file = init_model_dir / f"initial_model_seed{seed}.pt"

    global_model = create_model(
        model_name=config["model"]["name"],
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        fc_hidden=config["model"]["fc_hidden"],
    ).to(device)

    if canonical_model_file.exists():
        print(f"[Model] Loading verified canonical initial weights: {canonical_model_file.name}")
        global_model.load_state_dict(torch.load(canonical_model_file, map_location=device))
    else:
        torch.save(global_model.state_dict(), canonical_model_file)
        print(f"[Model] Registered canonical initial weights: {canonical_model_file.name}")

    param_info = global_model.count_parameters()
    print(f"[Model] Initialized {config['model']['name']} (Params: {param_info['total_parameters']:,})")

    # 6. Instantiate Clients
    clients = []
    batch_size = config["federated"]["batch_size"]
    lr = config["federated"]["learning_rate"]
    momentum = config["federated"]["momentum"]
    weight_decay = config["federated"]["weight_decay"]
    local_epochs = config["federated"]["local_epochs"]

    for cid in range(num_clients):
        if algorithm == "fedavg":
            client = FedAvgClient(
                cid=cid,
                dataset=client_datasets[cid],
                batch_size=batch_size,
                learning_rate=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                local_epochs=local_epochs,
                device=device,
                is_noisy=(cid == noisy_cid),
            )
        elif algorithm == "fedprox":
            mu = config["federated"].get("fedprox_mu", 0.01)
            client = FedProxClient(
                cid=cid,
                dataset=client_datasets[cid],
                mu=mu,
                batch_size=batch_size,
                learning_rate=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                local_epochs=local_epochs,
                device=device,
                is_noisy=(cid == noisy_cid),
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        clients.append(client)

    # 7. Setup Loggers & Checkpoint Manager
    output_dir = Path(config["tracking"]["output_dir"])
    local_logger = LocalLogger(output_dir=str(output_dir), run_name=run_name)
    wandb_logger = WandbLogger(
        project=config["experiment"]["project"],
        run_name=run_name,
        config=config,
        mode=config["tracking"]["wandb_mode"],
        output_dir=str(output_dir),
        enabled=config["tracking"]["use_wandb"],
    )
    checkpoint_dir = output_dir / run_name / "checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

    # Save initial model checkpoint (Round 0)
    checkpoint_manager.save_atomic(
        payload={"round": 0, "state_dict": global_model.state_dict(), "config": config},
        target_path=checkpoint_dir / "initial_checkpoint.pt",
    )

    # 8. Server Orchestration
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        val_loader=global_val_loader,
        client_val_loaders=client_val_loaders,
        total_rounds=config["federated"]["rounds"],
        checkpoint_manager=checkpoint_manager,
        local_logger=local_logger,
        wandb_logger=wandb_logger,
        resource_monitor=resource_monitor,
        device=device,
        config=config,
    )

    results = server.run()

    # 9. Optional Final Test Evaluation
    if args.eval_test:
        print(f"\n[Test Split Evaluation] Loading best checkpoint from {checkpoint_dir / 'best_checkpoint.pt'}...")
        best_ckpt = checkpoint_dir / "best_checkpoint.pt"
        if best_ckpt.exists():
            checkpoint_manager.load_checkpoint(best_ckpt, global_model)
        else:
            print("[WARN] Best checkpoint not found; evaluating current global model on test set.")

        test_loader = DataLoader(
            PathMNISTDataset(images=test_images, labels=test_labels),
            batch_size=64,
            shuffle=False,
        )
        test_evaluator = GlobalEvaluator(device=device)
        test_metrics = test_evaluator.evaluate(global_model, test_loader)

        print(f"[Test Split Results] Samples: {test_metrics['total_samples']:,}")
        print(f"                     Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
        print(f"                     Test Loss:     {test_metrics['loss']:.4f}")
        print(f"                     Test Macro-F1: {test_metrics['macro_f1']:.4f}")

        results["test_evaluation"] = test_metrics
        local_logger.log_metrics(step=config["federated"]["rounds"] + 1, metrics={"test": test_metrics})

    # Save Summary
    local_logger.save_summary(results)
    wandb_logger.finish()
    print(f"\nBenchmark run '{run_name}' finalized successfully. Artifacts in {output_dir / run_name}")


if __name__ == "__main__":
    main()
