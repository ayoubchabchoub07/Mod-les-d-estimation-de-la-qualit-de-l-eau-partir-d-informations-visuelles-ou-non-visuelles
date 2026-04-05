#!/usr/bin/env python3
"""
PHASE 3 : ENTRAÎNEMENT FINAL DU MODÈLE

Configuration finale issue de l'ablation complète (run_complete_study.py) :
  - Meilleure config Phase 2 : 2_with_weak_DO  (MAE=0.2366 NTU en 10 epochs)
  - turbidity_NTU = 1.0  (signal principal, CV=24.7%)
  - DO_mgL        = 0.2  (régularisateur léger, CV=5.85%)
  - pH            = 0.0  (désactivé, CV=0.06% — aucun signal visuel)
  - temperature_C = 0.0  (désactivé, CV=0.67% — aucun signal visuel)

HYPERPARAMÈTRES :
  - Batch Size   : 32
  - Epochs       : 20  (convergence complète avec CosineAnnealingLR)
  - Learning Rate: 1e-4 (conservative pour backbone pretrained)
  - Weight Decay : 1e-4 (L2 regularization légère)
  - Optimizer    : AdamW
  - Scheduler    : CosineAnnealingLR (T_max=30, eta_min=1e-6)
  - Dropout      : 0.3

SORTIES :
  - checkpoints/best_model.pth      ← meilleur modèle (val_loss minimale)
  - checkpoints/last_model.pth      ← dernier epoch
  - checkpoints/training_history.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from preprocess import create_preprocessor
from dataset import create_dataloaders
from model import create_model
from loss import create_loss_function


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    """Entraîne le modèle sur une epoch complète."""
    model.train()
    total_loss = 0.0
    task_losses_sum = {task: 0.0 for task in ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]}

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d} [TRAIN]")
    for images, targets in pbar:
        images  = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        predictions = model(images)
        loss, task_losses = criterion(predictions, targets)

        loss.backward()
        # Gradient clipping — évite les explosions de gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for task, tl in task_losses.items():
            task_losses_sum[task] += tl

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader)
    return total_loss / n, {t: s / n for t, s in task_losses_sum.items()}


def validate(
    model: nn.Module,
    loader,
    criterion,
    device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    """Valide le modèle sur le split VAL."""
    model.eval()
    total_loss = 0.0
    task_losses_sum = {task: 0.0 for task in ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]}

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d} [VAL]  ")
        for images, targets in pbar:
            images  = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            predictions = model(images)
            loss, task_losses = criterion(predictions, targets)

            total_loss += loss.item()
            for task, tl in task_losses.items():
                task_losses_sum[task] += tl

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader)
    return total_loss / n, {t: s / n for t, s in task_losses_sum.items()}


def main() -> None:
    """Point d'entrée principal."""
    print("=" * 80)
    print("PHASE 3 : ENTRAÎNEMENT FINAL")
    print("Config gagnante ablation : turbidity=1.0, DO=0.2, pH=0.0, temp=0.0")
    print("=" * 80)

    # ── Configuration ──────────────────────────────────────────────────────────
    config = {
        "batch_size"   : 32,
        "num_epochs"   : 20,
        "learning_rate": 1e-4,
        "weight_decay" : 1e-4,
        # CORRECTION : num_workers=0 obligatoire sur Windows (multiprocessing incompatible)
        "num_workers"  : 0,
        "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    }

    # ── Config λ gagnante (ablation Phase 2 : 2_with_weak_DO) ─────────────────
    loss_config = {
        "lambdas": {
            "turbidity_NTU": 1.0,   # ← signal principal (CV=24.7%)
            "pH"           : 0.0,   # ← désactivé (CV=0.06%, aucun signal)
            "temperature_C": 0.0,   # ← désactivé (CV=0.67%, aucun signal)
            "DO_mgL"       : 0.2,   # ← régularisateur léger (CV=5.85%)
        },
        "delta": 1.0,
    }

    print("\n📋 Configuration :")
    for key, value in config.items():
        print(f"  {key:20s} : {value}")
    print("\n🎯 Poids λ :")
    for task, lam in loss_config["lambdas"].items():
        status = "✅ ACTIF" if lam > 0 else "❌ INACTIF"
        print(f"  {task:20s} : {lam}  {status}")

    # ── Chemins ────────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent
    csv_path     = project_root / "processed" / "images_labels.csv"

    # Recherche automatique du dossier dataset
    dataset_root = None
    for parent in [project_root, *project_root.parents]:
        candidate = parent / "Tilapia RAS Dataset"
        if candidate.exists():
            dataset_root = candidate
            break
    if dataset_root is None:
        dataset_root = project_root.parent / "Tilapia RAS Dataset"
        print(f"⚠️  Dataset non trouvé automatiquement, tentative : {dataset_root}")

    output_dir = project_root / "checkpoints"
    output_dir.mkdir(exist_ok=True)

    device = torch.device(config["device"])
    print(f"\n💻 Device : {device}")

    # ── Données ────────────────────────────────────────────────────────────────
    print("\n1️⃣  Création du preprocessor...")
    preprocessor = create_preprocessor()

    print("\n2️⃣  Création des dataloaders...")
    dataloaders = create_dataloaders(
        csv_path     = csv_path,
        dataset_root = dataset_root,
        preprocessor = preprocessor,
        batch_size   = config["batch_size"],
        # CORRECTION : pin_memory dynamique (True=GPU, False=CPU)
        num_workers  = config["num_workers"],
    )
    train_loader = dataloaders["TRAIN"]
    val_loader   = dataloaders["VAL"]
    print(f"   TRAIN : {len(train_loader.dataset)} images")
    print(f"   VAL   : {len(val_loader.dataset)} images")

    # ── Modèle ─────────────────────────────────────────────────────────────────
    print("\n3️⃣  Création du modèle EfficientNet-B2...")
    model = create_model({
        "pretrained"      : True,
        "dropout_rate"    : 0.3,
        "freeze_backbone" : False,
    })
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Paramètres entraînables : {n_params:,}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    print("\n4️⃣  Création de la loss (config gagnante ablation)...")
    criterion = create_loss_function(loss_config)

    # ── Optimiseur ─────────────────────────────────────────────────────────────
    print("\n5️⃣  Création de l'optimiseur...")
    optimizer = AdamW(
        model.parameters(),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-6)

    print(f"   AdamW — lr={config['learning_rate']}, wd={config['weight_decay']}")
    print(f"   CosineAnnealingLR — T_max={config['num_epochs']}, eta_min=1e-6")

    # ── Boucle d'entraînement ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*80}\n")

    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 10  # arrêt si pas d'amélioration pendant 10 epochs

    history: Dict = {
        "train_loss"      : [],
        "val_loss"        : [],
        "train_task_losses": {t: [] for t in ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]},
        "val_task_losses"  : {t: [] for t in ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]},
        "learning_rates"  : [],
        "best_epoch"      : 0,
    }

    for epoch in range(config["num_epochs"]):

        train_loss, train_task_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_task_losses = validate(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Enregistrement historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(current_lr)
        for task in ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]:
            history["train_task_losses"][task].append(train_task_losses.get(task, 0.0))
            history["val_task_losses"][task].append(val_task_losses.get(task, 0.0))

        # Ratio overfitting
        ratio = val_loss / train_loss if train_loss > 0 else 0
        ratio_flag = "⚠️  OVERFITTING" if ratio > 3.0 else "✅"

        print(f"\n📊 Epoch {epoch+1:02d}/{config['num_epochs']} — Résumé :")
        print(f"   Train Loss : {train_loss:.4f}")
        print(f"   Val Loss   : {val_loss:.4f}   (ratio {ratio:.1f}×  {ratio_flag})")
        print(f"   LR         : {current_lr:.2e}")

        # Losses par tâche (seulement les tâches actives)
        active_tasks = [t for t, lam in loss_config["lambdas"].items() if lam > 0]
        print("   Losses tâches actives :")
        for task in active_tasks:
            t_loss = train_task_losses.get(task, 0.0)
            v_loss = val_task_losses.get(task, 0.0)
            print(f"     {task:20s} : Train={t_loss:.4f}  Val={v_loss:.4f}")

        # Sauvegarde meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            history["best_epoch"] = epoch + 1
            torch.save(
                {
                    "epoch"            : epoch,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss"         : val_loss,
                    "config"           : config,
                    "loss_config"      : loss_config,
                },
                output_dir / "best_model.pth",
            )
            print(f"   ✅ BEST sauvegardé  →  val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   (pas d'amélioration — patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")

        # Sauvegarde dernier epoch (toujours)
        torch.save(
            {
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"         : val_loss,
                "config"           : config,
                "loss_config"      : loss_config,
            },
            output_dir / "last_model.pth",
        )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping à epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
            break

        print(f"{'='*80}")

    # ── Fin ────────────────────────────────────────────────────────────────────
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*80}")
    print(f"\n📁 Fichiers sauvegardés :")
    print(f"   Meilleur modèle : {output_dir / 'best_model.pth'}")
    print(f"   Dernier modèle  : {output_dir / 'last_model.pth'}")
    print(f"   Historique      : {history_path}")
    print(f"\n🏆 Meilleure val loss : {best_val_loss:.4f}  (epoch {history['best_epoch']})")
    print("\n💡 Prochaine étape : python evaluate.py")


if __name__ == "__main__":
    main()