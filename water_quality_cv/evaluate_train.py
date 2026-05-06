#!/usr/bin/env python3
"""
ÉVALUATION FINALE — TEST SET

À utiliser UNE SEULE FOIS, après avoir choisi le modèle final.
Le TEST SET ne doit jamais être utilisé pour choisir des hyperparamètres.

Usage :
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/last_model.pth

Sorties :
    test_results.json        ← métriques complètes
    scatter_turbidity.png    ← scatter plot pred vs réalité
    error_distribution.png   ← histogramme des erreurs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from preprocess import create_preprocessor
from dataset import create_dataloaders
from model import create_model


# ══════════════════════════════════════════════════════════════════════════════
# Calcul des métriques — R² CORRIGÉ
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, targets: np.ndarray, task_name: str) -> dict:
    """
    Calcule MAE, RMSE, R² et MAPE pour une tâche.

    Correction R² :
      - R² = 1 - SS_res / SS_tot
      - Si SS_tot ≈ 0 (variabilité quasi-nulle dans le set) → R² = NaN
      - Cela se produit quand le TEST set contient un seul clip vidéo
        avec des valeurs très proches (ex: turbidité ∈ [6.9, 7.1] NTU)
      - Dans ce cas, MAE et RMSE restent fiables — utiliser ces métriques.
    """
    # Filtrer les NaN dans les targets (labels manquants)
    mask = ~np.isnan(targets)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "mape": float("nan"), "n_samples": 0, "note": "Aucun label valide"}

    p = preds[mask]
    t = targets[mask]
    n = len(t)

    mae  = float(np.abs(p - t).mean())
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mape = float(np.mean(np.abs((t - p) / (np.abs(t) + 1e-8))) * 100)

    # ── R² corrigé ────────────────────────────────────────────────────────────
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))

    if ss_tot < 1e-8:
        # Variabilité quasi-nulle : R² non calculable
        # Cela se produit si le TEST set a un seul clip très homogène.
        r2   = float("nan")
        note = (f"R² non calculable — SS_tot={ss_tot:.2e} ≈ 0 "
                f"(écart-type targets = {t.std():.4f} {get_unit(task_name)}). "
                f"Utiliser MAE={mae:.4f} comme métrique principale.")
    else:
        r2_raw = 1.0 - ss_res / ss_tot
        # Clamp à -10 : R² < -10 n'a aucune signification pratique
        r2   = float(max(-10.0, r2_raw))
        note = ""
        if r2 < 0:
            note = (f"R² négatif ({r2:.3f}) : le modèle est moins précis que "
                    f"prédire la moyenne. MAE={mae:.4f} reste la métrique principale.")

    return {
        "mae"      : mae,
        "rmse"     : rmse,
        "r2"       : r2,
        "mape"     : mape,
        "n_samples": n,
        "ss_res"   : ss_res,
        "ss_tot"   : ss_tot,
        "target_std" : float(t.std()),
        "target_mean": float(t.mean()),
        "note"     : note,
    }


def get_unit(task: str) -> str:
    units = {"turbidity_NTU": "NTU", "pH": "", "temperature_C": "°C", "DO_mgL": "mg/L"}
    return units.get(task, "")


def json_safe(obj):
    """
    Convertit récursivement un objet pour le rendre sérialisable en JSON.
    - float('nan') / float('inf')  → None  (JSON standard ne supporte pas NaN)
    - Path (WindowsPath/PosixPath) → str
    - numpy scalaires              → float/int Python natif
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, 'item'):   # numpy scalar
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return obj


def interpret_turbidity(mae: float) -> str:
    if math.isnan(mae):
        return "Non calculable"
    if mae < 0.20:
        return "🏆 Exceptionnel  (< 0.20 NTU)"
    if mae < 0.50:
        return "✅ Excellent     (< 0.50 NTU)"
    if mae < 1.00:
        return "⚠️  Acceptable    (< 1.00 NTU)"
    return "❌ Insuffisant   (≥ 1.00 NTU)"


# ══════════════════════════════════════════════════════════════════════════════
# Graphiques
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(preds: np.ndarray, targets: np.ndarray,
                 metrics: dict, output_path: str) -> None:
    """Scatter plot prédictions vs réalité avec informations de diagnostic."""
    mask = ~np.isnan(targets)
    p, t = preds[mask], targets[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Évaluation — Turbidité (TEST SET)", fontsize=14, fontweight="bold")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(t, p, alpha=0.4, s=18, color="steelblue", label="Prédictions")

    lims = [min(t.min(), p.min()) - 0.3, max(t.max(), p.max()) + 0.3]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Prédiction parfaite")
    ax.fill_between(lims,
                    [l - metrics["mae"] for l in lims],
                    [l + metrics["mae"] for l in lims],
                    alpha=0.1, color="red", label=f"±MAE ({metrics['mae']:.3f} NTU)")

    r2_str = f"{metrics['r2']:.3f}" if not math.isnan(metrics["r2"]) else "N/A (SS_tot≈0)"
    ax.set_title(f"MAE={metrics['mae']:.4f} NTU   RMSE={metrics['rmse']:.4f} NTU\n"
                 f"R²={r2_str}   n={metrics['n_samples']}")
    ax.set_xlabel("Turbidité réelle (NTU)")
    ax.set_ylabel("Turbidité prédite (NTU)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Histogramme des erreurs ───────────────────────────────────────────────
    ax2 = axes[1]
    errors = p - t
    ax2.hist(errors, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(0,                       color="red",    linestyle="--", linewidth=2, label="Erreur nulle")
    ax2.axvline(errors.mean(),           color="orange", linestyle="-",  linewidth=1.5,
                label=f"Biais = {errors.mean():.3f} NTU")
    ax2.axvline( metrics["mae"],         color="green",  linestyle=":",  linewidth=1.5,
                label=f"+MAE = {metrics['mae']:.3f}")
    ax2.axvline(-metrics["mae"],         color="green",  linestyle=":",  linewidth=1.5,
                label=f"-MAE = {-metrics['mae']:.3f}")

    pct_within_05 = (np.abs(errors) < 0.5).mean() * 100
    pct_within_10 = (np.abs(errors) < 1.0).mean() * 100
    ax2.set_title(f"Distribution des erreurs\n"
                  f"{pct_within_05:.1f}% dans ±0.5 NTU   {pct_within_10:.1f}% dans ±1.0 NTU")
    ax2.set_xlabel("Erreur prédite − réelle (NTU)")
    ax2.set_ylabel("Nombre d'images")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Scatter sauvegardé : {output_path}")


def plot_error_by_range(preds: np.ndarray, targets: np.ndarray,
                        output_path: str) -> None:
    """
    MAE par plage de turbidité.
    Si le TEST set est homogène (std ≈ 0), génère à la place un graphique
    d'analyse de la distribution des erreurs avec percentiles.
    """
    mask = ~np.isnan(targets)
    p, t = preds[mask], targets[mask]
    errors = p - t
    abs_errors = np.abs(errors)

    # ── Cas spécial : std ≈ 0 (clip TEST homogène) ────────────────────────────
    if t.std() < 1e-4:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Analyse des erreurs — Turbidité TEST SET\n"
            f"(Clip homogène : toutes les images ont turbidité ≈ {t.mean():.2f} NTU)",
            fontsize=12, fontweight="bold"
        )

        # Graphique 1 : erreurs triées (révèle la distribution complète)
        ax = axes[0]
        sorted_idx = np.argsort(errors)
        ax.scatter(range(len(errors)), errors[sorted_idx],
                   alpha=0.4, s=8, color="steelblue")
        ax.axhline(0,              color="red",    linestyle="--", linewidth=1.5, label="Erreur = 0")
        ax.axhline( abs_errors.mean(), color="green", linestyle=":",  linewidth=1.5,
                   label=f"MAE = {abs_errors.mean():.3f} NTU")
        ax.axhline(-abs_errors.mean(), color="green", linestyle=":",  linewidth=1.5)
        p25, p75 = np.percentile(errors, 25), np.percentile(errors, 75)
        ax.fill_between([0, len(errors)], p25, p75, alpha=0.1, color="blue",
                        label=f"IQR [{p25:.2f}, {p75:.2f}] NTU")
        ax.set_title("Erreurs triées (pred − réel)")
        ax.set_xlabel("Images (triées par erreur croissante)")
        ax.set_ylabel("Erreur (NTU)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Graphique 2 : percentiles des erreurs absolues
        ax2 = axes[1]
        pcts = [50, 75, 90, 95, 99]
        pct_vals = [np.percentile(abs_errors, p) for p in pcts]
        colors_p = ["#4CAF50","#8BC34A","#FFC107","#FF9800","#F44336"]
        bars = ax2.bar([f"P{p}" for p in pcts], pct_vals, color=colors_p,
                       edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, pct_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.axhline(0.5, color="green", linestyle="--", linewidth=1.5,
                    label="Seuil excellent (0.5 NTU)")
        ax2.axhline(1.0, color="orange", linestyle="--", linewidth=1.5,
                    label="Seuil acceptable (1.0 NTU)")
        within_05 = (abs_errors < 0.5).mean() * 100
        within_10 = (abs_errors < 1.0).mean() * 100
        ax2.set_title(f"Percentiles des erreurs absolues\n"
                      f"{within_05:.1f}% < 0.5 NTU   |   {within_10:.1f}% < 1.0 NTU")
        ax2.set_xlabel("Percentile")
        ax2.set_ylabel("Erreur absolue (NTU)")
        ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   📊 Analyse erreurs (clip homogène) sauvegardée : {output_path}")
        return

    # ── Cas normal : std > 0 → MAE par quintile ────────────────────────────────
    percentiles = np.percentile(t, [0, 20, 40, 60, 80, 100])
    labels, mae_vals, counts = [], [], []

    for i in range(len(percentiles) - 1):
        lo, hi = percentiles[i], percentiles[i + 1]
        m = (t >= lo) & (t < hi) if i < 4 else (t >= lo) & (t <= hi)
        if m.sum() > 0:
            labels.append(f"[{lo:.1f}, {hi:.1f}]")
            mae_vals.append(abs_errors[m].mean())
            counts.append(m.sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3" if v < np.mean(mae_vals) else "#FF5722" for v in mae_vals]
    bars = ax.bar(labels, mae_vals, color=colors, edgecolor="white", alpha=0.85)

    for bar, count, val in zip(bars, counts, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"MAE={val:.3f}\nn={count}", ha="center", va="bottom", fontsize=9)

    ax.axhline(np.mean(mae_vals), color="red", linestyle="--",
               linewidth=1.5, label=f"MAE global = {np.mean(mae_vals):.3f} NTU")
    ax.set_title("MAE par plage de turbidité (TEST SET)\n"
                 "Bleu = sous la moyenne, Rouge = au-dessus")
    ax.set_xlabel("Plage de turbidité (NTU)")
    ax.set_ylabel("MAE (NTU)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 MAE par plage sauvegardé : {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation principale
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(checkpoint_path: str = "checkpoints/best_model.pth") -> dict:
    """Évaluation complète sur le TEST set."""

    print("=" * 80)
    print("ÉVALUATION FINALE — TEST SET")
    print("=" * 80)
    print(f"\n📂 Checkpoint : {checkpoint_path}")

    # ── Chemins ────────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent
    csv_path     = project_root / "processed" / "images_labels.csv"

    dataset_root = None
    for parent in [project_root, *project_root.parents]:
        candidate = parent / "Tilapia RAS Dataset"
        if candidate.exists():
            dataset_root = candidate
            break
    if dataset_root is None:
        dataset_root = project_root.parent / "Tilapia RAS Dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device : {device}")

    # ── Charger le modèle ──────────────────────────────────────────────────────
    print("\n1️⃣  Chargement du modèle...")
    # dropout_rate=0.0 pour l'inférence (model.eval() le fait aussi, mais explicite = plus sûr)
    model = create_model({"pretrained": False, "dropout_rate": 0.0})

    # Résoudre le chemin checkpoint : relatif à evaluate.py si pas absolu
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"\n❌ Checkpoint introuvable : {ckpt_path}"
            f"\n   Vérifier que l'entraînement (train.py) s'est terminé."
            f"\n   Fichiers attendus :"
            f"\n     {project_root / 'checkpoints' / 'best_model.pth'}"
            f"\n     {project_root / 'checkpoints' / 'last_model.pth'}"
        )
    print(f"   Checkpoint : {ckpt_path}")

    # weights_only=False nécessaire car le checkpoint contient des dicts Python
    # (config, loss_config) — pas seulement des tenseurs.
    # Risque nul ici car c'est notre propre fichier généré par train.py.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"   Chargé depuis epoch {ckpt.get('epoch', '?') + 1}, val_loss={ckpt.get('val_loss', '?'):.4f}")
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    # ── DataLoader TEST ────────────────────────────────────────────────────────
    print("\n2️⃣  Création du DataLoader TEST...")
    preprocessor = create_preprocessor()
    loaders = create_dataloaders(
        csv_path     = csv_path,
        dataset_root = dataset_root,
        preprocessor = preprocessor,
        batch_size   = 32,
        num_workers  = 0,  # Windows safe
    )
    test_loader = loaders["TEST"]
    print(f"   TEST : {len(test_loader.dataset)} images")

    # ── Collecte des prédictions ───────────────────────────────────────────────
    print("\n3️⃣  Inférence sur le TEST set...")
    TASKS = ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]

    all_preds   = {t: [] for t in TASKS}
    all_targets = {t: [] for t in TASKS}

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="TEST"):
            images = images.to(device)
            preds  = model(images)

            for task in TASKS:
                all_preds[task].append(preds[task].cpu().numpy())
                if task in targets:
                    all_targets[task].append(targets[task].cpu().numpy())
                else:
                    # Si le label n'est pas dans le batch, remplir de NaN
                    all_targets[task].append(
                        np.full(preds[task].shape, float("nan"))
                    )

    # Concaténer
    preds_np   = {t: np.concatenate(all_preds[t])   for t in TASKS}
    targets_np = {t: np.concatenate(all_targets[t]) for t in TASKS}

    # ── Diagnostic du TEST set ────────────────────────────────────────────────
    print("\n4️⃣  Diagnostic du TEST set...")
    turb_t = targets_np["turbidity_NTU"]
    mask_valid = ~np.isnan(turb_t)
    if mask_valid.sum() > 0:
        t_vals = turb_t[mask_valid]
        print(f"   Turbidité — min={t_vals.min():.4f}  max={t_vals.max():.4f}  "
              f"mean={t_vals.mean():.4f}  std={t_vals.std():.6f} NTU")
        if t_vals.std() < 1e-4:
            print("   ⚠️  ATTENTION : std ≈ 0 → toutes les images TEST ont la même")
            print("      turbidité. Le clip TEST est homogène (pas de variabilité).")
            print("      → R² sera NaN. MAE reste valide et fiable.")
            print("      → MAE par plage remplacé par analyse de distribution des erreurs.")
    for task in ["pH", "temperature_C", "DO_mgL"]:
        tv = targets_np[task]
        mv = ~np.isnan(tv)
        if mv.sum() > 0:
            vals = tv[mv]
            print(f"   {task:20s} — mean={vals.mean():.4f}  std={vals.std():.6f}  "
                  f"{'⚠️  constant' if vals.std() < 1e-6 else 'ok'}")

    # ── Calcul des métriques ───────────────────────────────────────────────────
    print("\n5️⃣  Calcul des métriques...")
    print("=" * 70)
    print(f"{'RÉSULTATS FINAUX — TEST SET':^70}")
    print("=" * 70)

    all_metrics = {}
    for task in TASKS:
        m = compute_metrics(preds_np[task], targets_np[task], task)
        all_metrics[task] = m
        unit = get_unit(task)

        r2_str = f"{m['r2']:.4f}" if not math.isnan(m["r2"]) else "N/A"
        print(f"\n  🎯 {task}")
        print(f"     MAE   = {m['mae']:.4f} {unit}")
        print(f"     RMSE  = {m['rmse']:.4f} {unit}")
        print(f"     R²    = {r2_str}")
        print(f"     MAPE  = {m['mape']:.2f} %")
        print(f"     n     = {m['n_samples']}")
        if m["note"]:
            print(f"     ⚠️   {m['note']}")

    # Interprétation turbidité
    turb_mae = all_metrics["turbidity_NTU"]["mae"]
    print(f"\n{'─'*70}")
    print(f"  📌 Turbidité : {interpret_turbidity(turb_mae)}")
    print(f"{'─'*70}")

    # ── Graphiques ─────────────────────────────────────────────────────────────
    print("\n6️⃣  Génération des graphiques...")
    scatter_path  = str(project_root / "scatter_turbidity.png")
    range_path    = str(project_root / "mae_by_range.png")
    plot_scatter(
        preds_np["turbidity_NTU"],
        targets_np["turbidity_NTU"],
        all_metrics["turbidity_NTU"],
        scatter_path,
    )
    plot_error_by_range(
        preds_np["turbidity_NTU"],
        targets_np["turbidity_NTU"],
        range_path,
    )

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    print("\n7️⃣  Sauvegarde des résultats...")
    results = {
        "checkpoint"  : str(checkpoint_path),
        "device"      : str(device),
        "n_test_images": len(test_loader.dataset),
        "metrics"     : all_metrics,
        "interpretation": {
            "turbidity_mae" : interpret_turbidity(turb_mae),
            "r2_note"       : (
                "R² peut être NaN/négatif si le TEST set contient un seul clip "
                "vidéo avec faible variabilité (SS_tot ≈ 0). "
                "MAE et RMSE restent des métriques fiables dans ce cas."
            ),
        },
    }

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, indent=2, ensure_ascii=False)
    print("   ✅ Résultats sauvegardés → test_results.json")

    # ── Résumé final ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("✅ ÉVALUATION TERMINÉE")
    print(f"{'='*70}")
    print(f"\n  Turbidité MAE  : {turb_mae:.4f} NTU")
    print(f"  Turbidité RMSE : {all_metrics['turbidity_NTU']['rmse']:.4f} NTU")
    print(f"  {interpret_turbidity(turb_mae)}")
    print(f"\n  Fichiers générés :")
    print(f"    test_results.json       ← métriques complètes")
    print(f"    scatter_turbidity.png   ← scatter plot")
    print(f"    mae_by_range.png        ← MAE par plage")
    print(f"\n  💡 Prochaine étape : python predict.py")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Chemin par défaut résolu par rapport à evaluate.py lui-même
    # Évite le FileNotFoundError quand on lance depuis un autre répertoire
    _default_ckpt = str(Path(__file__).resolve().parent / "checkpoints" / "best_model.pth")

    parser = argparse.ArgumentParser(description="Évaluation finale sur le TEST set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=_default_ckpt,
        help="Chemin vers le checkpoint (défaut: <dossier_script>/checkpoints/best_model.pth)",
    )
    args = parser.parse_args()
    evaluate(checkpoint_path=args.checkpoint)