#!/usr/bin/env python3
"""
MONTE CARLO DROPOUT — Évaluation de l'incertitude épistémique
Projet : Estimation de qualité d'eau — Tilapia RAS

Référence : Gal & Ghahramani, ICML 2016
            "Dropout as a Bayesian Approximation"

Version optimale : fusion des deux versions (A + B)
  ✅ enable_mcd() propre        (version B)
  ✅ Sanity check fail-fast     (version B)
  ✅ Inférence image par image  (version B)
  ✅ JSON-safe serialisation    (version B)
  ✅ Argparse CLI               (version B)
  ✅ 4 graphiques               (version A)
  ✅ RMSE + R² globaux          (version A)
  ✅ Bug indentation corrigé    (version B avait un crash)
  ✅ Scatter STD vs MAE         (version A)
  ✅ Prédictions ± barres       (version A)

Usage :
    python mcd_evaluate.py
    python mcd_evaluate.py --T 100
    python mcd_evaluate.py --T 50 --mu 0.3
    python mcd_evaluate.py --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Imports projet ───────────────────────────────────────────────────────────
from model import create_model
from preprocess import create_preprocessor
from dataset import create_dataloaders

# ── Configuration globale ────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
OUTPUT_DIR      = Path("mcd_results")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tâches actives (λ > 0 dans train.py)
ACTIVE_TASKS = ["turbidity_NTU", "DO_mgL"]
TASK_UNITS   = {"turbidity_NTU": "NTU", "DO_mgL": "mg/L"}

# Tolérance pour définir "correct" : |pred - label| < tolérance
TOLERANCES = {
    "turbidity_NTU": 0.1,    # NTU
    "DO_mgL":        0.05,   # mg/L
}

# Couleurs graphiques
COLORS = {
    "title":   "#1F5C99",
    "correct": "#1D9E75",
    "wrong":   "#E24B4A",
    "do":      "#4CAF50",
    "grid":    "#E8E8E8",
}


# ════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════

def _is_valid(value: float | None) -> bool:
    """True si la valeur est un float fini."""
    if value is None:
        return False
    return not (math.isnan(value) or math.isinf(value))


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Moyenne et écart-type d'une liste."""
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt(mean: float, std: float) -> str:
    """Format affichage mean +/- std."""
    if not _is_valid(mean) or not _is_valid(std):
        return "nan"
    return f"{mean:.4f} +/- {std:.4f}"


def json_safe(obj: object) -> object:
    """Rend un objet sérialisable JSON (gère NaN, Inf, tensors, Path)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):
        v = obj.item()
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
    return obj


# ════════════════════════════════════════════════════════════════════════════
# 1. ACTIVATION DU DROPOUT À L'INFÉRENCE  (meilleure approche — version B)
# ════════════════════════════════════════════════════════════════════════════

def enable_mcd(model: nn.Module) -> nn.Module:
    """
    Active le dropout uniquement, en laissant le reste en mode eval().

    Stratégie correcte :
      - model.eval()  → fige BatchNorm (utilise stats globales, pas du mini-batch)
      - m.train()     → réactive UNIQUEMENT les nn.Dropout

    Pourquoi NE PAS faire model.train() global :
      - model.train() réactive aussi les BatchNorm → stats instables à l'inférence
      - Seul le Dropout doit être stochastique

    Raises:
        RuntimeError si aucun nn.Dropout n'est trouvé (MCD impossible)
    """
    model.eval()

    dropout_layers = []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            dropout_layers.append(m)

    if not dropout_layers:
        raise RuntimeError(
            "❌ Aucun nn.Dropout trouvé dans le modèle.\n"
            "   MCD nécessite nn.Dropout dans _make_regression_head().\n"
            "   Attendu : nn.Dropout(p=0.3) entre les deux Linear."
        )

    print(f"✅ MCD activé — {len(dropout_layers)} couche(s) Dropout "
          f"(p={dropout_layers[0].p:.1f}) | BatchNorm figé")
    return model


# ════════════════════════════════════════════════════════════════════════════
# 2. SANITY CHECK — FAIL FAST  (version B, absent dans version A)
# ════════════════════════════════════════════════════════════════════════════

def sanity_check_mcd(model: nn.Module, test_loader, T: int = 20) -> None:
    """
    Vérifie que MCD fonctionne avant la boucle complète.
    Prend 1 image, fait T passes, vérifie que STD > 0.

    Raises:
        ValueError si STD == 0 (dropout non actif → toutes passes identiques)
    """
    print("\n🔍 SANITY CHECK MCD...")

    images, _ = next(iter(test_loader))
    image = images[0].unsqueeze(0).to(DEVICE)   # [1, 3, H, W]

    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = model(image)
            preds.append(out["turbidity_NTU"].item())

    mean_v = np.mean(preds)
    std_v  = np.std(preds)

    print(f"   T={T} passes turbidité → Mean={mean_v:.4f} NTU | STD={std_v:.5f}")

    if std_v == 0.0:
        raise ValueError(
            "❌ MCD NON FONCTIONNEL : STD=0, toutes les passes sont identiques.\n"
            "   enable_mcd() n'a pas activé le dropout. Vérifier model.py."
        )
    if std_v < 1e-4:
        print(f"   ⚠️  STD très faible ({std_v:.6f}) — vérifier dropout_rate=0.3")
    else:
        print("   ✅ SANITY CHECK PASSED — stochasticité confirmée")


# ════════════════════════════════════════════════════════════════════════════
# 3. INFÉRENCE MCD SUR UNE IMAGE  (version B — image par image)
# ════════════════════════════════════════════════════════════════════════════

def mcd_predict(model: nn.Module, image: torch.Tensor, T: int) -> Dict[str, float]:
    """
    T passes stochastiques sur une seule image.

    Pourquoi image par image (et non par batch) :
      - Chaque appel model(image) génère un masque dropout indépendant
      - En batch, toutes les images d'un batch partagent le même masque
        → sous-estimation de la variance inter-images

    Returns:
        Dict avec {task}_mean, {task}_std, {task}_var pour chaque tâche active
    """
    preds: Dict[str, List[float]] = {task: [] for task in ACTIVE_TASKS}

    with torch.no_grad():
        for _ in range(T):
            out = model(image)
            for task in ACTIVE_TASKS:
                preds[task].append(out[task].item())

    result = {}
    for task in ACTIVE_TASKS:
        arr = np.array(preds[task])
        result[f"{task}_mean"] = float(arr.mean())
        result[f"{task}_std"]  = float(arr.std())
        result[f"{task}_var"]  = float(arr.var())

    return result


# ════════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION MCD SUR TOUT LE TEST SET
# ════════════════════════════════════════════════════════════════════════════

def evaluate_mcd(model: nn.Module, test_loader, T: int) -> List[Dict]:
    """
    Applique MCD sur les K images du TEST set.

    Pour chaque image k :
      - T passes → mean_pred, std_pred, var_pred par tâche
      - MAE vs vérité terrain
      - correct : |mean - label| < TOLERANCES[task]

    Returns:
        Liste de K dicts, un par image
    """
    results = []
    k = 0

    for images, targets in tqdm(test_loader, desc=f"MCD T={T}"):
        images = images.to(DEVICE)

        for i in range(images.shape[0]):
            img  = images[i].unsqueeze(0)
            pred = mcd_predict(model, img, T=T)

            record = {"k": k}

            for task in ACTIVE_TASKS:
                label = float(targets[task][i].item())
                mean  = pred[f"{task}_mean"]
                std   = pred[f"{task}_std"]
                mae   = abs(mean - label) if _is_valid(label) else float("nan")
                correct = (mae < TOLERANCES[task]) if _is_valid(mae) else None

                record[f"mean_{task}"]    = mean
                record[f"std_{task}"]     = std
                record[f"var_{task}"]     = pred[f"{task}_var"]
                record[f"label_{task}"]   = label
                record[f"mae_{task}"]     = mae
                record[f"correct_{task}"] = correct

            results.append(record)
            k += 1

    return results


# ════════════════════════════════════════════════════════════════════════════
# 5. MÉTRIQUES GLOBALES : MAE, RMSE, R²  (version A — absent dans B)
# ════════════════════════════════════════════════════════════════════════════

def compute_global_metrics(results: List[Dict]) -> Dict[str, Dict]:
    """
    Calcule MAE, RMSE, R² sur l'ensemble du TEST set (prédiction = mean MCD).
    Ces métriques sont comparables directement avec les résultats Phase 3.
    """
    metrics = {}
    for task in ACTIVE_TASKS:
        unit = TASK_UNITS[task]
        valid = [
            r for r in results
            if _is_valid(r.get(f"mae_{task}")) and _is_valid(r.get(f"label_{task}"))
        ]
        if not valid:
            metrics[task] = {}
            continue

        y_true = np.array([r[f"label_{task}"] for r in valid])
        y_pred = np.array([r[f"mean_{task}"]  for r in valid])
        stds   = np.array([r[f"std_{task}"]   for r in valid])

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics[task] = {
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "mean_STD": float(stds.mean()),
            "unit": unit,
        }
        print(f"   {task:20s} : MAE={mae:.4f} {unit} | "
              f"RMSE={rmse:.4f} | R²={r2:.4f} | STD_moy={stds.mean():.5f}")

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# 6. TABLEAU DE PERFORMANCES AVEC ÉCARTS-TYPES  (version B, enrichi)
# ════════════════════════════════════════════════════════════════════════════

def print_performance_table(results: List[Dict], T: int) -> Dict:
    """
    Tableau MAE +/- std et incertitude MCD pour corrects vs incorrects.
    Hypothèse attendue : incertitude(Corrects) < incertitude(Incorrects).
    """
    stats_all = {}

    for task in ACTIVE_TASKS:
        unit = TASK_UNITS[task]
        all_mae, all_std = [], []
        cor_mae, cor_std = [], []
        inc_mae, inc_std = [], []
        n_cor = n_inc = 0

        for r in results:
            mae_v   = r.get(f"mae_{task}")
            std_v   = r.get(f"std_{task}")
            correct = r.get(f"correct_{task}")
            if not _is_valid(mae_v) or not _is_valid(std_v) or correct is None:
                continue
            all_mae.append(float(mae_v))
            all_std.append(float(std_v))
            if correct:
                n_cor += 1
                cor_mae.append(float(mae_v))
                cor_std.append(float(std_v))
            else:
                n_inc += 1
                inc_mae.append(float(mae_v))
                inc_std.append(float(std_v))

        N = len(all_mae)
        pct = 100.0 * n_cor / N if N else float("nan")

        header = (
            f"{'Groupe':<22} | {'MAE mean+/-std':<22} | "
            f"{'Incert. mean+/-std':<22} | {'% correct':>9}"
        )
        print(f"\n{'─'*len(header)}")
        print(f" {task} ({unit})  |  T={T}  |  n={N}")
        print(f"{'─'*len(header)}")
        print(header)
        print(f"{'─'*len(header)}")

        for label, maes, stds in [
            (f"Tous      (n={N})",   all_mae, all_std),
            (f"Corrects  (n={n_cor})", cor_mae, cor_std),
            (f"Incorrects(n={n_inc})", inc_mae, inc_std),
        ]:
            m_mae, s_mae = _mean_std(maes)
            m_std, s_std = _mean_std(stds)
            pct_row = 100.0 * len(maes) / N if N else float("nan")
            if maes == inc_mae:
                pct_row = 0.0
            elif maes == cor_mae:
                pct_row = 100.0
            print(
                f" {label:<22} | {_fmt(m_mae, s_mae):<22} | "
                f"{_fmt(m_std, s_std):<22} | {pct_row:9.1f}"
            )

        print(f"{'─'*len(header)}")

        # Vérification hypothèse MCD
        m_cor_std = np.mean(cor_std) if cor_std else float("nan")
        m_inc_std = np.mean(inc_std) if inc_std else float("nan")
        if _is_valid(m_cor_std) and _is_valid(m_inc_std):
            detected = m_inc_std > m_cor_std
            print(f" MCD détecte ses erreurs : {'✅ OUI' if detected else '❌ NON'} "
                  f"(STD_cor={m_cor_std:.5f}  STD_inc={m_inc_std:.5f})")

        stats_all[task] = {
            "n": N, "n_cor": n_cor, "n_inc": n_inc, "pct_correct": pct,
            "all_mae": _mean_std(all_mae), "all_std": _mean_std(all_std),
            "cor_std": m_cor_std, "inc_std": m_inc_std,
        }

    return stats_all


# ════════════════════════════════════════════════════════════════════════════
# 7. GRAPHIQUES  (4 figures — version A + figures clés de B)
# ════════════════════════════════════════════════════════════════════════════

def plot_all(results: List[Dict], T: int, output_dir: Path) -> None:
    """Génère les 4 figures MCD."""

    output_dir.mkdir(exist_ok=True)

    # ── Figure 1 : Histogrammes STD corrects vs incorrects ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        f"Monte Carlo Dropout — Incertitude épistémique (T={T})\n"
        "Tilapia RAS — TEST set",
        fontsize=12, fontweight="bold", color=COLORS["title"]
    )

    for col, task in enumerate(ACTIVE_TASKS):
        ax = axes[col]
        unit = TASK_UNITS[task]

        cor = [r[f"std_{task}"] for r in results
               if r.get(f"correct_{task}") is True and _is_valid(r.get(f"std_{task}"))]
        inc = [r[f"std_{task}"] for r in results
               if r.get(f"correct_{task}") is False and _is_valid(r.get(f"std_{task}"))]

        all_vals = cor + inc
        if all_vals:
            bins = np.linspace(0, max(all_vals) + 1e-6, 35)
            ax.hist(cor, bins=bins, alpha=0.72,
                    color=COLORS["correct"], label=f"Corrects (n={len(cor)})")
            ax.hist(inc, bins=bins, alpha=0.72,
                    color=COLORS["wrong"],   label=f"Incorrects (n={len(inc)})")
            if cor:
                ax.axvline(np.mean(cor), color="#0F6E56", ls="--", lw=1.5,
                           label=f"Moy. cor = {np.mean(cor):.5f}")
            if inc:
                ax.axvline(np.mean(inc), color="#A32D2D", ls="--", lw=1.5,
                           label=f"Moy. inc = {np.mean(inc):.5f}")

        ax.set_xlabel(f"STD des T prédictions ({unit})")
        ax.set_ylabel("Nombre d'images")
        ax.set_title(f"{task}")
        ax.legend(fontsize=8)
        ax.grid(True, color=COLORS["grid"], lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = output_dir / "fig1_std_histograms.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {p.name}")

    # ── Figure 2 : Scatter STD vs MAE  (version A) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        f"MCD — Corrélation incertitude / erreur (T={T})",
        fontsize=12, fontweight="bold", color=COLORS["title"]
    )

    for col, task in enumerate(ACTIVE_TASKS):
        ax = axes[col]
        unit = TASK_UNITS[task]

        valid = [r for r in results
                 if _is_valid(r.get(f"std_{task}")) and _is_valid(r.get(f"mae_{task}"))]
        if not valid:
            continue

        x = np.array([r[f"mae_{task}"] for r in valid])
        y = np.array([r[f"std_{task}"] for r in valid])

        sc = ax.scatter(x, y, c=y, cmap="RdYlGn_r",
                        alpha=0.45, s=12, linewidths=0)
        plt.colorbar(sc, ax=ax, label=f"STD ({unit})")

        # Ligne de tendance
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, np.poly1d(z)(xline), "r--", lw=1.8, label="Tendance")

        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
        ax.set_xlabel(f"|prédiction − vérité| ({unit})")
        ax.set_ylabel(f"STD MCD ({unit})")
        ax.set_title(f"{task}\nr = {corr:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, color=COLORS["grid"], lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = output_dir / "fig2_std_vs_mae.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {p.name}")

    # ── Figure 3 : Prédictions ± barres d'incertitude  (version A) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        f"MCD — Prédictions avec intervalles de confiance (T={T})",
        fontsize=12, fontweight="bold", color=COLORS["title"]
    )

    for col, task in enumerate(ACTIVE_TASKS):
        ax = axes[col]
        unit = TASK_UNITS[task]
        color = COLORS["correct"] if task == "turbidity_NTU" else COLORS["do"]

        sample = [r for r in results if _is_valid(r.get(f"label_{task}"))]
        sample = sorted(sample, key=lambda r: r[f"label_{task}"])
        if len(sample) > 250:
            idx = np.linspace(0, len(sample) - 1, 250, dtype=int)
            sample = [sample[i] for i in idx]

        x = [r[f"label_{task}"] for r in sample]
        y = [r[f"mean_{task}"]  for r in sample]
        e = [r[f"std_{task}"]   for r in sample]

        ax.errorbar(x, y, yerr=e, fmt="o", alpha=0.4, markersize=3,
                    color=color, ecolor="lightsteelblue", elinewidth=0.7,
                    label="Prédiction ± STD")
        lim = [min(min(x), min(y)), max(max(x), max(y))]
        ax.plot(lim, lim, "r--", lw=1.8, label="Prédiction parfaite")
        ax.set_xlabel(f"Valeur réelle ({unit})")
        ax.set_ylabel(f"Prédiction MCD ({unit})")
        ax.set_title(f"{task}")
        ax.legend(fontsize=8)
        ax.grid(True, color=COLORS["grid"], lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = output_dir / "fig3_predictions.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {p.name}")


# ════════════════════════════════════════════════════════════════════════════
# 8. OPTIMISATION DU SEUIL µ  (version B corrigée + courbe version A)
# ════════════════════════════════════════════════════════════════════════════

def optimize_threshold(
    results: List[Dict],
    mu_values: List[float],
    output_dir: Path,
    global_mae: float = 0.041,
    T: int | None = None,
) -> List[Dict]:
    """
    Pour chaque µ : partitionne en Retained (STD ≤ µ) et Referred (STD > µ).

    Génère :
      - Tableau terminal
      - fig4_threshold.png (courbe couverture vs MAE)

    NOTE : µ opère sur la STD brute de turbidité (NTU), ce qui est
    directement interprétable (pas de normalisation arbitraire).

    Bug corrigé version B :
      - Indentation incorrecte sur `cor_ref` / `inc_ref` causait un crash.
    """
    valid = [
        r for r in results
        if _is_valid(r.get("std_turbidity_NTU"))
        and _is_valid(r.get("mae_turbidity_NTU"))
        and r.get("correct_turbidity_NTU") is not None
    ]
    # Compatibilité clé (selon la task naming dans evaluate_mcd)
    if not valid:
        valid = [
            r for r in results
            if _is_valid(r.get("std_turbidity_NTU", r.get("std_turb")))
            and _is_valid(r.get("mae_turbidity_NTU", r.get("mae_turb")))
            and r.get("correct_turbidity_NTU", r.get("correct_turb")) is not None
        ]

    def _get(r, key, fallback_key):
        return r.get(key, r.get(fallback_key))

    N = len(valid)
    table = []

    print(f"\n{'='*70}")
    print(f"  OPTIMISATION DU SEUIL µ  |  turbidité  |  n={N} images")
    print(f"{'='*70}")
    print(f"  {'µ':>5} | {'Ret':>5} | {'%Ret':>6} | {'Réf':>5} | "
          f"{'MAE_ret':>8} | {'Cor_ret':>7} | {'Inc_ret':>7}")
    print(f"  {'─'*62}")

    for mu in mu_values:
        retained = [r for r in valid
                    if _get(r, "std_turbidity_NTU", "std_turb") <= mu]
        referred = [r for r in valid
                    if _get(r, "std_turbidity_NTU", "std_turb") >  mu]

        n_ret   = len(retained)
        n_ref   = len(referred)
        pct_ret = 100.0 * n_ret / N if N else float("nan")
        mae_ret = float(np.mean([_get(r, "mae_turbidity_NTU", "mae_turb")
                                 for r in retained])) if retained else float("nan")

        # ✅ CORRECTION BUG version B : toutes les variables dans le même bloc
        cor_ret = sum(1 for r in retained
                      if _get(r, "correct_turbidity_NTU", "correct_turb"))
        inc_ret = sum(1 for r in retained
                      if not _get(r, "correct_turbidity_NTU", "correct_turb"))
        cor_ref = sum(1 for r in referred
                      if _get(r, "correct_turbidity_NTU", "correct_turb"))
        inc_ref = sum(1 for r in referred
                      if not _get(r, "correct_turbidity_NTU", "correct_turb"))

        mae_str = f"{mae_ret:>8.4f}" if _is_valid(mae_ret) else "     nan"
        opt_flag = " ← optimal?" if pct_ret >= 70 else ""
        print(f"  {mu:>5.2f} | {n_ret:>5} | {pct_ret:>5.1f}% | {n_ref:>5} | "
              f"{mae_str} | {cor_ret:>7} | {inc_ret:>7}{opt_flag}")

        table.append({
            "mu": mu, "n_ret": n_ret, "n_ref": n_ref,
            "pct_ret": pct_ret, "mae_ret": mae_ret,
            "cor_ret": cor_ret, "inc_ret": inc_ret,
            "cor_ref": cor_ref, "inc_ref": inc_ref,
        })

    print(f"{'='*70}")

    # ── Figure 4 : courbe couverture vs MAE ─────────────────────────────────
    pct_list  = [r["pct_ret"] for r in table]
    mae_list  = [r["mae_ret"] for r in table]
    mu_labels = [f"µ={r['mu']:.2f}" for r in table]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pct_list, mae_list, "o-",
            color=COLORS["title"], lw=2, markersize=7, zorder=3)

    for x, y, lbl in zip(pct_list, mae_list, mu_labels):
        ax.annotate(lbl, (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color="#444")

    ax.axhline(global_mae, color=COLORS["correct"], ls="--", lw=1.3,
               label=f"MAE global (sans seuil) = {global_mae:.3f} NTU")
    ax.axvspan(70, 100, alpha=0.06, color="green", label="Zone ≥ 70% retained")

    ax.set_xlabel("% images retenues (décision automatique)")
    ax.set_ylabel("MAE turbidité sur images retenues (NTU)")
    if T is None:
        title_suffix = "Monte Carlo Dropout — Tilapia RAS"
    else:
        title_suffix = f"Monte Carlo Dropout (T={T}) — Tilapia RAS"
    ax.set_title(
        f"Seuil µ — Trade-off couverture / précision\n"
        f"{title_suffix}",
        fontsize=11, fontweight="bold", color=COLORS["title"]
    )
    ax.legend(fontsize=9)
    ax.grid(True, color=COLORS["grid"], lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = output_dir / "fig4_threshold.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   ✅ {p.name}")

    return table


# ════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Dropout — Water Quality — Tilapia RAS"
    )
    parser.add_argument("--T",   type=int,   default=50,
                        help="Nombre de passes stochastiques (défaut: 50)")
    parser.add_argument("--mu",  type=float, default=None,
                        help="Seuil µ unique (défaut: sweep 0.1→0.9)")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH),
                        help="Chemin vers best_model.pth")
    args = parser.parse_args()

    T          = args.T
    checkpoint = Path(args.checkpoint)
    mu_values  = ([round(args.mu, 2)] if args.mu is not None
                  else [round(x * 0.05, 2) for x in range(1, 20)])  # 0.05→0.95

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  MONTE CARLO DROPOUT — Tilapia RAS Water Quality")
    print(f"  T={T} passes  |  device={DEVICE}  |  tasks={ACTIVE_TASKS}")
    print(f"  Checkpoint : {checkpoint}")
    print("=" * 65)

    # ── 1. Modèle ────────────────────────────────────────────────────────────
    print("\n1️⃣  Chargement du modèle...")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"❌ Checkpoint introuvable : {checkpoint}\n"
            "   Lancer train.py d'abord."
        )
    model = create_model({
        "pretrained": False, "dropout_rate": 0.3, "freeze_backbone": False,
    })
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    print(f"   Epoch={ckpt.get('epoch','?')+1}  |  "
          f"val_loss={ckpt.get('val_loss','?'):.4f}")

    # ── 2. Activer MCD ───────────────────────────────────────────────────────
    print("\n2️⃣  Activation MCD...")
    model = enable_mcd(model)

    # ── 3. TEST set ──────────────────────────────────────────────────────────
    print("\n3️⃣  Chargement TEST set...")
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

    preprocessor = create_preprocessor()
    dataloaders  = create_dataloaders(
        csv_path=csv_path, dataset_root=dataset_root,
        preprocessor=preprocessor,
        batch_size=8,    # petit batch recommandé (T passes par image)
        num_workers=0,   # Windows-safe
    )
    test_loader = dataloaders["TEST"]
    print(f"   TEST : {len(test_loader.dataset)} images  |  "
          f"~{len(test_loader.dataset) * T // 1000}k forward passes total")

    # ── 4. Sanity check ──────────────────────────────────────────────────────
    sanity_check_mcd(model, test_loader, T=min(20, T))

    # ── 5. Inférence MCD ─────────────────────────────────────────────────────
    print(f"\n4️⃣  Inférence MCD sur {len(test_loader.dataset)} images (T={T})...")
    results = evaluate_mcd(model, test_loader, T=T)

    # ── 6. Sauvegarde JSON ───────────────────────────────────────────────────
    json_path = OUTPUT_DIR / "mcd_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, indent=2)
    print(f"\n   💾 {json_path}")

    # ── 7. Métriques globales ─────────────────────────────────────────────────
    print("\n5️⃣  Métriques globales (MAE / RMSE / R²)...")
    global_metrics = compute_global_metrics(results)
    global_mae_turb = global_metrics.get("turbidity_NTU", {}).get("MAE", 0.041)

    # ── 8. Tableau correct/incorrect ─────────────────────────────────────────
    print("\n6️⃣  Tableau de performances...")
    perf_stats = print_performance_table(results, T)

    # ── 9. Graphiques ─────────────────────────────────────────────────────────
    print("\n7️⃣  Génération des figures...")
    plot_all(results, T, OUTPUT_DIR)

    # ── 10. Optimisation µ ────────────────────────────────────────────────────
    print("\n8️⃣  Optimisation du seuil µ...")
    threshold_table = optimize_threshold(
        results, mu_values, OUTPUT_DIR, global_mae=global_mae_turb, T=T
    )

    # ── Résumé ────────────────────────────────────────────────────────────────
    turb = perf_stats.get("turbidity_NTU", {})
    print(f"\n{'='*65}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*65}")
    print(f"  MAE turbidité (MCD)     : {global_metrics.get('turbidity_NTU',{}).get('MAE', '?'):.4f} NTU")
    print(f"  MAE DO        (MCD)     : {global_metrics.get('DO_mgL',{}).get('MAE', '?'):.4f} mg/L")
    print(f"  % correct turb.         : {turb.get('pct_correct','?'):.1f}%  (seuil={TOLERANCES['turbidity_NTU']} NTU)")
    print(f"  STD moyen corrects      : {turb.get('cor_std','?'):.5f} NTU")
    print(f"  STD moyen incorrects    : {turb.get('inc_std','?'):.5f} NTU")
    print(f"\n  Fichiers → {OUTPUT_DIR}/")
    for name in ["mcd_results.json", "fig1_std_histograms.png",
                 "fig2_std_vs_mae.png", "fig3_predictions.png",
                 "fig4_threshold.png"]:
        print(f"    {name}")
    print(f"{'='*65}")
    print("\n💡 Prochaine étape : analyser fig4_threshold.png → choisir µ optimal")


if __name__ == "__main__":
    main()