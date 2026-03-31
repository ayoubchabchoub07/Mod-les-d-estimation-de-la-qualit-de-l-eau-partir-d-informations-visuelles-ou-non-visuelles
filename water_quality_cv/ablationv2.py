#!/usr/bin/env python3
"""
ÉTUDE V2 : ABLATION TURBIDITÉ / pH - PHASE UNIQUE
DO et température FIXES à 0.0 dans toutes les configs.

Stratégie : grille complète sur lambda_pH (0.0 -> 1.0) avec lambda_turb=1.0 fixe.
  - 8 configs x 10 epochs max x patience=4 (early stopping)
  - Temps estimé : ~15h GPU
    logger.info("# Grille : DO in {0.0, 0.2, 0.4, 0.5, 0.7, 0.9}")
✅ Version sans surveillance : pas de input(), logs dans fichier
"""

from pathlib import Path
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import logging

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from preprocess import create_preprocessor
from dataset import create_dataloaders
from model import create_model
from loss import create_loss_function


def resolve_dataset_root(base_dir: Path) -> Path:
    for parent in [base_dir, *base_dir.parents]:
        candidate = parent / "Tilapia RAS Dataset"
        if candidate.exists():
            return candidate
    return base_dir.parent / "Tilapia RAS Dataset"


# ==============================================================================
# LOGGING : Console + Fichier simultanément
# ==============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # Fichier
    fh = logging.FileHandler(output_dir / "ablation_log.txt", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ==============================================================================
# MÉTRIQUES
# ==============================================================================

def calculate_metrics(predictions, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return None, None, None, None

    pred = predictions[mask]
    targ = targets[mask]

    mae  = torch.abs(pred - targ).mean().item()
    rmse = torch.sqrt(torch.mean((pred - targ) ** 2)).item()

    ss_res = torch.sum((targ - pred) ** 2).item()
    ss_tot = torch.sum((targ - targ.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    mape = torch.mean(torch.abs((targ - pred) / (targ + 1e-8))) * 100
    mape = mape.item() if not torch.isnan(mape) else None

    return mae, rmse, r2, mape


def evaluate_all_tasks(model, loader, device):
    model.eval()
    TASKS = ['turbidity_NTU', 'pH', 'temperature_C', 'DO_mgL']
    all_preds   = {t: [] for t in TASKS}
    all_targets = {t: [] for t in TASKS}

    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            preds   = model(images)
            for t in TASKS:
                all_preds[t].append(preds[t].cpu())
                all_targets[t].append(targets[t].cpu())

    metrics = {}
    for t in TASKS:
        p = torch.cat(all_preds[t])
        g = torch.cat(all_targets[t])
        mae, rmse, r2, mape = calculate_metrics(p, g)
        metrics[t] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

    return metrics


# ==============================================================================
# ENTRAÎNEMENT D'UN MODÈLE
# ==============================================================================

def train_model(config_name, lambdas, num_epochs, batch_size=32, logger=None):

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log(f"\n{'='*80}")
    log(f"📊 MODÈLE : {config_name}")
    log(f"{'='*80}")
    log("🎯 Poids (lambda) :")
    for param, weight in lambdas.items():
        status = "✅ ACTIF" if weight > 0 else "❌ INACTIF"
        log(f"   {param:20s} : {weight:.2f}  {status}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"💻 Device : {device}")

    csv_path     = project_root / "processed" / "images_labels.csv"
    dataset_root = resolve_dataset_root(project_root)

    preprocessor = create_preprocessor()
    dataloaders  = create_dataloaders(
        csv_path=csv_path,
        dataset_root=dataset_root,
        preprocessor=preprocessor,
        batch_size=batch_size,
        num_workers=0
    )

    train_loader = dataloaders['TRAIN']
    val_loader   = dataloaders['VAL']
    test_loader  = dataloaders.get('TEST', val_loader)

    model     = create_model({'pretrained': True, 'dropout_rate': 0.3})
    model     = model.to(device)
    criterion = create_loss_function({'lambdas': lambdas, 'delta': 1.0})
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss    = float('inf')
    best_epoch       = 0
    best_model_state = None
    patience_counter = 0
    PATIENCE         = 4   # early stopping : arrêt si pas d'amélioration pendant 4 epochs
    history          = {'train_losses': [], 'val_losses': [], 'test_losses': []}

    log(f"\n🚀 Entraînement ({num_epochs} epochs)...\n")

    epoch_start = datetime.now()

    for epoch in range(num_epochs):

        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:02d}/{num_epochs} [TRAIN]",
            leave=False
        ):
            images  = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            optimizer.zero_grad()
            predictions = model(images)
            loss, _     = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── VALIDATION ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1:02d}/{num_epochs} [VAL]  ",
                leave=False
            ):
                images  = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                predictions = model(images)
                loss, _     = criterion(predictions, targets)
                val_loss   += loss.item()
        val_loss /= len(val_loader)

        # ── TEST LOSS (suivi epoch par epoch, sans maj des poids) ──
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images  = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                predictions = model(images)
                loss, _     = criterion(predictions, targets)
                test_loss  += loss.item()
        test_loss /= len(test_loader)

        scheduler.step()
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['test_losses'].append(test_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            saved_marker     = " ✅ BEST"
            patience_counter = 0
        else:
            patience_counter += 1
            saved_marker = f" (patience {patience_counter}/{PATIENCE})"

        # Early stopping intra-ablation
        if patience_counter >= PATIENCE:
            log(f"  ⏹  Early stopping epoch {epoch+1} - pas d'amélioration depuis {PATIENCE} epochs")
            break

        # ETA
        elapsed_s  = (datetime.now() - epoch_start).total_seconds()
        per_epoch  = elapsed_s / (epoch + 1)
        remaining  = per_epoch * (num_epochs - epoch - 1)
        eta_str    = f"ETA ~{remaining/60:.0f} min"

        log(f"  Epoch {epoch+1:02d}/{num_epochs}: Train={train_loss:.4f}  Val={val_loss:.4f}  Test={test_loss:.4f}{saved_marker}  [{eta_str}]")

    # ── ÉVALUATION FINALE ──
    model.load_state_dict(best_model_state)
    log(f"\n📊 Évaluation finale (meilleur epoch : {best_epoch})...")

    val_metrics  = evaluate_all_tasks(model, val_loader,  device)
    test_metrics = evaluate_all_tasks(model, test_loader, device)

    log(f"\n{'='*60}")
    log("RÉSULTATS PAR TÂCHE (TEST)")
    log(f"{'='*60}")
    for task in ['turbidity_NTU', 'pH', 'temperature_C', 'DO_mgL']:
        if lambdas[task] > 0:
            m = test_metrics[task]
            log(f"  🎯 {task}:")
            log(f"     MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    results = {
        'config_name': config_name,
        'lambdas':     lambdas,
        'best_epoch':  best_epoch,
        'best_val_loss': best_val_loss,
        'val_metrics':  val_metrics,
        'test_metrics': test_metrics,
        'history':      history
    }

    # Cleanup mémoire
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ==============================================================================
# TABLEAUX ET GRAPHIQUES
# ==============================================================================

def create_comparison_table(results, phase_name, logger=None):
    def log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    log(f"\n{'='*80}")
    log(f"📊 TABLEAU COMPARATIF - {phase_name}")
    log(f"{'='*80}\n")
    log(f"{'Modèle':<30} | {'Val Loss':>9} | {'Turb MAE':>9} {'Turb R²':>9} | {'DO MAE':>9} {'DO R²':>9}")
    log("-" * 95)

    for r in results:
        turb = r['test_metrics']['turbidity_NTU']
        do   = r['test_metrics']['DO_mgL']
        log(
            f"{r['config_name']:<30} | {r['best_val_loss']:>9.4f} | "
            f"{(turb['mae'] or 0):>9.4f} {(turb['r2'] or 0):>9.4f} | "
            f"{(do['mae'] or 0):>9.4f} {(do['r2'] or 0):>9.4f}"
        )
    log("-" * 95)


def plot_individual_curves(all_results, output_dir, phase_name):
    """
    Une sous-figure par configuration : Train (bleu) / Val (orange) / Test (vert).
    Ligne rouge verticale = best epoch (critère val_loss).
    Echelle Y indépendante par sous-figure (chaque config a sa propre plage de loss).
    Layout : 2 colonnes, N/2 lignes - lisible jusqu'à 10 configs.
    """
    n     = len(all_results)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    slug  = phase_name.lower().replace(" ", "_").replace("/", "_")

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7, nrows * 4.5),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, r in enumerate(all_results):
        ax     = axes_flat[idx]
        n_ep   = len(r['history']['train_losses'])
        epochs = range(1, n_ep + 1)

        train_l = r['history']['train_losses']
        val_l   = r['history']['val_losses']
        test_l  = r['history'].get('test_losses', [])

        ax.plot(epochs, train_l, label='Train',
                color='#1565C0', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, val_l,   label='Val',
                color='#E65100', linewidth=2, marker='s', markersize=4)
        if test_l:
            ax.plot(epochs, test_l, label='Test',
                    color='#2E7D32', linewidth=2,
                    marker='^', markersize=4, linestyle='--')

        # Ligne verticale best epoch
        best_ep = r['best_epoch']
        ax.axvline(x=best_ep, color='#C62828', linestyle='--',
                   linewidth=1.5, alpha=0.8, label=f'Best val = epoch {best_ep}')

        # Annotation valeurs au best epoch
        bv = val_l[best_ep - 1]
        bt = train_l[best_ep - 1]
        ratio = bv / bt if bt > 0 else 0
        overfit_color = '#C62828' if ratio > 3 else '#2E7D32'
        ax.annotate(
            f'val={bv:.4f}  train={bt:.4f}  ratio={ratio:.1f}x',
            xy=(best_ep, bv),
            xytext=(best_ep + max(1, n_ep*0.08), bv),
            fontsize=7, color=overfit_color,
            arrowprops=dict(arrowstyle='->', color=overfit_color, lw=1))

        # MAE turbidité TEST dans le titre
        mae = r['test_metrics']['turbidity_NTU']['mae']
        r2  = r['test_metrics']['turbidity_NTU']['r2']
        mae_str = f"MAE={mae:.4f} NTU" if mae else "MAE=N/A"
        r2_str  = f"  R²={r2:.3f}"     if r2  else "  R²=NaN"

        # Lambda actifs dans le titre
        lph  = r['lambdas']['pH']
        lturb = r['lambdas']['turbidity_NTU']
        lam_str = f'lturb={lturb}  lpH={lph}'

        ax.set_title(f"{r['config_name']}  [{lam_str}]  {mae_str}{r2_str}",
                     fontsize=9, fontweight='bold', pad=6)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss',  fontsize=8)
        ax.set_xlim(0.5, n_ep + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8)

    # Cacher cases vides
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f'{phase_name} - Courbes Train / Val / Test par configuration'
        ' | Ligne rouge = best val epoch | ratio = Val/Train au best epoch',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = output_dir / f'{slug}_individual_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Courbes individuelles sauvegardées : {path}")
    return path


def plot_results(all_results, output_dir, phase_name):
    """
    Figure 2 : tableau de bord comparatif (vue d'ensemble).
    - Haut gauche  : toutes les Val Loss superposées
    - Haut milieu  : toutes les Test Loss superposées
    - Haut droite  : MAE turbidité TEST (barres)
    - Bas gauche   : Train / Val / Test loss du MEILLEUR modèle
    - Bas milieu   : pH MAE TEST (barres)
    - Bas droite   : ratio Val/Train au dernier epoch (overfitting)
    """
    n     = len(all_results)
    names = [r['config_name'] for r in all_results]
    slug  = phase_name.lower().replace(" ", "_").replace("/", "_")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{phase_name} - Vue d\'ensemble comparative', fontsize=14, fontweight='bold')

    PALETTE = plt.cm.tab10.colors

    # ── [0,0] Val Loss superposées ────────────────────────────────────────────
    ax = axes[0, 0]
    for i, r in enumerate(all_results):
        ep = range(1, len(r['history']['val_losses']) + 1)
        ax.plot(ep, r['history']['val_losses'],
                label=r['config_name'], color=PALETTE[i % 10],
                linewidth=2, marker='o', markersize=3)
        ax.axvline(x=r['best_epoch'], color=PALETTE[i % 10],
                   linestyle=':', linewidth=1, alpha=0.6)
    ax.set_title("Val Loss - toutes configs")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Loss')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── [0,1] Test Loss superposées ───────────────────────────────────────────
    ax = axes[0, 1]
    for i, r in enumerate(all_results):
        tl = r['history'].get('test_losses', [])
        if tl:
            ep = range(1, len(tl) + 1)
            ax.plot(ep, tl, label=r['config_name'], color=PALETTE[i % 10],
                    linewidth=2, marker='^', markersize=3, linestyle='--')
            ax.axvline(x=r['best_epoch'], color=PALETTE[i % 10],
                       linestyle=':', linewidth=1, alpha=0.6)
    ax.set_title("Test Loss - toutes configs")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Loss')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── [0,2] MAE turbidité TEST (barres) ─────────────────────────────────────
    ax = axes[0, 2]
    mae_vals = [r['test_metrics']['turbidity_NTU']['mae'] or 0 for r in all_results]
    best_mae = min(mae_vals)
    colors   = ['#4CAF50' if abs(m - best_mae) < 1e-6 else '#90CAF9' for m in mae_vals]
    bars     = ax.bar(range(n), mae_vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('MAE (NTU)'); ax.set_title('Turbidité - MAE TEST (vert = meilleur)')
    ax.grid(True, alpha=0.3, axis='y')

    # ── [1,0] Courbe complète du meilleur modèle ──────────────────────────────
    ax    = axes[1, 0]
    best  = min(all_results, key=lambda x: x['test_metrics']['turbidity_NTU']['mae'] or 999)
    ep    = range(1, len(best['history']['train_losses']) + 1)
    ax.plot(ep, best['history']['train_losses'], label='Train', color='#2196F3',
            linewidth=2, marker='o', markersize=3)
    ax.plot(ep, best['history']['val_losses'],   label='Val',   color='#FF9800',
            linewidth=2, marker='s', markersize=3)
    if best['history'].get('test_losses'):
        ax.plot(ep, best['history']['test_losses'], label='Test', color='#4CAF50',
                linewidth=2, marker='^', markersize=3, linestyle='--')
    ax.axvline(x=best['best_epoch'], color='red', linestyle=':', linewidth=2,
               label=f'Best epoch {best["best_epoch"]}')
    ax.set_title(f'Meilleur modèle : {best["config_name"]}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1,1] Effet lambda_pH sur MAE turbidité (courbe + barres) ───────────
    ax = axes[1, 1]
    # Trier par lambda_pH croissant pour la lisibilité
    sorted_r = sorted(all_results, key=lambda r: r['lambdas']['pH'])
    lph_vals = [r['lambdas']['pH'] for r in sorted_r]
    mae_turb_sorted = [r['test_metrics']['turbidity_NTU']['mae'] or 0 for r in sorted_r]
    ref_mae  = next((r['test_metrics']['turbidity_NTU']['mae'] or 0
                     for r in sorted_r if r['lambdas']['pH'] == 0.0), None)
    best_mae_val = min(mae_turb_sorted)
    colors_ph = ['#4CAF50' if abs(m - best_mae_val) < 1e-6 else '#90CAF9'
                 for m in mae_turb_sorted]
    bars2 = ax.bar(range(len(sorted_r)), mae_turb_sorted,
                   color=colors_ph, edgecolor='white', linewidth=0.5)
    for bar, val, lp in zip(bars2, mae_turb_sorted, lph_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    if ref_mae:
        ax.axhline(ref_mae, color='#C62828', linestyle='--',
                   linewidth=1.2, label=f'Référence (pH=0) = {ref_mae:.4f}')
        ax.legend(fontsize=7)
    ax.set_xticks(range(len(sorted_r)))
    ax.set_xticklabels([f"pH={lp}" for lp in lph_vals], rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('MAE turbidité (NTU)')
    ax.set_title('Effet lpH -> MAE turbidite TEST (vert=meilleur, rouge=ref sans pH)')
    ph_mae  = [r['test_metrics']['pH']['mae'] or 0 for r in all_results]
    best_ph = min(ph_mae)
    colors  = ['#4CAF50' if abs(m - best_ph) < 1e-6 else '#FFCC80' for m in ph_mae]
    bars    = ax.bar(range(n), ph_mae, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ph_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('MAE (pH)'); ax.set_title('pH - MAE TEST (vert = meilleur)')
    ax.grid(True, alpha=0.3, axis='y')

    # ── [1,2] Ratio overfitting Val/Train au dernier epoch ────────────────────
    ax = axes[1, 2]
    ratios = []
    for r in all_results:
        tr  = r['history']['train_losses'][-1] if r['history']['train_losses'] else 1
        vl  = r['history']['val_losses'][-1]   if r['history']['val_losses']   else 1
        ratios.append(round(vl / tr, 2) if tr > 0 else 0)
    colors = ['#EF9A9A' if rat > 3.0 else '#A5D6A7' for rat in ratios]
    bars   = ax.bar(range(n), ratios, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=7)
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5, label='Seuil 3x')
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('Val Loss / Train Loss')
    ax.set_title('Ratio overfitting (rouge > 3x)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = output_dir / f'{slug}_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("ablation_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("🔬 ÉTUDE V3 : ABLATION λ_DO — SPLIT TEMPOREL — pH=0 & TEMP=0 FIXES")
    logger.info(f"📁 Résultats dans : {output_dir}/")
    logger.info(f"🕐 Démarrage      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # ==========================================================================
    # PHASE 1 : ABLATION
    # ==========================================================================

    logger.info("\n" + "#" * 80)
    logger.info("# ABLATION lambda_DO — pH=0, TEMP=0, TURB=1.0 FIXES")
    logger.info("# Objectif : valider DO=0.2 et trouver le lambda_DO optimal")
    logger.info("# Grille : DO in {0.0, 0.2, 0.4, 0.5, 0.7, 0.9}")
    logger.info("#" * 80)

    # ── Règle absolue : DO=0.0 et temperature_C=0.0 dans TOUTES les configs ──
    # lambda_turb = 1.0 fixe (référence) sauf config de référence pH pur
    # lambda_pH varie de 0.0 à 1.0 pour couvrir tout l'espace utile
    ablation_configs = {
        '1_turb_only': {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.0},
        '2_DO_02':     {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.2},
        '3_DO_04':     {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.4},
        '4_DO_05':     {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.5},
        '5_DO_07':     {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.7},
        '6_DO_09':     {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.9},
    }

    NUM_EPOCHS = 10   # max - early stopping (patience=4) arrêtera avant si convergé
    est_h = len(ablation_configs) * NUM_EPOCHS * 14 / 60
    logger.info(f"⏱️  Temps estimé : ~{est_h:.1f}h max  ({len(ablation_configs)} configs x {NUM_EPOCHS} epochs x 14 min)")
    logger.info(f"   (early stopping patience=4 -> temps réel probablement ~{est_h*0.75:.1f}h)\n")

    all_results = []
    t0 = datetime.now()

    for i, (config_name, lambdas) in enumerate(ablation_configs.items(), 1):
        logger.info(f"\n[{i}/{len(ablation_configs)}] Démarrage : {config_name}")
        result = train_model(config_name, lambdas, NUM_EPOCHS, logger=logger)
        all_results.append(result)

        # Sauvegarde intermédiaire après chaque modèle
        with open(output_dir / 'ablation_partial.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"💾 Sauvegarde intermédiaire ({i}/{len(ablation_configs)} modèles)")

    elapsed = (datetime.now() - t0).total_seconds() / 60
    logger.info(f"\n⏱️  Ablation terminée en {elapsed:.1f} min ({elapsed/60:.1f}h)")

    # Sauvegarde finale
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Graphiques et tableau
    create_comparison_table(all_results, "ABLATION TURB/pH", logger)
    plot_path       = plot_results(all_results, output_dir, "Ablation Turb pH")
    plot_path_indiv = plot_individual_curves(all_results, output_dir, "Ablation Turb pH")
    logger.info(f"📊 Graphique comparatif  : {plot_path}")
    logger.info(f"📊 Courbes individuelles : {plot_path_indiv}")

    # ── Analyse : effet de lambda_pH sur MAE turbidité ─────────────────────
    logger.info(f"\n{'='*80}")
    logger.info("📊 EFFET DE LAMBDA_DO SUR MAE TURBIDITE (TEST)")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Config':<20} {'lambda_DO':>10} {'MAE turb TEST':>15} {'vs turb_only':>14}")
    logger.info(f"  {'-'*62}")

    ref = next((r for r in all_results if r['config_name'] == '1_turb_only'), None)
    ref_mae = ref['test_metrics']['turbidity_NTU']['mae'] if ref else None

    sorted_res = sorted(all_results, key=lambda r: r['lambdas']['DO_mgL'])
    for r in sorted_res:
        mae  = r['test_metrics']['turbidity_NTU']['mae'] or 0
        ldo  = r['lambdas']['DO_mgL']
        if ref_mae:
            delta = mae - ref_mae
            flag  = " ✅ meilleur" if delta < -0.005 else (" ❌ pire" if delta > 0.005 else " ≈ egal")
            logger.info(f"  {r['config_name']:<20} {ldo:>10.1f} {mae:>15.4f} {delta:>+10.4f}{flag}")
        else:
            logger.info(f"  {r['config_name']:<20} {ldo:>10.1f} {mae:>15.4f}")

    # ── Meilleur modèle ────────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info("🏆 RAPPORT FINAL")
    logger.info(f"{'='*80}")

    best = min(all_results, key=lambda x: x['test_metrics']['turbidity_NTU']['mae'] or 999)
    m    = best['test_metrics']['turbidity_NTU']

    logger.info(f"\n  Meilleur modèle : {best['config_name']}")
    logger.info(f"  Best epoch      : {best['best_epoch']}")
    logger.info(f"  Best val_loss   : {best['best_val_loss']:.4f}")
    logger.info(f"  MAE  turb TEST  : {m['mae']:.4f} NTU")
    logger.info(f"  RMSE turb TEST  : {m['rmse']:.4f} NTU")
    logger.info(f"  R²   turb TEST  : {m['r2']:.4f}" if m['r2'] else "  R²   turb TEST  : NaN")

    logger.info("\n📝 CONFIGURATION RECOMMANDÉE POUR TRAIN.PY :")
    logger.info("  loss_config = {")
    logger.info("      'lambdas': {")
    for param, value in best['lambdas'].items():
        logger.info(f"          '{param}': {value},")
    logger.info("      },")
    logger.info("      'delta': 1.0")
    logger.info("  }")

    logger.info(f"\n⏱️  Temps total : {elapsed:.0f} min ({elapsed/60:.1f}h)")
    logger.info(f"📁 Fichiers dans : {output_dir}/")
    logger.info(f"   - ablation_log.txt              ← log complet")
    logger.info(f"   - ablation_results.json         ← historique train/val/test losses")
    logger.info(f"   - ablation_partial.json         ← checkpoint (reprise si crash)")
    logger.info(f"   - ablation_turb_pH_comparison.png        ← vue d'ensemble")
    logger.info(f"   - ablation_turb_pH_individual_curves.png ← courbes par config")
    logger.info("\n✅ ABLATION TERMINÉE")


if __name__ == "__main__":
    main()