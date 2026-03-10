#!/usr/bin/env python3
"""
ÉTUDE COMPLÈTE : ABLATION + OPTIMISATION
Phase 1 : Vraie ablation (mono-task vs multi-task)
Phase 2 : Optimisation fine des poids
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
    log("🎯 Poids (λ) :")
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
    history          = {'train_losses': [], 'val_losses': []}

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

        scheduler.step()
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            saved_marker     = " ✅ BEST"
        else:
            saved_marker = ""

        # ETA
        elapsed_s  = (datetime.now() - epoch_start).total_seconds()
        per_epoch  = elapsed_s / (epoch + 1)
        remaining  = per_epoch * (num_epochs - epoch - 1)
        eta_str    = f"ETA ~{remaining/60:.0f} min"

        log(f"  Epoch {epoch+1:02d}/{num_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}{saved_marker}  [{eta_str}]")

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


def plot_results(all_results, output_dir, phase_name):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{phase_name} - Comparaison des Modèles', fontsize=16, fontweight='bold')

    # 1. Courbes val loss
    ax = axes[0, 0]
    for r in all_results:
        ax.plot(r['history']['val_losses'], label=r['config_name'], marker='o', markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Loss')
    ax.set_title("Courbes d'apprentissage"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Turbidité MAE
    ax = axes[0, 1]
    names    = [r['config_name'] for r in all_results]
    mae_vals = [r['test_metrics']['turbidity_NTU']['mae'] or 0 for r in all_results]
    colors   = ['green' if m == min(mae_vals) else 'steelblue' for m in mae_vals]
    ax.bar(range(len(names)), mae_vals, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE (NTU)'); ax.set_title('Turbidité - MAE (TEST)'); ax.grid(True, alpha=0.3, axis='y')

    # 3. Turbidité R²
    ax = axes[1, 0]
    r2_vals = [r['test_metrics']['turbidity_NTU']['r2'] or 0 for r in all_results]
    colors  = ['green' if r2 == max(r2_vals) else 'steelblue' for r2 in r2_vals]
    ax.bar(range(len(names)), r2_vals, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('R²'); ax.set_title('Turbidité - R² (TEST)'); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3, axis='y')

    # 4. DO MAE
    ax = axes[1, 1]
    do_mae = [r['test_metrics']['DO_mgL']['mae'] or 0 for r in all_results]
    colors = ['green' if m == min(do_mae) else 'coral' for m in do_mae]
    ax.bar(range(len(names)), do_mae, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE (mg/L)'); ax.set_title('DO - MAE (TEST)'); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / f'{phase_name.lower().replace(" ", "_").replace("/","_")}_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def analyze_multitask_benefit(ablation_results, logger=None):
    def log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    turb_mono = next((r for r in ablation_results if '1_turbidity_only' in r['config_name']), None)
    do_mono   = next((r for r in ablation_results if '4_DO_only'        in r['config_name']), None)
    multi     = next((r for r in ablation_results if '5_multi_task'     in r['config_name']), None)

    if not all([turb_mono, do_mono, multi]):
        log("⚠️  Analyse multi-task impossible : résultats manquants")
        return "turbidity_only"

    turb_imp = (turb_mono['test_metrics']['turbidity_NTU']['mae'] - multi['test_metrics']['turbidity_NTU']['mae']) \
               / turb_mono['test_metrics']['turbidity_NTU']['mae'] * 100
    do_imp   = (do_mono['test_metrics']['DO_mgL']['mae'] - multi['test_metrics']['DO_mgL']['mae']) \
               / do_mono['test_metrics']['DO_mgL']['mae'] * 100

    log(f"\n{'='*80}")
    log("🔬 ANALYSE : BÉNÉFICE MULTI-TASK")
    log(f"{'='*80}")
    log(f"  Turbidité  : {'✅ +' if turb_imp > 0 else '❌ '}{turb_imp:.1f}%")
    log(f"  DO         : {'✅ +' if do_imp  > 0 else '❌ '}{do_imp:.1f}%")

    if turb_imp > 0 and do_imp > 0:
        rec = "multi_task"
        log("  💡 Verdict : Multi-task BÉNÉFIQUE pour les deux → recommandé")
    elif turb_imp > 0:
        rec = "multi_task"
        log("  💡 Verdict : Multi-task améliore turbidité → recommandé (prioritaire)")
    else:
        rec = "turbidity_only"
        log("  💡 Verdict : Multi-task dégrade turbidité → turbidity_only recommandé")

    return rec


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("ablation_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("🔬 ÉTUDE COMPLÈTE : ABLATION + OPTIMISATION")
    logger.info(f"📁 Résultats dans : {output_dir}/")
    logger.info(f"🕐 Démarrage      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # ==========================================================================
    # PHASE 1 : ABLATION
    # ==========================================================================

    logger.info("\n" + "#" * 80)
    logger.info("# PHASE 1 : ABLATION (5 MODÈLES × 15 EPOCHS)")
    logger.info("#" * 80)

    ablation_configs = {
        '1_turbidity_only': {
            'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.0
        },
        '2_pH_only': {
            'turbidity_NTU': 0.0, 'pH': 1.0, 'temperature_C': 0.0, 'DO_mgL': 0.0
        },
        '3_temperature_only': {
            'turbidity_NTU': 0.0, 'pH': 0.0, 'temperature_C': 1.0, 'DO_mgL': 0.0
        },
        '4_DO_only': {
            'turbidity_NTU': 0.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 1.0
        },
        '5_multi_task': {
            'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.8
        }
    }

    NUM_EPOCHS_P1 = 15
    est_h = len(ablation_configs) * NUM_EPOCHS_P1 * 8 / 60
    logger.info(f"⏱️  Temps estimé : ~{est_h:.1f}h  (5 modèles × 15 epochs × 8 min)\n")
    # ← Plus de input() ici

    phase1_results = []
    t0 = datetime.now()

    for i, (config_name, lambdas) in enumerate(ablation_configs.items(), 1):
        logger.info(f"\n[{i}/{len(ablation_configs)}] Démarrage : {config_name}")
        result = train_model(config_name, lambdas, NUM_EPOCHS_P1, logger=logger)
        phase1_results.append(result)

        # Sauvegarde intermédiaire après chaque modèle
        with open(output_dir / 'phase1_ablation_partial.json', 'w') as f:
            json.dump(phase1_results, f, indent=2, default=str)
        logger.info(f"💾 Sauvegarde intermédiaire OK ({i}/{len(ablation_configs)} modèles)")

    elapsed_p1 = (datetime.now() - t0).total_seconds() / 60
    logger.info(f"\n⏱️  Phase 1 terminée en {elapsed_p1:.1f} min")

    with open(output_dir / 'phase1_ablation.json', 'w') as f:
        json.dump(phase1_results, f, indent=2, default=str)

    create_comparison_table(phase1_results, "PHASE 1 - ABLATION", logger)
    plot_path = plot_results(phase1_results, output_dir, "Phase 1 Ablation")
    logger.info(f"📊 Graphique Phase 1 : {plot_path}")

    recommendation = analyze_multitask_benefit(phase1_results, logger)

    # ==========================================================================
    # PHASE 2 : OPTIMISATION DES POIDS
    # ==========================================================================

    logger.info("\n" + "#" * 80)
    logger.info("# PHASE 2 : OPTIMISATION DES POIDS")
    logger.info("#" * 80)
    # ← Plus de input() ici

    if recommendation == "multi_task":
        logger.info("✅ Multi-task retenu → optimisation des λ\n")
        optimization_configs = {
            '1_baseline_multi':  {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.8},
            '2_DO_equal':        {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 1.0},
            '3_DO_lower':        {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.5},
            '4_turb_dominant':   {'turbidity_NTU': 1.5, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.5},
            '5_DO_favored':      {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 1.2},
        }
    else:
        logger.info("⚠️  turbidity_only retenu → optimisation mono-task\n")
        optimization_configs = {
            '1_baseline_turb':   {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.0},
            '2_with_weak_DO':    {'turbidity_NTU': 1.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.2},
            '3_turb_boosted':    {'turbidity_NTU': 2.0, 'pH': 0.0, 'temperature_C': 0.0, 'DO_mgL': 0.0},
        }

    NUM_EPOCHS_P2 = 10
    est_h2 = len(optimization_configs) * NUM_EPOCHS_P2 * 8 / 60
    logger.info(f"⏱️  Temps estimé phase 2 : ~{est_h2:.1f}h\n")

    phase2_results = []
    t0 = datetime.now()

    for i, (config_name, lambdas) in enumerate(optimization_configs.items(), 1):
        logger.info(f"\n[{i}/{len(optimization_configs)}] Démarrage : {config_name}")
        result = train_model(config_name, lambdas, NUM_EPOCHS_P2, logger=logger)
        phase2_results.append(result)

        with open(output_dir / 'phase2_optimization_partial.json', 'w') as f:
            json.dump(phase2_results, f, indent=2, default=str)
        logger.info(f"💾 Sauvegarde intermédiaire OK ({i}/{len(optimization_configs)} modèles)")

    elapsed_p2 = (datetime.now() - t0).total_seconds() / 60
    logger.info(f"\n⏱️  Phase 2 terminée en {elapsed_p2:.1f} min")

    with open(output_dir / 'phase2_optimization.json', 'w') as f:
        json.dump(phase2_results, f, indent=2, default=str)

    create_comparison_table(phase2_results, "PHASE 2 - OPTIMISATION", logger)
    plot_path2 = plot_results(phase2_results, output_dir, "Phase 2 Optimisation")
    logger.info(f"📊 Graphique Phase 2 : {plot_path2}")

    # ==========================================================================
    # RAPPORT FINAL
    # ==========================================================================

    logger.info("\n" + "#" * 80)
    logger.info("# RAPPORT FINAL")
    logger.info("#" * 80)

    all_results = phase1_results + phase2_results
    best = min(all_results, key=lambda x: x['test_metrics']['turbidity_NTU']['mae'] or 999)

    logger.info(f"\n🏆 MEILLEUR MODÈLE : {best['config_name']}")
    logger.info(f"   Best epoch   : {best['best_epoch']}")
    logger.info(f"   Val Loss     : {best['best_val_loss']:.4f}")
    m = best['test_metrics']['turbidity_NTU']
    logger.info(f"   Turbidité TEST → MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    logger.info("\n📝 CONFIGURATION RECOMMANDÉE POUR TRAIN.PY :")
    logger.info("criterion = create_loss_function({")
    logger.info("    'lambdas': {")
    for param, value in best['lambdas'].items():
        logger.info(f"        '{param}': {value},")
    logger.info("    },")
    logger.info("    'delta': 1.0")
    logger.info("})")

    total_min = elapsed_p1 + elapsed_p2
    logger.info(f"\n⏱️  Temps total : {total_min:.0f} min ({total_min/60:.1f}h)")
    logger.info(f"📁 Tous les fichiers dans : {output_dir}/")
    logger.info(f"   - ablation_log.txt              ← ce log complet")
    logger.info(f"   - phase1_ablation.json")
    logger.info(f"   - phase2_optimization.json")
    logger.info(f"   - phase_1_ablation_comparison.png")
    logger.info(f"   - phase_2_optimisation_comparison.png")
    logger.info("\n✅ ÉTUDE COMPLÈTE TERMINÉE")


if __name__ == "__main__":
    main()
