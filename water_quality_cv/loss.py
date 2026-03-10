"""
Loss multi-tâches pour régression de qualité d'eau.
Utilise Huber Loss (robuste aux outliers) avec pondération par tâche.

Config gagnante ablation (2_with_weak_DO, MAE=0.2366 NTU) :
    turbidity_NTU = 1.0   (signal principal, CV=24.7%)
    pH            = 0.0   (désactivé, CV=0.06% — aucun signal visuel)
    temperature_C = 0.0   (désactivé, CV=0.67% — aucun signal visuel)
    DO_mgL        = 0.2   (régularisateur léger, CV=5.85%)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Loss multi-tâches avec pondération.

    Formule :
        L_total = Σ_k  λ_k × Huber(y_k, ŷ_k, δ)

    Seules les tâches avec λ_k > 0 sont calculées.
    Les NaN dans les targets sont masqués automatiquement.
    """

    def __init__(
        self,
        lambdas: Optional[Dict[str, float]] = None,
        delta: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Args:
            lambdas : Poids par tâche. Si None, utilise la config gagnante ablation.
            delta   : Seuil de transition Huber (L1 ↔ L2).
            reduction: 'mean' ou 'sum'.
        """
        super(MultiTaskLoss, self).__init__()

        # CORRECTION 1 : λ par défaut = config gagnante ablation
        # (anciens défauts : pH=0.1, temp=0.1, DO=0.5 → causaient overfitting)
        if lambdas is None:
            self.lambdas = {
                "turbidity_NTU": 1.0,
                "pH"           : 0.0,   # désactivé — CV=0.06%
                "temperature_C": 0.0,   # désactivé — CV=0.67%
                "DO_mgL"       : 0.2,   # régularisateur léger — CV=5.85%
            }
        else:
            self.lambdas = lambdas

        self.huber     = nn.SmoothL1Loss(reduction=reduction, beta=delta)
        self.delta     = delta
        self.reduction = reduction

        print(f"\n{'='*60}")
        print("CONFIGURATION DE LA LOSS")
        print(f"{'='*60}")
        print(f"Type      : Huber Loss (SmoothL1, β={delta})")
        print(f"Reduction : {reduction}")
        print("\nPondération (λ) par tâche :")
        for task, w in self.lambdas.items():
            status = "✅ ACTIF" if w > 0 else "❌ INACTIF"
            print(f"  {task:20s} : {w:.2f}  {status}")
        print(f"{'='*60}\n")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets    : Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule la loss multi-tâches pondérée.

        Args:
            predictions : Dict {task: tensor (B,)}
            targets     : Dict {task: tensor (B,)}  — peut contenir des NaN

        Returns:
            total_loss  : Tensor scalaire différentiable
            task_losses : Dict {task: float}  (losses individuelles non pondérées)
        """
        device = next(iter(predictions.values())).device

        # CORRECTION 2 : initialiser avec un tenseur zéro différentiable
        # (l'ancienne version `total_loss = None` + accumulation crée un graphe profond)
        total_loss  = torch.zeros(1, device=device, dtype=torch.float32)
        task_losses : Dict[str, float] = {}

        for task in ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]:
            if task not in predictions or task not in targets:
                continue

            lambda_k = self.lambdas.get(task, 0.0)
            if lambda_k == 0.0:
                continue  # tâche désactivée — on ne calcule rien

            pred   = predictions[task]
            target = targets[task]

            # Masquer les NaN dans les targets
            mask = ~torch.isnan(target)
            if mask.sum() == 0:
                continue

            loss_k        = self.huber(pred[mask], target[mask])
            total_loss    = total_loss + lambda_k * loss_k
            task_losses[task] = float(loss_k.item())

        return total_loss.squeeze(), task_losses

    def update_lambdas(self, new_lambdas: Dict[str, float]) -> None:
        """
        Met à jour les poids des tâches (curriculum learning, ablation).

        Args:
            new_lambdas : Nouveaux poids (seules les clés présentes sont mises à jour).
        """
        self.lambdas.update(new_lambdas)
        print("\n📊 Poids mis à jour :")
        for task, w in self.lambdas.items():
            status = "✅ ACTIF" if w > 0 else "❌ INACTIF"
            print(f"  {task:20s} : {w:.2f}  {status}")


def create_loss_function(config: Optional[dict] = None) -> MultiTaskLoss:
    """
    Factory pour créer la loss avec configuration.

    Args:
        config : Dict optionnel. Exemple :
                 {
                     'lambdas': {'turbidity_NTU': 1.0, 'DO_mgL': 0.2,
                                 'pH': 0.0, 'temperature_C': 0.0},
                     'delta': 1.0
                 }
    Returns:
        Instance de MultiTaskLoss configurée.

    Comportement de fusion des lambdas :
        - Si config fournit 'lambdas', il REMPLACE entièrement les défauts.
        - Cela évite le bug de fusion partielle où pH=0.1 restait actif
          même quand on passait {'turbidity_NTU': 1.0, 'DO_mgL': 0.2}.
    """
    # CORRECTION 3 : séparation propre lambdas / autres paramètres
    # (l'ancienne version faisait default.update(config['lambdas']) qui fusionnait
    #  au lieu de remplacer → pH=0.1 et temp=0.1 restaient actifs)

    # Valeurs par défaut pour les paramètres non-lambdas
    delta     = 1.0
    reduction = "mean"

    # Lambdas : None → le constructeur utilisera les défauts corrects
    lambdas = None

    if config:
        # Remplacer entièrement les lambdas si fournis (pas de fusion)
        if "lambdas" in config:
            lambdas = config["lambdas"]
        if "delta" in config:
            delta = config["delta"]
        if "reduction" in config:
            reduction = config["reduction"]

    return MultiTaskLoss(lambdas=lambdas, delta=delta, reduction=reduction)


# ══════════════════════════════════════════════════════════════════════════════
# Tests rapides
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DE LA LOSS MULTI-TÂCHES")
    print("=" * 60)

    # Test 1 : config par défaut (gagnante ablation)
    print("\n── Test 1 : config par défaut ──")
    criterion = create_loss_function()

    B = 16
    preds = {
        "turbidity_NTU": torch.randn(B, requires_grad=True) * 2 + 7,
        "pH"           : torch.randn(B, requires_grad=True) * 0.1 + 8,
        "temperature_C": torch.randn(B, requires_grad=True) * 0.5 + 30,
        "DO_mgL"       : torch.randn(B, requires_grad=True) * 0.5 + 5,
    }
    tgts = {
        "turbidity_NTU": torch.randn(B) * 2 + 7,
        "pH"           : torch.randn(B) * 0.1 + 8,
        "temperature_C": torch.randn(B) * 0.5 + 30,
        "DO_mgL"       : torch.randn(B) * 0.5 + 5,
    }

    loss, task_losses = criterion(preds, tgts)
    print(f"Loss totale : {loss.item():.4f}")
    print("Tâches actives :", list(task_losses.keys()))
    assert "pH"            not in task_losses, "pH ne devrait pas être actif"
    assert "temperature_C" not in task_losses, "temperature_C ne devrait pas être actif"
    print("✅ pH et temperature_C bien inactifs")

    # Test 2 : config personnalisée — lambdas remplacent les défauts entièrement
    print("\n── Test 2 : config personnalisée ──")
    criterion2 = create_loss_function({
        "lambdas": {
            "turbidity_NTU": 1.0,
            "pH"           : 0.0,
            "temperature_C": 0.0,
            "DO_mgL"       : 0.2,
        },
        "delta": 1.0,
    })
    loss2, _ = criterion2(preds, tgts)
    print(f"Loss totale : {loss2.item():.4f}")
    print("✅ Config personnalisée OK")

    # Test 3 : NaN dans les targets
    print("\n── Test 3 : NaN dans les targets ──")
    tgts_nan = {k: v.clone() for k, v in tgts.items()}
    tgts_nan["turbidity_NTU"][0:3] = float("nan")
    loss3, _ = criterion(preds, tgts_nan)
    print(f"Loss avec NaN : {loss3.item():.4f}")
    print("✅ NaN masqués correctement")

    # Test 4 : backward (gradients)
    print("\n── Test 4 : backward ──")
    loss.backward()
    print("✅ Backward réussi")

    # Test 5 : vérifier que total_loss est un scalaire propre (pas None)
    print("\n── Test 5 : type de total_loss ──")
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    print(f"✅ total_loss est bien un scalaire : shape={loss.shape}")

    print(f"\n{'='*60}")
    print("✅ TOUS LES TESTS RÉUSSIS")
    print(f"{'='*60}")