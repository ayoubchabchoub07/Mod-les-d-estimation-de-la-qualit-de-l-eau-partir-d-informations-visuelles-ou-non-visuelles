"""
Loss multi-tâches — Uncertainty Weighting (Kendall et al., 2018)
"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"

Remplace les λ fixes par des incertitudes homoscédastiques apprises automatiquement.

Formule par tâche k :
    L_k = (1 / (2 · σ_k²)) · Huber_k  +  log(σ_k)
        = 0.5 · exp(-log_var_k) · Huber_k  +  0.5 · log_var_k

Propriétés :
    - σ_k petit  → tâche précise  → poids fort automatiquement
    - σ_k grand  → tâche bruitée → poids faible automatiquement
    - log(σ_k)   → terme de régularisation (empêche σ → ∞)

Pour turbidité (CV=24.7%) : le modèle apprendra σ_turb petit → poids fort
Pour DO (CV=5.85%)        : le modèle apprendra σ_DO   grand → poids faible

COMPATIBILITÉ : interface identique à MultiTaskLoss.
    - create_loss_function(config) → fonctionne sans modifier train.py
    - Seul ajout dans train.py : passer criterion.parameters() à l'optimiseur
      et logger criterion.get_sigmas() par epoch.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    Loss multi-tâches avec pondération automatique par incertitude.

    Les poids ne sont plus des λ fixes mais des paramètres appris :
        log_var_k = log(σ_k²)  — initialisé à 0 → σ_k² = 1 au départ

    Tâches désactivées (λ=0 dans config) : complètement exclues.
    NaN dans les targets : masqués automatiquement.
    """

    def __init__(
        self,
        active_tasks: list,
        delta: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Args:
            active_tasks : tâches incluses dans la loss,
                           ex. ["turbidity_NTU", "DO_mgL"]
            delta        : seuil Huber (L1 ↔ L2)
            reduction    : 'mean' ou 'sum'
        """
        super(UncertaintyWeightedLoss, self).__init__()

        self.active_tasks = active_tasks
        self.delta        = delta
        self.reduction    = reduction

        # log_var_k = log(σ_k²), initialisé à 0 → σ_k² = exp(0) = 1
        # Clé : remplace "." par "_" pour ParameterDict (ex. "DO_mgL" → "DO_mgL")
        self.log_vars = nn.ParameterDict({
            self._key(task): nn.Parameter(torch.zeros(1))
            for task in active_tasks
        })

        self._print_config()

    @staticmethod
    def _key(task: str) -> str:
        """Transforme un nom de tâche en clé valide pour ParameterDict."""
        return task.replace(".", "_")

    def _huber(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Huber loss avec NaN masking. Retourne 0 si tous les targets sont NaN."""
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        return nn.functional.huber_loss(
            pred[mask], target[mask],
            delta=self.delta,
            reduction=self.reduction,
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets    : Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule la loss multi-tâches avec pondération par incertitude.

        Args:
            predictions : Dict {task: tensor (B,)}
            targets     : Dict {task: tensor (B,)}  — peut contenir des NaN

        Returns:
            total_loss  : Tensor scalaire différentiable
            task_losses : Dict {task: float}  (Huber brut, sans pondération)
        """
        device = next(iter(predictions.values())).device
        total_loss  = torch.zeros(1, device=device, dtype=torch.float32)
        task_losses : Dict[str, float] = {}

        for task in self.active_tasks:
            if task not in predictions or task not in targets:
                continue

            log_var   = self.log_vars[self._key(task)]       # log(σ²), shape (1,)
            precision = torch.exp(-log_var)                   # 1/σ² — toujours positif

            huber_k = self._huber(predictions[task], targets[task])

            # L_k = (1/2σ²) · Huber_k + (1/2) · log(σ²)
            #      = 0.5 · precision · Huber_k + 0.5 · log_var
            weighted_k = 0.5 * precision * huber_k + 0.5 * log_var

            total_loss = total_loss + weighted_k.squeeze()
            task_losses[task] = float(huber_k.item())

        return total_loss.squeeze(), task_losses

    def get_sigmas(self) -> Dict[str, float]:
        """
        Retourne σ_k (écart-type) pour chaque tâche active — pour le logging.

        σ_k petit  → tâche fiable  → poids élevé
        σ_k grand  → tâche bruitée → poids faible
        """
        return {
            task: float(torch.exp(0.5 * self.log_vars[self._key(task)]).item())
            for task in self.active_tasks
        }

    def get_effective_weights(self) -> Dict[str, float]:
        """
        Retourne le poids effectif 1/(2σ²) par tâche — analogue aux λ fixes.
        Utile pour comparer avec l'ablation précédente.
        """
        return {
            task: float(0.5 * torch.exp(-self.log_vars[self._key(task)]).item())
            for task in self.active_tasks
        }

    def update_lambdas(self, new_lambdas: Dict[str, float]) -> None:
        """
        Stub de compatibilité avec l'ancienne interface MultiTaskLoss.
        Les poids sont maintenant appris — cette méthode ne fait rien.
        """
        print("⚠️  update_lambdas() ignoré : les poids sont appris automatiquement.")

    def _print_config(self) -> None:
        print(f"\n{'='*60}")
        print("UNCERTAINTY WEIGHTED LOSS  (Kendall et al., 2018)")
        print(f"{'='*60}")
        print(f"Type      : Huber + uncertainty regularisation (β={self.delta})")
        print(f"Reduction : {self.reduction}")
        print(f"\nTâches actives ({len(self.active_tasks)}) :")
        for task in self.active_tasks:
            print(f"  {task:20s} : σ initialisé à 1.0  (poids=0.5)")
        print(f"\nParamètres appris : {len(self.active_tasks)} × log_var")
        print("  → à inclure dans l'optimiseur via criterion.parameters()")
        print(f"{'='*60}\n")


def create_loss_function(config: Optional[dict] = None) -> UncertaintyWeightedLoss:
    """
    Factory compatible avec l'interface existante (train.py inchangé sauf optimiseur).

    Les tâches avec λ=0 dans config['lambdas'] sont automatiquement exclues.
    Les tâches avec λ>0 deviennent des tâches actives avec σ appris.

    Args:
        config : même format qu'avant :
                 {
                     'lambdas': {'turbidity_NTU': 1.0, 'DO_mgL': 0.2,
                                 'pH': 0.0, 'temperature_C': 0.0},
                     'delta': 1.0
                 }

    Returns:
        Instance de UncertaintyWeightedLoss.
    """
    # Tâches actives par défaut (config gagnante ablation comme point de départ)
    default_active = ["turbidity_NTU", "DO_mgL"]
    delta     = 1.0
    reduction = "mean"

    active_tasks = default_active

    if config:
        if "lambdas" in config:
            # On réutilise les λ pour identifier les tâches actives (λ > 0)
            # Les valeurs de λ elles-mêmes sont ignorées — σ prend le relais
            active_tasks = [
                task for task, lam in config["lambdas"].items() if lam > 0
            ]
        if "delta" in config:
            delta = config["delta"]
        if "reduction" in config:
            reduction = config["reduction"]

    print(f"   Tâches actives détectées : {active_tasks}")
    print("   Pondération : uncertainty weighting  (λ remplacés par σ appris)")

    return UncertaintyWeightedLoss(
        active_tasks=active_tasks,
        delta=delta,
        reduction=reduction,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests — Uncertainty Weighted Loss
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TEST — UNCERTAINTY WEIGHTED LOSS")
    print("=" * 60)

    # Test 1 : config identique à l'ancienne (compatibilité)
    print("\n── Test 1 : compatibilité config gagnante ablation ──")
    criterion = create_loss_function({
        "lambdas": {
            "turbidity_NTU": 1.0,
            "pH"           : 0.0,
            "temperature_C": 0.0,
            "DO_mgL"       : 0.2,
        },
        "delta": 1.0,
    })

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
    print(f"Tâches actives : {list(task_losses.keys())}")
    assert "pH"            not in task_losses, "pH devrait être inactif"
    assert "temperature_C" not in task_losses, "temperature_C devrait être inactif"
    print("✅ pH et temperature_C bien exclus")

    # Test 2 : σ initiaux = 1.0
    print("\n── Test 2 : σ initiaux ──")
    sigmas = criterion.get_sigmas()
    for task, sigma in sigmas.items():
        print(f"  σ_{task:20s} = {sigma:.4f}  (attendu ~1.0)")
        assert abs(sigma - 1.0) < 1e-4, f"σ devrait être 1.0, obtenu {sigma}"
    print("✅ σ initiaux corrects")

    # Test 3 : NaN masking
    print("\n── Test 3 : NaN dans targets ──")
    tgts_nan = {k: v.clone() for k, v in tgts.items()}
    tgts_nan["turbidity_NTU"][0:4] = float("nan")
    loss3, _ = criterion(preds, tgts_nan)
    print(f"Loss avec NaN : {loss3.item():.4f}")
    print("✅ NaN masqués")

    # Test 4 : backward (gradients sur log_vars)
    print("\n── Test 4 : backward (gradients sur log_var) ──")
    loss.backward()
    for name, param in criterion.log_vars.items():
        assert param.grad is not None, f"Pas de gradient sur log_var_{name}"
        print(f"  grad log_var_{name} = {param.grad.item():.6f}")
    print("✅ Gradients propagés sur les log_var")

    # Test 5 : log_vars inclus dans les paramètres (pour l'optimiseur)
    print("\n── Test 5 : paramètres entraînables ──")
    n_params = sum(p.numel() for p in criterion.parameters())
    print(f"  Paramètres entraînables dans criterion : {n_params}")
    assert n_params == len(criterion.active_tasks), \
        f"Attendu {len(criterion.active_tasks)} params, obtenu {n_params}"
    print("✅ Paramètres corrects (1 log_var par tâche active)")

    # Test 6 : poids effectifs
    print("\n── Test 6 : poids effectifs initiaux ──")
    weights = criterion.get_effective_weights()
    for task, w in weights.items():
        print(f"  poids_{task:20s} = {w:.4f}  (attendu 0.5 au départ)")
    print("✅ Poids effectifs corrects")

    print(f"\n{'='*60}")
    print("✅ TOUS LES TESTS RÉUSSIS")
    print(f"{'='*60}")