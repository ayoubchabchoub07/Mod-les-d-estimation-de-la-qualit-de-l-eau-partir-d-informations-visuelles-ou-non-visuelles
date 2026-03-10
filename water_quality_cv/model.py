"""
Architecture multi-tâches pour estimation de qualité d'eau
Backbone : EfficientNet-B2 (pré-entraîné ImageNet)
Têtes : 4 MLPs de régression (turbidité, pH, température, DO)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import timm


class WaterQualityModel(nn.Module):
    """
    Modèle multi-tâches pour prédiction de 4 paramètres d'eau

    Architecture :
    1. Backbone EfficientNet-B2 (features partagées)
    2. Global Average Pooling
    3. 4 têtes MLP indépendantes (une par paramètre)
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            pretrained: Utiliser les poids ImageNet
            dropout_rate: Taux de dropout dans les têtes
            freeze_backbone: Geler le backbone (fine-tuning des têtes uniquement)
        """
        super(WaterQualityModel, self).__init__()

        print(f"📦 Chargement EfficientNet-B2 (pretrained={pretrained})...")
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        self.feature_dim = 1408

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head_turbidity = self._make_regression_head(
            in_features=self.feature_dim,
            dropout_rate=dropout_rate,
            head_name="turbidity",
        )

        self.head_pH = self._make_regression_head(
            in_features=self.feature_dim,
            dropout_rate=dropout_rate,
            head_name="pH",
        )

        self.head_temperature = self._make_regression_head(
            in_features=self.feature_dim,
            dropout_rate=dropout_rate,
            head_name="temperature",
        )

        self.head_DO = self._make_regression_head(
            in_features=self.feature_dim,
            dropout_rate=dropout_rate,
            head_name="DO",
        )

        if freeze_backbone:
            print("🔒 Backbone gelé (seules les têtes seront entraînées)")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._print_model_info()

    def _make_regression_head(
        self, in_features: int, dropout_rate: float, head_name: str
    ) -> nn.Sequential:
        """
        Crée une tête MLP pour régression

        Architecture : Linear(1408 → 256) → ReLU → Dropout → Linear(256 → 1)

        Args:
            in_features: Dimension d'entrée (1408 pour EfficientNet-B2)
            dropout_rate: Taux de dropout
            head_name: Nom de la tête (pour debug)

        Returns:
            Module Sequential
        """
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Tensor (batch_size, 3, 260, 260)

        Returns:
            Dict {
                'turbidity_NTU': tensor (batch_size,),
                'pH': tensor (batch_size,),
                'temperature_C': tensor (batch_size,),
                'DO_mgL': tensor (batch_size,)
            }
        """
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.flatten(1)

        turbidity = self.head_turbidity(features).squeeze(1)
        pH = self.head_pH(features).squeeze(1)
        temperature = self.head_temperature(features).squeeze(1)
        DO = self.head_DO(features).squeeze(1)

        return {
            "turbidity_NTU": turbidity,
            "pH": pH,
            "temperature_C": temperature,
            "DO_mgL": DO,
        }

    def _print_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        print(f"\n{'='*80}")
        print("ARCHITECTURE DU MODÈLE")
        print(f"{'='*80}")
        print("Backbone         : EfficientNet-B2")
        print(f"Feature dim      : {self.feature_dim}")
        print("Têtes            : 4 × MLP (256 hidden units)")
        print(f"\nParamètres totaux      : {total_params:,}")
        print(f"Paramètres entraînables : {trainable_params:,}")
        print(f"Paramètres gelés        : {total_params - trainable_params:,}")
        print(f"{'='*80}\n")


def create_model(config: Optional[dict] = None) -> WaterQualityModel:
    """
    Factory pour créer le modèle avec configuration

    Args:
        config: Dict de configuration (optionnel)

    Returns:
        Instance de WaterQualityModel
    """
    default_config = {
        "pretrained": True,
        "dropout_rate": 0.3,
        "freeze_backbone": False,
    }

    if config:
        default_config.update(config)

    model = WaterQualityModel(**default_config)
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("TEST DU MODÈLE")
    print("=" * 80)

    model = create_model(
        {
            "pretrained": True,
            "dropout_rate": 0.3,
            "freeze_backbone": False,
        }
    )

    print("\n🔍 Test avec un batch fictif...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 260, 260)

    print(f"   Input shape : {dummy_input.shape}")

    with torch.no_grad():
        outputs = model(dummy_input)

    print("\n✅ Forward pass réussi !")
    print("\n   Outputs :")
    for param_name, values in outputs.items():
        print(
            f"     {param_name:15s} : shape={values.shape}, "
            f"dtype={values.dtype}"
        )

    print("\n🔍 Vérification des dimensions...")
    all_correct = True
    for param_name, values in outputs.items():
        expected_shape = (batch_size,)
        if values.shape != expected_shape:
            print(
                f"   ❌ {param_name}: attendu {expected_shape}, obtenu {values.shape}"
            )
            all_correct = False

    if all_correct:
        print("   ✅ Toutes les dimensions sont correctes")

    print(f"\n{'='*80}")
    print("✅ MODÈLE OPÉRATIONNEL")
    print(f"{'='*80}")
