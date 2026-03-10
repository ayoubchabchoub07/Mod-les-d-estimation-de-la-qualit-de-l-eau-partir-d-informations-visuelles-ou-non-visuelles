"""
Module de prétraitement pour images sous-marines
Implémente CLAHE + White Balance pour améliorer la qualité des images
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple


class UnderwaterPreprocessor:
    """
    Pipeline de prétraitement pour images sous-marines Tilapia-RAS
    
    Étapes :
    1. Chargement et redimensionnement
    2. White Balance (Simplest Color Balance)
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalisation ImageNet
    5. Conversion en tensor PyTorch
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (260, 260),
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        use_white_balance: bool = True,
        wb_percentile: Tuple[float, float] = (1.0, 1.0),
        normalize_imagenet: bool = True,
        return_numpy: bool = False
    ):
        """
        Args:
            target_size: Taille cible (width, height) pour EfficientNet-B2
            use_clahe: Activer CLAHE
            clahe_clip_limit: Limite de clipping CLAHE (1.0-40.0, typique: 2.0)
            clahe_tile_size: Taille des tuiles CLAHE
            use_white_balance: Activer white balance
            wb_percentile: (low, high) percentiles pour color balance
            normalize_imagenet: Normalisation ImageNet (obligatoire pour EfficientNet)
            return_numpy: Si True, retourne numpy array au lieu de tensor
        """
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.use_white_balance = use_white_balance
        self.wb_percentile = wb_percentile
        self.normalize_imagenet = normalize_imagenet
        self.return_numpy = return_numpy
        
        # Créer l'objet CLAHE (lazy-safe)
        self.clahe = None
        if self.use_clahe:
            self._init_clahe()
        
        # Paramètres ImageNet
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def _init_clahe(self) -> None:
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # cv2.CLAHE n'est pas picklable
        state["clahe"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_clahe:
            self._init_clahe()

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Applique CLAHE sur le canal L en espace LAB
        
        Args:
            image: Image RGB uint8
        
        Returns:
            Image RGB uint8 avec CLAHE appliqué
        """
        # Convertir RGB → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Appliquer CLAHE sur le canal L (luminosité)
        if self.clahe is None:
            self._init_clahe()
        l_clahe = self.clahe.apply(l)
        
        # Reconstruire l'image
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return result
    
    def apply_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Applique Simplest Color Balance
        
        Args:
            image: Image RGB uint8
        
        Returns:
            Image RGB uint8 avec white balance appliqué
        """
        result = image.astype(np.float32)
        s_low, s_high = self.wb_percentile
        
        for i in range(3):  # Pour chaque canal RGB
            channel = result[:, :, i].flatten()
            
            # Calculer les percentiles
            p_low = np.percentile(channel, s_low)
            p_high = np.percentile(channel, 100 - s_high)
            
            # Clipper et étirer
            channel = np.clip(channel, p_low, p_high)
            
            # Normaliser entre 0 et 255
            if p_high > p_low:  # Éviter division par zéro
                channel = (channel - p_low) / (p_high - p_low) * 255.0
            
            result[:, :, i] = channel.reshape(result.shape[:2])
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def load_and_resize(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Charge et redimensionne une image
        
        Args:
            image_input: Chemin vers l'image ou ndarray BGR/RGB
        
        Returns:
            Image RGB uint8 redimensionnée
        """
        # Charger l'image
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Impossible de charger l'image : {image_input}")
            # OpenCV charge en BGR → convertir en RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_input.copy()
            # Si c'est déjà un ndarray, supposer RGB
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Format d'image invalide : {image.shape}")
        
        # Redimensionner avec INTER_AREA (meilleur pour downsampling)
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        return resized
    
    def __call__(self, image_input: Union[str, Path, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Pipeline complet de prétraitement
        
        Args:
            image_input: Chemin vers l'image ou ndarray
        
        Returns:
            Tensor PyTorch (3, H, W) normalisé ou ndarray selon config
        """
        # 1. Charger et redimensionner
        image = self.load_and_resize(image_input)
        
        # 2. White Balance (avant CLAHE)
        if self.use_white_balance:
            image = self.apply_white_balance(image)
        
        # 3. CLAHE (après WB)
        if self.use_clahe:
            image = self.apply_clahe(image)
        
        # 4. Normaliser [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 5. Normalisation ImageNet
        if self.normalize_imagenet:
            image = (image - self.imagenet_mean) / self.imagenet_std
        
        # 6. Convertir en tensor ou garder numpy
        if self.return_numpy:
            return image
        else:
            # Convertir HWC → CHW pour PyTorch
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            return tensor


def create_preprocessor(config: dict = None) -> UnderwaterPreprocessor:
    """
    Factory pour créer un preprocessor avec config
    
    Args:
        config: Dictionnaire de configuration (optionnel)
    
    Returns:
        Instance de UnderwaterPreprocessor
    """
    default_config = {
        'target_size': (260, 260),
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'use_white_balance': True,
        'wb_percentile': (1.0, 1.0),
        'normalize_imagenet': True,
        'return_numpy': False
    }
    
    if config:
        default_config.update(config)
    
    return UnderwaterPreprocessor(**default_config)


# Test du module
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Créer le preprocessor
    preprocessor = UnderwaterPreprocessor(
        target_size=(260, 260),
        use_clahe=True,
        use_white_balance=True,
        return_numpy=True  # Pour visualisation
    )
    
    # Test sur une image
    test_image_path = "D:\\p2M\\Tilapia RAS Dataset\\Frames\\Original\\GX010206\\images\\GX010206_00000.jpg"  # Remplacer par un chemin réel
    
    try:
        processed = preprocessor(test_image_path)
        print(f"✅ Prétraitement réussi !")
        print(f"   Shape  : {processed.shape}")
        print(f"   Dtype  : {processed.dtype}")
        print(f"   Range  : [{processed.min():.2f}, {processed.max():.2f}]")
        
        # Visualiser
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original
        original = cv2.imread(test_image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Prétraité (dénormaliser pour affichage)
        denorm = processed * preprocessor.imagenet_std + preprocessor.imagenet_mean
        denorm = np.clip(denorm, 0, 1)
        axes[1].imshow(denorm)
        axes[1].set_title('Prétraité (WB + CLAHE)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('preprocessing_test.png', dpi=150, bbox_inches='tight')
        print(f"✅ Visualisation sauvegardée : preprocessing_test.png")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
