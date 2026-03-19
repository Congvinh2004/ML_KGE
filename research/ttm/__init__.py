"""
Triple Trustiness Model (TTM) - TransE with Triple Trustiness

Based on: Zhao et al., "Embedding Learning with Triple Trustiness on Noisy Knowledge Graph", 2019

This module contains all components for TTM:
- Models: TransE_TTM
- Loss: TTM_Loss
- Strategy: TTM_NegativeSampling
- Utils: TrustinessCalculator
"""

from .models.TransE_TTM import TransE_TTM
from .models.TrustinessCalculator import TrustinessCalculator
from .loss.TTM_Loss import TTM_Loss
from .strategy.TTM_NegativeSampling import TTM_NegativeSampling

__all__ = [
    'TransE_TTM',
    'TrustinessCalculator',
    'TTM_Loss',
    'TTM_NegativeSampling'
]

__version__ = '1.0.0'

