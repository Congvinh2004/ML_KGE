"""
Baseline Module

Chứa baseline TransE model và utilities để so sánh
"""

from .baseline_config import (
    get_baseline_config,
    get_baseline_performance,
    print_baseline_info,
    BASELINE_CONFIG,
    BASELINE_PERFORMANCE
)

__all__ = [
    'get_baseline_config',
    'get_baseline_performance',
    'print_baseline_info',
    'BASELINE_CONFIG',
    'BASELINE_PERFORMANCE',
]
