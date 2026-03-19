"""
Configuration file cho các thực nghiệm

Tập trung tất cả các hyperparameters và settings ở đây để dễ quản lý
"""

import os

# ============================================
# Dataset Paths
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'benchmarks')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'research', 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'research', 'results')

# Dataset paths
FB15K237_PATH = os.path.join(DATASET_DIR, 'FB15K237')
WN18RR_PATH = os.path.join(DATASET_DIR, 'WN18RR')

# ============================================
# TransE Configuration
# ============================================
TRANSE_CONFIG = {
    # Model parameters
    'dim': 200,                    # Embedding dimension
    'p_norm': 1,                   # L1 or L2 norm (1 or 2)
    'norm_flag': True,             # Normalize embeddings
    'margin': 5.0,                 # Margin for margin loss
    'epsilon': None,               # Epsilon for initialization
    
    # Custom features (cho TransE_Custom)
    'dropout': 0.0,                # Dropout rate (0.0 = disabled)
    'use_batch_norm': False,       # Use Batch Normalization
    'init_method': 'xavier',       # Initialization method: 'xavier', 'uniform', 'normal'
    
    # Training parameters
    'train_times': 1000,           # Number of epochs
    'alpha': 1.0,                  # Learning rate
    'batch_size': None,            # Will be set automatically by dataloader
    'nbatches': 100,               # Number of batches per epoch
    'neg_ent': 5,                  # Number of negative entities per positive
    'neg_rel': 0,                  # Number of negative relations per positive
    
    # DataLoader parameters
    'threads': 4,                  # Number of threads (0 = single thread, good for Windows)
    'sampling_mode': 'normal',     # Sampling mode
    'bern_flag': 1,                # Bernoulli sampling flag
    'filter_flag': 1,              # Filter flag for negative sampling
    
    # GPU settings
    'use_gpu': True,               # Use GPU if available
    
    # Evaluation
    'type_constrain': False,       # Use type constraint in evaluation
}

# ============================================
# TransE_Custom Configuration (Example)
# ============================================
TRANSE_CUSTOM_CONFIG = {
    **TRANSE_CONFIG,  # Inherit base config
    
    # Custom modifications
    'dropout': 0.1,                # Add dropout
    'use_batch_norm': True,        # Add batch normalization
    'init_method': 'normal',       # Different initialization
}

# ============================================
# Dataset-specific Configs
# ============================================
FB15K237_CONFIG = {
    **TRANSE_CONFIG,
    'dataset_path': FB15K237_PATH,
    'dim': 200,
    'margin': 5.0,
    'alpha': 1.0,
}

WN18RR_CONFIG = {
    **TRANSE_CONFIG,
    'dataset_path': WN18RR_PATH,
    'dim': 100,
    'margin': 1.0,
    'alpha': 0.01,
    'nbatches': 50,
    'neg_ent': 1,
}

# ============================================
# Helper Functions
# ============================================
def get_config(dataset='FB15K237', model='TransE'):
    """
    Lấy config theo dataset và model
    
    Args:
        dataset: 'FB15K237' hoặc 'WN18RR'
        model: 'TransE' hoặc 'TransE_Custom'
    
    Returns:
        dict: Configuration dictionary
    """
    if dataset == 'FB15K237':
        base_config = FB15K237_CONFIG.copy()
    elif dataset == 'WN18RR':
        base_config = WN18RR_CONFIG.copy()
    else:
        base_config = TRANSE_CONFIG.copy()
        base_config['dataset_path'] = os.path.join(DATASET_DIR, dataset)
    
    if model == 'TransE_Custom':
        base_config.update({
            'dropout': 0.1,
            'use_batch_norm': True,
        })
    
    return base_config


def create_directories():
    """Tạo các thư mục cần thiết"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return CHECKPOINT_DIR, RESULTS_DIR
