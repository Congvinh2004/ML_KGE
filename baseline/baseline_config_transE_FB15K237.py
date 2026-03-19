"""
Baseline Configuration - Hyperparameters tiêu chuẩn

File này chứa các hyperparameters baseline đã được tối ưu
KHÔNG THAY ĐỔI để giữ làm tiêu chuẩn so sánh
"""

# ⭐ BASELINE HYPERPARAMETERS - Đã tối ưu trên FB15K237
# ⚠️ KHÔNG THAY ĐỔI để giữ làm tiêu chuẩn so sánh!

BASELINE_CONFIG = {
    # Model parameters
    'dim': 300,              # Embedding dimension
    'p_norm': 1,             # L1 distance (Manhattan distance)
    'norm_flag': True,       # Normalize embeddings
    'margin': 6.0,           # Margin for margin loss
    
    # Training parameters
    'train_times': 1500,     # Number of epochs
    'alpha': 0.5,            # Learning rate
    'nbatches': 50,          # Number of batches per epoch
    'neg_ent': 10,           # Negative samples per positive triple
    'neg_rel': 0,            # Negative relations (not used)
    
    # DataLoader parameters
    'threads': 4,            # Number of threads for data loading
    'sampling_mode': 'normal',  # Sampling mode
    'bern_flag': 1,          # Bernoulli sampling flag
    'filter_flag': 1,        # Filter flag for negative sampling
    
    # Evaluation
    'type_constrain': True,  # Use type constraint in evaluation
    
    # GPU
    'use_gpu': True,         # Use GPU if available
}

# Baseline performance (expected results trên FB15K237)
BASELINE_PERFORMANCE = {
    'MRR': 0.2901,
    'MR': 102.2,
    'Hits@10': 0.4895,
    'Hits@3': 0.3195,
    'Hits@1': 0.1915,
}

# Dataset
DATASET = 'FB15K237'

# So sánh với tiêu chuẩn
COMPARISON = {
    'Hits@10': {
        'baseline': 0.4895,
        'openke_paper': 0.476,
        'original_paper': 0.486,
        'improvement_over_openke': '+2.8%',
        'improvement_over_original': '+0.7%',
    }
}


def get_baseline_config():
    """Lấy baseline configuration"""
    return BASELINE_CONFIG.copy()


def get_baseline_performance():
    """Lấy baseline performance metrics"""
    return BASELINE_PERFORMANCE.copy()


def print_baseline_info():
    """In thông tin baseline"""
    print("="*60)
    print("📊 BASELINE CONFIGURATION")
    print("="*60)
    print("\nHyperparameters:")
    for key, value in BASELINE_CONFIG.items():
        print(f"   {key:20s}: {value}")
    
    print("\nExpected Performance (FB15K237):")
    for key, value in BASELINE_PERFORMANCE.items():
        print(f"   {key:20s}: {value}")
    
    print("\nComparison with Standards:")
    print(f"   Hits@10 improvement over OpenKE: {COMPARISON['Hits@10']['improvement_over_openke']}")
    print(f"   Hits@10 improvement over Original: {COMPARISON['Hits@10']['improvement_over_original']}")
    print("="*60)


if __name__ == '__main__':
    print_baseline_info()
