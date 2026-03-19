"""
Trustiness Calculator for TransT Model

Tính trustiness value cho triples dựa trên:
1. Entity Type Instances
2. Entity Descriptions

Based on: Zhao et al., "Embedding Learning with Triple Trustiness on Noisy Knowledge Graph", 2019
"""

import torch
import numpy as np
from collections import defaultdict
import os


class TrustinessCalculator:
    """
    Tính trustiness value cho triples trong Knowledge Graph
    
    Trustiness được tính từ:
    1. Type-based trustiness: Dựa trên entity type instances
    2. Description-based trustiness: Dựa trên entity descriptions
    """
    
    def __init__(self, dataset_path, alpha=0.5, beta=0.5):
        """
        Args:
            dataset_path: Đường dẫn đến dataset
            alpha: Weight cho type-based trustiness (default: 0.5)
            beta: Weight cho description-based trustiness (default: 0.5)
        """
        self.dataset_path = dataset_path
        self.alpha = alpha
        self.beta = beta
        
        # Trustiness scores cho mỗi triple
        self.trustiness_scores = {}
        
        # Entity types và descriptions
        self.entity_types = {}
        self.entity_descriptions = {}
        
        # Type-based và description-based trustiness
        self.type_trustiness = {}
        self.desc_trustiness = {}
        
    def load_entity_types(self):
        """
        Load entity types từ file type_constrain.txt hoặc entity2type.txt
        
        Format: entity_id -> [type1, type2, ...]
        """
        type_file = os.path.join(self.dataset_path, 'type_constrain.txt')
        
        if not os.path.exists(type_file):
            print(f"⚠️  Type file không tồn tại: {type_file}")
            print("   → Sẽ dùng type-based trustiness = 1.0 (default)")
            return {}
        
        entity_types = defaultdict(list)
        
        try:
            with open(type_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Bỏ qua dòng đầu
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_id = int(parts[0])
                        types = [int(t) for t in parts[1:]]
                        entity_types[entity_id] = types
        except Exception as e:
            print(f"⚠️  Lỗi khi load entity types: {e}")
            return {}
        
        self.entity_types = dict(entity_types)
        print(f"✅ Loaded {len(self.entity_types)} entities với types")
        return self.entity_types
    
    def load_entity_descriptions(self):
        """
        Load entity descriptions từ file entity2desc.txt hoặc entity2text.txt
        
        Format: entity_id -> description_text
        """
        desc_files = [
            os.path.join(self.dataset_path, 'entity2desc.txt'),
            os.path.join(self.dataset_path, 'entity2text.txt'),
            os.path.join(self.dataset_path, 'entity2name.txt'),
        ]
        
        entity_descriptions = {}
        
        for desc_file in desc_files:
            if os.path.exists(desc_file):
                try:
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                entity_id = int(parts[0])
                                description = parts[1]
                                entity_descriptions[entity_id] = description
                    print(f"✅ Loaded descriptions từ: {desc_file}")
                    break
                except Exception as e:
                    print(f"⚠️  Lỗi khi load descriptions: {e}")
        
        if not entity_descriptions:
            print("⚠️  Không tìm thấy entity descriptions")
            print("   → Sẽ dùng description-based trustiness = 1.0 (default)")
        
        self.entity_descriptions = entity_descriptions
        return self.entity_descriptions
    
    def calculate_type_trustiness(self, h, r, t):
        """
        Tính type-based trustiness cho triple (h, r, t)
        
        Logic:
        - Nếu entity có types phù hợp với relation → trustiness cao
        - Nếu không có type info → trustiness = 1.0 (default)
        """
        if not self.entity_types:
            return 1.0  # Default trustiness
        
        h_types = self.entity_types.get(h, [])
        t_types = self.entity_types.get(t, [])
        
        # Nếu không có type info
        if not h_types or not t_types:
            return 1.0
        
        # Tính trustiness dựa trên type consistency
        # Có thể implement logic phức tạp hơn dựa trên paper
        # Ở đây dùng simple heuristic
        
        # Nếu có types → trustiness = 1.0
        # Có thể cải thiện bằng cách check type compatibility với relation
        return 1.0
    
    def calculate_desc_trustiness(self, h, r, t):
        """
        Tính description-based trustiness cho triple (h, r, t)
        
        Logic:
        - Tính semantic similarity giữa entity descriptions
        - Nếu descriptions tương thích → trustiness cao
        """
        if not self.entity_descriptions:
            return 1.0  # Default trustiness
        
        h_desc = self.entity_descriptions.get(h, "")
        t_desc = self.entity_descriptions.get(t, "")
        
        # Nếu không có descriptions
        if not h_desc or not t_desc:
            return 1.0
        
        # Tính semantic similarity (có thể dùng word embeddings, BERT, etc.)
        # Ở đây dùng simple heuristic
        # Có thể cải thiện bằng cách dùng sentence embeddings
        
        # Simple: Nếu có descriptions → trustiness = 1.0
        # Có thể cải thiện bằng cách tính similarity
        return 1.0
    
    def calculate_trustiness(self, h, r, t):
        """
        Tính combined trustiness cho triple (h, r, t)
        
        Formula:
        trustiness(h,r,t) = α * type_trustiness + β * desc_trustiness
        """
        type_trust = self.calculate_type_trustiness(h, r, t)
        desc_trust = self.calculate_desc_trustiness(h, r, t)
        
        combined_trust = self.alpha * type_trust + self.beta * desc_trust
        
        return combined_trust
    
    def calculate_all_trustiness(self, triples):
        """
        Tính trustiness cho tất cả triples
        
        Args:
            triples: List of (h, r, t) tuples
        
        Returns:
            dict: {(h, r, t): trustiness_score}
        """
        print("📊 Calculating trustiness scores...")
        
        trustiness_dict = {}
        
        for i, (h, r, t) in enumerate(triples):
            if i % 10000 == 0:
                print(f"   Processed {i}/{len(triples)} triples...")
            
            trustiness = self.calculate_trustiness(h, r, t)
            trustiness_dict[(h, r, t)] = trustiness
        
        self.trustiness_scores = trustiness_dict
        print(f"✅ Calculated trustiness for {len(trustiness_dict)} triples")
        
        return trustiness_dict
    
    def get_trustiness(self, h, r, t):
        """Lấy trustiness score cho triple (h, r, t)"""
        return self.trustiness_scores.get((h, r, t), 1.0)  # Default = 1.0
    
    def save_trustiness(self, output_path):
        """Lưu trustiness scores vào file"""
        with open(output_path, 'w') as f:
            f.write(f"{len(self.trustiness_scores)}\n")
            for (h, r, t), trust in self.trustiness_scores.items():
                f.write(f"{h}\t{r}\t{t}\t{trust}\n")
        print(f"✅ Saved trustiness scores to: {output_path}")
    
    def load_trustiness(self, input_path):
        """Load trustiness scores từ file"""
        trustiness_dict = {}
        
        if not os.path.exists(input_path):
            print(f"⚠️  Trustiness file không tồn tại: {input_path}")
            return {}
        
        try:
            with open(input_path, 'r') as f:
                lines = f.readlines()
                total = int(lines[0].strip())
                
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        h, r, t, trust = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                        trustiness_dict[(h, r, t)] = trust
                
                print(f"✅ Loaded {len(trustiness_dict)} trustiness scores")
        except Exception as e:
            print(f"⚠️  Lỗi khi load trustiness: {e}")
            return {}
        
        self.trustiness_scores = trustiness_dict
        return trustiness_dict

