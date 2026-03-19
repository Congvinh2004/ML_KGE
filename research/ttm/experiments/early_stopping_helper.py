"""
Early Stopping Helper cho TransT Training
Hỗ trợ test định kỳ và early stopping dựa trên validation metrics
"""

import os
import json
from datetime import datetime


class EarlyStopping:
    """
    Early Stopping dựa trên validation metrics
    """
    def __init__(self, 
                 patience_percent=0.12,  # 12% tổng epochs (ví dụ: 150 epochs cho 1200 epochs)
                 min_delta=0.0001,      # Cải thiện tối thiểu để coi là "cải thiện"
                 monitor='Hits@10',    # Metric để theo dõi
                 mode='max',           # 'max' cho Hits@10, MRR (càng lớn càng tốt), 'min' cho MR (càng nhỏ càng tốt)
                 min_epochs=500):      # Tối thiểu train N epochs trước khi early stop
        self.patience_percent = patience_percent
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.min_epochs = min_epochs
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.patience_epochs = None  # Sẽ được tính khi biết total_epochs
        
        # Lưu lịch sử metrics
        self.history = []
        
    def set_total_epochs(self, total_epochs):
        """Tính patience dựa trên % tổng epochs"""
        self.patience_epochs = max(1, int(total_epochs * self.patience_percent))
        print(f"📊 Early Stopping: patience = {self.patience_epochs} epochs ({self.patience_percent*100:.1f}% of {total_epochs})")
        print(f"   Monitor: {self.monitor}, Mode: {self.mode}, Min delta: {self.min_delta}")
        
    def __call__(self, epoch, metrics):
        """
        Kiểm tra xem có nên early stop không
        Returns: (should_stop, is_best, best_score)
        """
        if self.patience_epochs is None:
            raise ValueError("Must call set_total_epochs() first!")
            
        # Lấy score từ metrics
        score = metrics.get(self.monitor)
        if score is None:
            print(f"⚠️  Warning: Metric '{self.monitor}' not found in metrics. Available: {list(metrics.keys())}")
            return False, False, self.best_score
        
        # Lưu vào history
        self.history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'score': score
        })
        
        # Kiểm tra nếu chưa đủ min_epochs
        if epoch < self.min_epochs:
            if self.best_score is None or self._is_better(score, self.best_score):
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            return False, False, self.best_score
        
        # Kiểm tra xem có cải thiện không
        if self.best_score is None:
            # Lần đầu tiên
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False, True, self.best_score
        
        is_best = False
        if self._is_better(score, self.best_score):
            # Cải thiện!
            improvement = abs(score - self.best_score)
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            is_best = True
            print(f"✅ Epoch {epoch}: {self.monitor} improved to {score:.4f} (best: {self.best_score:.4f} at epoch {self.best_epoch})")
        else:
            # Không cải thiện
            self.counter += 1
            if self.counter >= self.patience_epochs:
                self.early_stop = True
                print(f"\n🛑 Early stopping triggered!")
                print(f"   Epoch: {epoch}")
                print(f"   Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch}")
                print(f"   Current {self.monitor}: {score:.4f}")
                print(f"   No improvement for {self.counter} epochs (patience: {self.patience_epochs})")
                return True, False, self.best_score
            else:
                print(f"⏳ Epoch {epoch}: {self.monitor} = {score:.4f} (best: {self.best_score:.4f} at epoch {self.best_epoch}, no improvement for {self.counter}/{self.patience_epochs} epochs)")
        
        return False, is_best, self.best_score
    
    def _is_better(self, current, best):
        """Kiểm tra xem current có tốt hơn best không"""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:  # mode == 'min'
            return current < best - self.min_delta
    
    def get_best_epoch(self):
        """Trả về epoch tốt nhất"""
        return self.best_epoch
    
    def get_best_score(self):
        """Trả về score tốt nhất"""
        return self.best_score
    
    def save_history(self, filepath):
        """Lưu lịch sử metrics vào file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'patience_percent': self.patience_percent,
                'patience_epochs': self.patience_epochs,
                'monitor': self.monitor,
                'mode': self.mode,
                'best_epoch': self.best_epoch,
                'best_score': self.best_score,
                'history': self.history
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ History saved to: {filepath}")


def test_model_periodically(model, test_dataloader, epoch, test_interval=50, 
                            type_constrain=False, use_gpu=True, 
                            early_stopping=None, checkpoint_path=None, 
                            best_checkpoint_path=None):
    """
    Test model định kỳ và cập nhật early stopping
    
    Args:
        model: Model để test
        test_dataloader: DataLoader cho test set
        epoch: Epoch hiện tại
        test_interval: Test mỗi N epochs (mặc định: 50)
        type_constrain: Có dùng type constraint không
        use_gpu: Có dùng GPU không
        early_stopping: EarlyStopping object (optional)
        checkpoint_path: Đường dẫn để lưu checkpoint hiện tại
        best_checkpoint_path: Đường dẫn để lưu best checkpoint
        
    Returns:
        metrics: Dict chứa MRR, MR, Hits@10, Hits@3, Hits@1
        should_stop: Có nên dừng training không
        is_best: Có phải best model không
    """
    from openke.config import Tester
    
    # Kiểm tra xem có cần test không
    if epoch % test_interval != 0:
        return None, False, False
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing model at epoch {epoch}...")
    print(f"{'='*60}")
    
    # Test model
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=use_gpu)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=type_constrain)
    
    metrics = {
        'MRR': mrr,
        'MR': mr,
        'Hits@10': hit10,
        'Hits@3': hit3,
        'Hits@1': hit1
    }
    
    print(f"\n📊 Results at epoch {epoch}:")
    print(f"   MRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")
    print(f"{'='*60}\n")
    
    # Lưu checkpoint hiện tại
    if checkpoint_path and hasattr(model, 'save_checkpoint'):
        model.save_checkpoint(checkpoint_path)
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")
    
    # Kiểm tra early stopping
    should_stop = False
    is_best = False
    
    if early_stopping:
        should_stop, is_best, best_score = early_stopping(epoch, metrics)
        
        # Lưu best checkpoint nếu là best model
        if is_best and best_checkpoint_path and hasattr(model, 'save_checkpoint'):
            model.save_checkpoint(best_checkpoint_path)
            print(f"⭐ Best checkpoint saved: {os.path.basename(best_checkpoint_path)}")
            print(f"   Best {early_stopping.monitor}: {best_score:.4f} at epoch {epoch}")
    
    return metrics, should_stop, is_best
















