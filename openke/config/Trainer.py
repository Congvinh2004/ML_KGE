# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None,
				 checkpoint_prefix = None,
				 save_model = None,
				 resume_from_checkpoint = None,
				 resume_from_latest = False):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir
		self.checkpoint_prefix = checkpoint_prefix  # Prefix cho tên checkpoint file
		self.save_model = save_model  # Model thực tế để lưu (nếu khác với self.model)
		self.resume_from_checkpoint = resume_from_checkpoint  # Đường dẫn checkpoint cụ thể
		self.resume_from_latest = resume_from_latest  # Tự động tìm checkpoint mới nhất
		self.start_epoch = 0  # Epoch bắt đầu (sẽ được cập nhật khi resume)
		self.original_train_times = train_times  # Lưu train_times ban đầu để tính resume

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	def run(self):
		if self.use_gpu:
			self.model.cuda()

		# Resume từ checkpoint nếu có
		if self.resume_from_latest or self.resume_from_checkpoint:
			checkpoint_path, metadata = self._load_checkpoint_for_resume()
			if checkpoint_path:
				print(f"✅ Resumed from checkpoint: {os.path.basename(checkpoint_path)}")
				print(f"   📊 Epoch: {metadata['epoch']}, Loss: {metadata['loss']:.4f}")
				self.start_epoch = metadata['epoch']
				# Cập nhật train_times để chỉ train phần còn lại
				# Sử dụng original_train_times nếu có, nếu không dùng train_times hiện tại
				target_epochs = getattr(self, 'original_train_times', self.train_times)
				remaining_epochs = target_epochs - self.start_epoch
				if remaining_epochs > 0:
					print(f"   ⏳ Remaining epochs: {remaining_epochs} (from {self.start_epoch} to {target_epochs})")
					self.train_times = remaining_epochs
				else:
					print(f"   ⚠️  Training already completed! (Epoch {self.start_epoch}/{target_epochs})")
					return

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			actual_epoch = self.start_epoch + epoch + 1  # Epoch thực tế (tính cả phần đã train)
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			training_range.set_description("Epoch %d | loss: %f" % (actual_epoch, res))
			
			# Lưu checkpoint định kỳ
			if self.save_steps and self.checkpoint_dir and actual_epoch % self.save_steps == 0:
				print("\n💾 Epoch %d has finished, saving checkpoint..." % (actual_epoch))
				self._save_checkpoint(actual_epoch, res)

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir

	def _save_checkpoint(self, epoch, loss):
		"""
		Lưu checkpoint với tên file và metadata
		"""
		if not self.checkpoint_dir:
			return
		
		# Tạo tên file checkpoint
		if self.checkpoint_prefix:
			checkpoint_name = f"{self.checkpoint_prefix}_epoch_{epoch:05d}.ckpt"
		else:
			checkpoint_name = f"checkpoint_epoch_{epoch:05d}.ckpt"
		
		checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
		
		# Đảm bảo thư mục tồn tại
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		
		# Lưu model
		# Nếu có save_model (model thực tế), lưu nó;
		# nếu không thì lưu self.model (wrapper như NegativeSampling).
		model_to_save = self.save_model if self.save_model is not None else self.model
		
		# Kiểm tra xem model có method save_checkpoint không
		if hasattr(model_to_save, 'save_checkpoint'):
			model_to_save.save_checkpoint(checkpoint_path)
		else:
			# Fallback: lưu state_dict trực tiếp
			torch.save(model_to_save.state_dict(), checkpoint_path)
		
		# Lưu metadata (epoch, loss, timestamp)
		metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
		metadata = {
			'epoch': epoch,
			'loss': float(loss),
			'timestamp': datetime.datetime.now().isoformat(),
			'train_times': self.train_times,
			'alpha': self.alpha,
			'opt_method': self.opt_method
		}
		
		with open(metadata_path, 'w', encoding='utf-8') as f:
			json.dump(metadata, f, indent=2, ensure_ascii=False)
		
		# In thông tin
		file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
		print(f"   ✅ Checkpoint saved: {checkpoint_name} ({file_size:.2f} MB)")
		print(f"   📄 Metadata: {os.path.basename(metadata_path)}")

	def _load_checkpoint_for_resume(self):
		"""
		Tìm và load checkpoint để resume training
		Returns: (checkpoint_path, metadata) hoặc (None, None) nếu không tìm thấy
		"""
		checkpoint_path = None
		metadata = None
		
		# Nếu có đường dẫn cụ thể
		if self.resume_from_checkpoint:
			checkpoint_path = self.resume_from_checkpoint
			if not os.path.exists(checkpoint_path):
				print(f"⚠️  Checkpoint not found: {checkpoint_path}")
				return None, None
		# Nếu tự động tìm checkpoint mới nhất
		elif self.resume_from_latest and self.checkpoint_dir and self.checkpoint_prefix:
			checkpoint_path, metadata = self._find_latest_checkpoint()
			if not checkpoint_path:
				print(f"⚠️  No checkpoint found in {self.checkpoint_dir}")
				return None, None
		
		if not checkpoint_path:
			return None, None
		
		# Load metadata
		metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
		if os.path.exists(metadata_path):
			with open(metadata_path, 'r', encoding='utf-8') as f:
				metadata = json.load(f)
		else:
			# Nếu không có metadata, thử extract epoch từ tên file
			import re
			match = re.search(r'epoch_(\d+)', checkpoint_path)
			if match:
				metadata = {'epoch': int(match.group(1)), 'loss': 0.0}
			else:
				print(f"⚠️  Cannot determine epoch from checkpoint: {checkpoint_path}")
				return None, None
		
		# Load model
		model_to_load = self.save_model if self.save_model is not None else self.model
		if hasattr(model_to_load, 'load_checkpoint'):
			model_to_load.load_checkpoint(checkpoint_path)
		else:
			# Fallback: load state_dict trực tiếp
			state_dict = torch.load(checkpoint_path, map_location='cuda' if self.use_gpu else 'cpu')
			model_to_load.load_state_dict(state_dict)
		
		return checkpoint_path, metadata

	def _find_latest_checkpoint(self):
		"""
		Tìm checkpoint mới nhất dựa trên prefix
		Returns: (checkpoint_path, metadata) hoặc (None, None)
		"""
		if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
			return None, None
		
		# Tìm tất cả checkpoint files với prefix
		checkpoint_files = []
		for filename in os.listdir(self.checkpoint_dir):
			if filename.startswith(self.checkpoint_prefix) and filename.endswith('.ckpt'):
				filepath = os.path.join(self.checkpoint_dir, filename)
				# Lấy epoch từ tên file hoặc metadata
				metadata_path = filepath.replace('.ckpt', '_metadata.json')
				epoch = 0
				if os.path.exists(metadata_path):
					try:
						with open(metadata_path, 'r', encoding='utf-8') as f:
							meta = json.load(f)
							epoch = meta.get('epoch', 0)
					except:
						pass
				else:
					# Extract từ tên file
					import re
					match = re.search(r'epoch_(\d+)', filename)
					if match:
						epoch = int(match.group(1))
				
				checkpoint_files.append((filepath, epoch))
		
		if not checkpoint_files:
			return None, None
		
		# Sắp xếp theo epoch và lấy checkpoint mới nhất
		checkpoint_files.sort(key=lambda x: x[1], reverse=True)
		latest_checkpoint_path, latest_epoch = checkpoint_files[0]
		
		# Load metadata
		metadata_path = latest_checkpoint_path.replace('.ckpt', '_metadata.json')
		metadata = {'epoch': latest_epoch, 'loss': 0.0}
		if os.path.exists(metadata_path):
			with open(metadata_path, 'r', encoding='utf-8') as f:
				metadata = json.load(f)
		
		return latest_checkpoint_path, metadata