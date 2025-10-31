import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TestDataLoader
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader

if __name__ == '__main__':
	# dataloader for training - sử dụng PyTorch data loader để tránh lỗi Base.dll
	train_dataloader = PyTorchTrainDataLoader(
		in_path = "./benchmarks/FB15K237/", 
		nbatches = 100,
		threads = 0,  # Đặt = 0 để tắt multiprocessing (tránh lỗi trên Windows)
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = 5,
		neg_rel = 0)

	# dataloader for test
	test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

	# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 200, 
		p_norm = 1, 
		norm_flag = True)


	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	# Đổi use_gpu = True nếu có GPU, False nếu chỉ có CPU
	USE_GPU = True  # Đổi thành False nếu không có GPU
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 1.0, use_gpu = USE_GPU)  # Giảm xuống 100 epochs để test nhanh hơn
	trainer.run()
	transe.save_checkpoint('./checkpoint/transe.ckpt')

	# test the model
	transe.load_checkpoint('./checkpoint/transe.ckpt')
	tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = USE_GPU)
	tester.run_link_prediction(type_constrain = False)