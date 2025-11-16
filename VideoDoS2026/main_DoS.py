import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
from pathlib import Path
import logging
import argparse
import time
import pynvml
from numpy import log
import numpy as np
from PIL import Image

import torch
print("Visible GPU ids  :", torch.cuda.device_count())
# from lavis.models import load_model_and_preprocess
# from lavis.models import load_model_and_preprocess
from torchvision import transforms as T
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch, imageio, numpy as np
# for video
# from decord import VideoReader, cpu
# from transformers import AutoModel
# from transformers import pipeline
# from LLaVA.llava.model.builder import load_pretrained_model
# import copy
# from LLaVA.llava.model.builder import load_pretrained_model
# from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from LLaVA.llava.conversation import conv_templates, SeparatorStyle

#######################################
import sys
sys.path.append('./')
from videollama2.videollama2 import model_init, mm_infer, mm_infer_loss
from videollama2.videollama2.utils import disable_torch_init



def norm01(x):
			x = x - x.min()
			return x / x.max()
#######################################

@torch.no_grad()
def measure_gpu(model, processor, tokenizer,inputs, iter_num, device_id):
	pynvml.nvmlInit()
	handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
	multi_cap_list = []
	t1 = time.time()
	power_list = []
	avg_length, verbose_length, verbose_caption = 0, 0, 0
	for _ in range(iter_num):

		modal = 'video'
		instruct = 'Please tell me the information of the video.'
		
		
		if isinstance(inputs, str):
			temp = processor[modal](inputs,num_frames=args.frames,aspect_ratio='pad') # temp torch.Size([16, 3, 336, 336])
			# print("temp",temp.shape)
			# temp =  norm01(temp)

			# # print('saving the clean clip')
			# video_np = (temp.clamp(0, 1)*255).cpu().detach().numpy().astype(np.uint8)
			# video_np = np.transpose(video_np, (0, 2, 3, 1))   # 8,336,336,3
			# os.makedirs(args.save_adv,exist_ok=True)
			# print("saving the adversarial video")
			# save_path = os.path.join(args.save_adv,'clean.mp4')
			# imageio.mimsave(save_path, video_np, fps=8)  # fps 1s 放N帧 fps=8， 8/8 = 1

			caption = mm_infer(temp, instruct, model=model, tokenizer=tokenizer, do_sample=True, modal=modal)
			# caption = mm_infer_loss(inputs, instruct, model=model, tokenizer=tokenizer, do_sample=True, modal=modal)
			print("【caption】   ",caption)
		else:
			# adv_vid直接用tensor形式送进即可
			caption = mm_infer(inputs, instruct, model=model, tokenizer=tokenizer, do_sample=True, modal=modal)
			
			print("【caption】   ",caption)
			
		length = len(caption.split(' '))
		multi_cap_list.append(caption)
		avg_length += (length / iter_num)
		if length > verbose_length:
			verbose_length = length
			verbose_caption = caption
		power = pynvml.nvmlDeviceGetPowerUsage(handle)
		power_list.append(power)
	t2 = time.time()
	latency = (t2 - t1) / iter_num
	s_energy = sum(power_list) / len(power_list) * latency
	energy = s_energy / (10 ** 3) / iter_num
	pynvml.nvmlShutdown()
	return latency, energy, verbose_caption, avg_length, multi_cap_list


class TestModel:
	def test_model(self, args, logger):
		print("========= In test_model: loading the model and vis_processors ===========")
		modal = 'video'
		model_path = '/data1/tdx/video_DoS/Verbose_Images/DoS_Videos/VideoLLaMA2-7B'
		model, processor, tokenizer = model_init(model_path, device_map='auto')
		model.eval()
		print("loaded the Video-LLM")

		ITER, STEP_SIZE, EPSILON = args.iter, args.step_size, args.epsilon
		input_text = 'Please tell me the information of the video.'

		###################测试干净的video#####################
		
		inputs = args.video_path # modal_path = 'ass.mp4' 
		print("============================Start describe the clean video================================")
		verbose_len, verbose_energy, verbose_latency = 0, 0, 0
		ori_latency, ori_energy, ori_caption, ori_len, ori_multi_cap_list = measure_gpu(model, processor, tokenizer,inputs,3, args.gpu)
		output_text = ori_caption

		verbose_multi_cap_list = []
		verbose_len_list, verbose_energy_list, verbose_latency_list, ori_latency_list, ori_energy_list, ori_len_list = [], [], [], [], [], []
		ori_latency_list.append(ori_latency)
		ori_energy_list.append(ori_energy)
		ori_len_list.append(ori_len)


		print("==============Start attacking the video-llm====================")
		video = processor[modal](args.video_path,num_frames=args.frames,aspect_ratio='pad') # tensor

		delta = torch.randn_like(video, requires_grad=True)
		for tdx in range(ITER):
			result = mm_infer_loss(video + delta, input_text,  model, tokenizer, modal='video')
			# image_or_video, instruct, model, tokenizer, modal='video', **kwargs
			loss1, loss2, loss3 = result["loss1"], result["loss2"], result["loss3"]
			loss1_val, loss2_val, loss3_val = loss1.detach().clone(), loss2.detach().clone(), loss3.detach().clone()

			ratio1 = 10.0 * log(tdx + 1) - 20.0
			ratio2 = 0.5 * log(tdx + 1) + 1.0

			if tdx == 0:
				lambda1 = torch.abs(loss1_val / loss2_val / ratio1)
				lambda2 = torch.abs(loss1_val / loss3_val / ratio2)
			else:
				cur_lambda1 = torch.abs(loss1_val / loss2_val / ratio1)
				cur_lambda2 = torch.abs(loss1_val / loss3_val / ratio2)                     
				lambda1 = 0.9 * last_lambda1 + 0.1 * cur_lambda1
				lambda2 = 0.9 * last_lambda2 + 0.1 * cur_lambda2

			last_lambda1, last_lambda2 = lambda1, lambda2  
			loss = loss1 + lambda1 * loss2 + lambda2 * loss3

			model.zero_grad()
			loss.backward(retain_graph=False)
			delta.data = delta - STEP_SIZE * torch.sign(delta.grad.detach())
			delta.data = self.clamp(delta, video).clamp(-EPSILON, EPSILON)
			delta.grad.zero_()

			output_latency, output_energy, output_text, output_len, output_multi_cap_list = measure_gpu(model, {"video": video + delta, "args": args, "prompt": [input_text], "logger": logger}, 3, args.gpu)

			if output_len > verbose_len:
				verbose_energy = output_energy
				verbose_latency = output_latency
				verbose_len = output_len
				verbose_multi_cap_list = output_multi_cap_list

		
		verbose_latency_list.append(verbose_latency)
		verbose_energy_list.append(verbose_energy)
		verbose_len_list.append(verbose_len)

		for len_idx in range(3):
			logger.info('Original sequences: %s', ori_multi_cap_list[len_idx])
			logger.info('Verbose sequences: %s', verbose_multi_cap_list[len_idx])
			logger.info('------------------------')

		logger.info('Original videos, Length: %.2f, Energy: %.2f, Latency: %.2f', ori_len, ori_energy, ori_latency)
		logger.info('Verbose videos, Length: %.2f, Energy: %.2f, Latency: %.2f', verbose_len, verbose_energy, verbose_latency)



def parse_args():
	'''PARAMETERS'''
	parser = argparse.ArgumentParser('generate verbose images')
	parser.add_argument('--epsilon', type=float, default=0.032, help='the perturbation magnitude')
	# 新增args
	parser.add_argument('--root_path', type=str, default='/data1/tdx/video_DoS/Verbose_Images', help='实验根目录')
	parser.add_argument('--dataset', type=str, default='activitynet', help='data目录')
	parser.add_argument('--save_adv', type=str, default='/data1/tdx/video_DoS/Verbose_Images/DoS_Videos/adv', help='save the adversarial samples')
	parser.add_argument('--video_path', type=str,default='/data1/tdx/video_DoS/Verbose_Images/DoS_Videos/videos/1F8zCQ8B4Do_000328_000338.mp4',help='video_path(mp4)')
	parser.add_argument('--delta', type=str, default='/data1/tdx/video_DoS/Verbose_Images/adv/delta_1000.pt', help='delta')
	parser.add_argument('--frames', type=float, default=16, help='the frame of video') # Llama2

	parser.add_argument('--step_size', type=float, default=0.0039, help='the step size')
	parser.add_argument('--iter', type=int, default=1000, help='the iteration')
	parser.add_argument('--gpu', type=int, default=0, help='GPU index')
	parser.add_argument('--seed', type=int, default=256, help='random seed')
	return parser.parse_args()


def main(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	def log_string(str):
		logger.info(str)
		print(str)

	'''CREATE DIR'''
	exp_dir = Path(os.path.join(args.root_path, 'log'))
	exp_dir.mkdir(exist_ok=True)
	log_dir = exp_dir.joinpath(args.dataset)
	log_dir.mkdir(exist_ok=True)

	'''LOG'''
	args = parse_args()
	logger = logging.getLogger("OPT")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler = logging.FileHandler('%s/log.txt' % log_dir)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	log_string('PARAMETER ...')
	log_string(args)

	test_blip2 = TestModel()
	test_blip2.test_model(args, logger)


if __name__ == '__main__':
	args = parse_args()
	main(args)



