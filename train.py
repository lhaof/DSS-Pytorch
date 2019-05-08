from __future__ import division
import socket
import argparse
import scipy.io as sio
from datetime import datetime
import time
import glob
import os

# PyTorch includes
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Tensorboard include
from tensorboardX import SummaryWriter

from model import DSSNet, get_params, dssloss

# Dataloaders includes
from dataloaders import msrab
from dataloaders import utils
from dataloaders import custom_transforms as trforms

import numpy as np

def eval_mae(y_pred, y):
	return torch.abs(y_pred - y.float()).mean()

def eval_pr( y_pred, y, num):
	prec, recall = torch.zeros(num), torch.zeros(num)
	thlist = torch.linspace(0, 1 - 1e-10, num)
	prec, recall, thlist = prec.cuda() , recall.cuda(), thlist.cuda()
	for i in range(num):
		y_temp = (y_pred >= thlist[i]).float()
		tp = (y_temp * y.float()).sum()
		prec[i], recall[i] = tp / (y_temp.sum() + 1e-10), tp / (y.float().sum() + 1e-10)
	return prec, recall

def get_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('-gpu'            , type=str  , default='0')

	## Model settings
	parser.add_argument('-model_name'     , type=str  , default= 'DSS')
	parser.add_argument('-criterion'      , type=str  , default= 'BCE')   # cross_entropy 
	parser.add_argument('-num_classes'    , type=int  , default= 1)
	parser.add_argument('-input_size'     , type=int  , default=512)
	parser.add_argument('-output_stride'  , type=int  , default=16)

	## Train settings
	parser.add_argument('-train_dataset'  , type=str  , default= 'MSRA-B')
	parser.add_argument('-batch_size'     , type=int  , default= 1)
	parser.add_argument('-nepochs'        , type=int  , default= 10)
	parser.add_argument('-resume_epoch'   , type=int  , default= 0)
	parser.add_argument('-load_pretrain'  , type=str  , default= 'vgg16.pth')
	
	## Optimizer settings
	parser.add_argument('-naver_grad'     , type=str  , default= 10)
	parser.add_argument('-lr'             , type=float, default=1e-8)
	parser.add_argument('-weight_decay'   , type=float, default=5e-4)
	parser.add_argument('-momentum'       , type=float, default=0.9)
	parser.add_argument('-update_lr_every', type=int  , default= 3)

	## Visualization settings
	parser.add_argument('-save_every'     , type=int  , default= 2)
	parser.add_argument('-log_every'      , type=int  , default=100)
	parser.add_argument('-load_path'      , type=str  , default= '')
	parser.add_argument('-run_id'         , type=int  , default= -1)
	parser.add_argument('-use_eval'       , type=int  , default= 1)
	parser.add_argument('-use_test'       , type=int  , default= 1)
	return parser.parse_args() 

def main(args):
	# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	torch.manual_seed(1234)
	save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
	if args.resume_epoch != 0:
		runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
		run_id = int(runs[-1].split('_')[-1]) if runs else 0
	else:
		runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
		run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

	if args.run_id >= 0:
		run_id = args.run_id

	save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
	log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
	writer = SummaryWriter(log_dir=log_dir)

	net = DSSNet()
	# load VGG16 encoder or pretrained DSS
	if args.load_pretrain is not None:
		pretrain_weights = torch.load(args.load_pretrain)
		pretrain_keys = list(pretrain_weights.keys())
		net_keys = list(net.state_dict().keys())
		for key in pretrain_keys:
			_key = key 
			if _key in net_keys:
				net.state_dict()[_key].copy_(pretrain_weights[key])
			else:
				print('missing key: ',_key)
	print('created and initialized a DSS model.')
	net.cuda()

	lr_ = args.lr
	optimizer = optim.SGD(get_params(net, args.lr),momentum=args.momentum,weight_decay=args.weight_decay)

	# optimizer = optim.Adam(get_params(net, 1e-6))

	criterion = dssloss()

	composed_transforms_tr = transforms.Compose([
		# trforms.FixedResize(size=(args.input_size, args.input_size)),
		trforms.Normalize_caffevgg(mean=(104.00698793,116.66876762,122.67891434), std=(1.0,1.0,1.0)),
		trforms.ToTensor()])
	
	composed_transforms_ts = transforms.Compose([
		# trforms.FixedResize(size=(args.input_size, args.input_size)),
		trforms.Normalize_caffevgg(mean=(104.00698793,116.66876762,122.67891434), std=(1.0,1.0,1.0)),
		trforms.ToTensor()])

	train_data = msrab.MSRAB(max_num_samples=-1, split="train", transform=composed_transforms_tr)
	val_data = msrab.MSRAB(max_num_samples=-1, split="val", transform=composed_transforms_ts)

	trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
	testloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

	num_iter_tr = len(trainloader)
	num_iter_ts = len(testloader)
	nitrs = args.resume_epoch * num_iter_tr
	nsamples = args.resume_epoch * len(train_data) 
	print('nitrs: %d num_iter_tr: %d'%(nitrs, num_iter_tr))
	print('nsamples: %d tot_num_samples: %d'%(nsamples, len(train_data)))

	aveGrad = 0
	global_step = 0
	epoch_losses = []
	recent_losses = []
	start_t = time.time()
	print('Training Network')

	best_f, cur_f = 0.0, 0.0
	lr_ = args.lr
	for epoch in range(args.resume_epoch,args.nepochs):

		### do validation
		if args.use_test == 1:
			cnt = 0
			sum_testloss = 0.0

			avg_mae = 0.0
			avg_prec, avg_recall = 0.0, 0.0

			if args.use_eval == 1:
				net.eval()
			for ii, sample_batched in enumerate(testloader):
				inputs, labels = sample_batched['image'], sample_batched['label']

				# Forward pass of the mini-batch
				inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
				inputs, labels = inputs.cuda(), labels.cuda()

				with torch.no_grad():
					outputs = net.forward(inputs)
					loss = criterion(outputs, labels)
				sum_testloss += loss.item()
				
				predictions = [torch.nn.Sigmoid()(outputs_i) for outputs_i in outputs]
				if len(predictions) >= 7: 
					predictions = (predictions[2]+predictions[3]+predictions[4]+predictions[6]) / 4.0
				else:
					predictions = predictions[0]
				predictions = (predictions-predictions.min()+1e-8) / (predictions.max()-predictions.min()+1e-8)

				avg_mae += eval_mae(predictions, labels).cpu().item()
				prec, recall = eval_pr(predictions, labels, 100)
				avg_prec, avg_recall = avg_prec + prec, avg_recall + recall

				cnt += predictions.size(0)
				
				if ii % num_iter_ts == num_iter_ts-1:
					mean_testloss = sum_testloss / num_iter_ts
					avg_mae = avg_mae / num_iter_ts
					avg_prec = avg_prec / num_iter_ts
					avg_recall = avg_recall / num_iter_ts
					f = (1+0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
					f[f != f] = 0 # delete the nan
					maxf = f.max()

					print('Validation:')
					print('epoch: %d, numImages: %d testloss: %.2f mmae: %.4f maxf: %.4f' % (
						epoch, cnt, mean_testloss, avg_mae, maxf))
					writer.add_scalar('data/validloss', mean_testloss, nsamples)
					writer.add_scalar('data/validmae', avg_mae, nsamples)
					writer.add_scalar('data/validmaxf', maxf, nsamples)

					cur_f = maxf
					if cur_f > best_f:
						save_path = os.path.join(save_dir, 'models', args.model_name + '_best' + '.pth')
						torch.save(net.state_dict(), save_path)
						print("Save model at {}\n".format(save_path))
						best_f = cur_f


		### train one epoch
		net.train()
		epoch_losses = []
		for ii, sample_batched in enumerate(trainloader):
			
			inputs, labels = sample_batched['image'], sample_batched['label']
			inputs, labels = Variable(inputs, requires_grad=True), Variable(labels) 
			global_step += inputs.data.shape[0] 
			inputs, labels = inputs.cuda(), labels.cuda()

			outputs = net.forward(inputs)
			loss = criterion(outputs, labels)
			trainloss = loss.item()
			epoch_losses.append(trainloss)
			if len(recent_losses) < args.log_every:
				recent_losses.append(trainloss)
			else:
				recent_losses[nitrs % len(recent_losses)] = trainloss

			# Backward the averaged gradient
			loss /= args.naver_grad
			loss.backward()
			aveGrad += 1
			nitrs += 1
			nsamples += args.batch_size

			# Update the weights once in p['nAveGrad'] forward passes
			if aveGrad % args.naver_grad == 0:
				optimizer.step()
				optimizer.zero_grad()
				aveGrad = 0

			if nitrs % args.log_every == 0:
				meanloss = sum(recent_losses) / len(recent_losses)
				print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs'%(
					epoch,ii,meanloss,time.time()-start_t))
				writer.add_scalar('data/trainloss',meanloss,nsamples)

			# Show 10 * 3 images results each epoch
			if (ii < 50 and ii % 10 == 0) or (ii % max(1, (num_iter_tr // 10)) == 0):
			# if ii % 10 == 0:
				tmp = inputs[:1].clone().cpu().data.numpy()
				tmp += np.array((104.00698793,116.66876762,122.67891434)).reshape(1, 3, 1, 1)
				tmp = np.ascontiguousarray(tmp[:, ::-1, :, :])
				tmp = torch.tensor(tmp).float()
				grid_image = make_grid(tmp, 3, normalize=True)
				writer.add_image('Image', grid_image, global_step)
				
				predictions = [nn.Sigmoid()(outputs_i)[:1] for outputs_i in outputs]
				final_prediction = (predictions[2]+predictions[3]+predictions[4]+predictions[6]) / 4.0
				predictions.append(final_prediction)
				predictions = torch.cat(predictions, dim=0)

				grid_image = make_grid(utils.decode_seg_map_sequence(predictions.narrow(1, 0, 1).detach().cpu().numpy()), 2, normalize=False, range=(0, 255))
				writer.add_image('Predicted label', grid_image, global_step)

				grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:1], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
				writer.add_image('Groundtruth label', grid_image, global_step)


		meanloss = sum(epoch_losses) / len(epoch_losses)
		print('epoch: %d meanloss: %.2f'%(epoch,meanloss))
		writer.add_scalar('data/epochloss', meanloss, nsamples)


		### save model
		if epoch % args.save_every == args.save_every - 1:
			save_path = os.path.join(save_dir, 'models', args.model_name + '_epoch-' + str(epoch) + '.pth')
			torch.save(net.state_dict(), save_path)
			print("Save model at {}\n".format(save_path))


		### adjust lr
		if epoch % args.update_lr_every == args.update_lr_every - 1:
			lr_ = lr_ * 0.1
			print('current learning rate: ', lr_)
			optimizer = optim.SGD(get_params(net, lr_),momentum=args.momentum,weight_decay=args.weight_decay)

if __name__ == '__main__':
	args = get_arguments()
	main(args) 