import numpy as np
import torch
import torch.nn as nn

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
	n, c, h, w = logit.size()
	# logit = logit.permute(0, 2, 3, 1)
	target = target.squeeze(1)
	if weight is None:
		criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
	else:
		criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
	loss = criterion(logit, target.long())

	if size_average:
		loss /= (h * w)

	if batch_average:
		loss /= n

	return loss

def BCE_2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
	n, c, h, w = logit.size()
	if weight is None:
		criterion = nn.BCEWithLogitsLoss(weight=weight, size_average=False)
	else:
		criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), size_average=False)
	loss = criterion(logit, target)

	if size_average:
		loss /= (h * w)
	
	if batch_average:
		loss /= n
	
	return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
	return base_lr * ((1 - float(iter_) / max_iter) ** power)

def get_mae(preds, labels):
	assert(preds.numel() == labels.numel())
	if preds.dim() == labels.dim():
		mae = torch.mean(torch.abs(preds - labels))
		return mae.item()
	else:
		mae = torch.mean(torch.abs(preds.squeeze() - labels.squeeze()))
		return mae.item()

def get_prec_recall(preds, labels):
	assert(preds.numel() == labels.numel())
	preds_ = preds.cpu().data.numpy()
	labels_ = labels.cpu().data.numpy()
	preds_ = preds_.squeeze(1)
	labels_ = labels_.squeeze(1)
	assert(len(preds_.shape) == len(labels_.shape)) 
	assert(len(preds_.shape) == 3)
	prec_list = []
	recall_list = []

	assert(preds_.min() >= 0 and preds_.max() <= 1)
	assert(labels_.min() >= 0 and labels_.max() <= 1)
	for i in range(preds_.shape[0]):
		pred_, label_ = preds_[i], labels_[i]
		thres_ = pred_.sum() * 2.0 / pred_.size

		binari_ = np.zeros(shape=pred_.shape, dtype=np.uint8)
		binari_[np.where(pred_ >= thres_)] = 1

		label_ = label_.astype(np.uint8)
		matched_ = np.multiply(binari_, label_)

		TP = matched_.sum()
		TP_FP = binari_.sum()
		TP_FN = label_.sum()
		prec = (TP + 1e-6) / (TP_FP + 1e-6)
		recall = (TP + 1e-6) / (TP_FN + 1e-6)
		prec_list.append(prec) 
		recall_list.append(recall)
	return prec_list, recall_list

def decode_segmap(label_mask):
	r = label_mask.copy()
	g = label_mask.copy()
	b = label_mask.copy()
	rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
	rgb[:, :, 0] = r * 255.0
	rgb[:, :, 1] = g * 255.0
	rgb[:, :, 2] = b * 255.0
	rgb = np.rint(rgb).astype(np.uint8)
	return rgb

def decode_seg_map_sequence(label_masks):
	assert(label_masks.ndim == 3 or label_masks.ndim == 4)
	if label_masks.ndim == 4:
		label_masks = label_masks.squeeze(1)
	assert(label_masks.ndim == 3)
	rgb_masks = []
	for i in range(label_masks.shape[0]):
		rgb_mask = decode_segmap(label_masks[i])
		rgb_masks.append(rgb_mask)
	rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
	return rgb_masks
