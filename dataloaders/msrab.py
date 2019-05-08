import os
import torch
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
import scipy.io as sio


class MSRAB(data.Dataset):

	def __init__(self, max_num_samples=-1, root='dataset/MSRA-B', split="train", transform=None, return_size=False):

		self.max_num_samples = max_num_samples
		self.root = root
		self.split = split
		self.transform = transform
		self.return_size = return_size
		self.files = {}
		self.n_classes = 1

		matpath = os.path.join(self.root, self.split+'ImgSet.mat')
		matfile = sio.loadmat(matpath)[self.split+'ImgSet']
		self.files[self.split] = [matfile[i][0][0] for i in range(matfile.shape[0])]

		if not self.files[split]:
			raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

		print("Found %d %s images" % (len(self.files[split]), split))

	def __len__(self):
		if self.max_num_samples > 0:
			return min(self.max_num_samples, len(self.files[self.split]))
		return len(self.files[self.split])

	def __getitem__(self, index):

		img_name = self.files[self.split][index]
		img_path = os.path.join(self.root, 'imgs', img_name[:-4]+'.jpg')
		lbl_path = os.path.join(self.root, 'gt', img_name[:-4]+'.png')

		_img = Image.open(img_path).convert('RGB')
		_w, _h = _img.size
		_size = (_h, _w)
		_tmp = np.array(Image.open(lbl_path).convert('L')) 
		_tmp = _tmp / max(1e-6, _tmp.max())
		_target = Image.fromarray(_tmp.astype(np.uint8))
		sample = {'image': _img, 'label': _target}
		if self.transform:
			sample = self.transform(sample)
		if self.return_size:
			sample['size'] = torch.tensor(_size)
		sample['label_name'] = img_name[:-4] + '.png'
		return sample


if __name__ == '__main__':
	print(os.getcwd())
	# from dataloaders import custom_transforms as tr
	import custom_transforms as tr
	from torch.utils.data import DataLoader
	from torchvision import transforms
	import matplotlib.pyplot as plt

	composed_transforms_tr = transforms.Compose([
		tr.RandomHorizontalFlip(),
		tr.RandomScale((0.5, 0.75)),
		tr.RandomCrop((512, 1024)),
		tr.RandomRotate(5),
		tr.ToTensor()])

	msrab_train= MSRAB(split='train',transform=composed_transforms_tr)

	dataloader = DataLoader(msrab_train, batch_size=2, shuffle=True, num_workers=2)

	for ii, sample in enumerate(dataloader):
		print(ii, sample["image"].size(), sample["label"].size(), type(sample["image"]), type(sample["label"]))
		for jj in range(sample["image"].size()[0]):
			img = sample['image'].numpy()
			gt = sample['label'].numpy()
			tmp = np.array(gt[jj]*255.0).astype(np.uint8)
			tmp = np.squeeze(tmp, axis=0)
			tmp = np.expand_dims(tmp, axis=2)
			segmap = np.concatenate((tmp,tmp,tmp), axis=2)
			img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
			plt.figure()
			plt.title('display')
			plt.subplot(211)
			plt.imshow(img_tmp)
			plt.subplot(212)
			plt.imshow(segmap)

		if ii == 1:
			break
	plt.show(block=True)

