from torch.utils.data import Dataset
import os
import random
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.transforms.functional as F


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
class SemanticSegmentationDataset(Dataset):
	"""Image (semantic) segmentation dataset."""

	def __init__(self, root_dir, image_processor, crop_size, train='train'):
		"""
		Args:
			root_dir (string): Root directory of the dataset containing the images + annotations.
			image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
			train (bool): Whether to load "training" or "validation" images + annotations.
		"""
		self.root_dir = root_dir
		self.image_processor = image_processor
		self.train = train
		self.crop_size = crop_size

		if train == 'train':
			sub_path = 'train' 
		elif train == 'val':
			sub_path = "val" 
		else:
			sub_path = ''
		self.img_dir = os.path.join(self.root_dir, sub_path, 'images')
		self.ann_dir = os.path.join(self.root_dir, sub_path, 'labels')
		print(self.img_dir, self.ann_dir)

		# read images
		image_file_names = []
		for root, dirs, files in os.walk(self.img_dir):
			image_file_names.extend(files)
		self.images = sorted(image_file_names)

		# read annotations
		annotation_file_names = []
		for root, dirs, files in os.walk(self.ann_dir):
			annotation_file_names.extend(files)
		self.annotations = sorted(annotation_file_names)

		assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

	def __len__(self):
		# print(len(self.images))
		return len(self.images)

	def __getitem__(self, idx):
		image_path = os.path.join(self.img_dir, self.images[idx])
		image = Image.open(os.path.join(self.img_dir, self.images[idx]))
		segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
		# Compose -> transforms(image, label)
		if self.train != 'predict':
			# random crop
			i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
			image = F.crop(image, i, j, h, w)
			segmentation_map = F.crop(segmentation_map, i, j, h, w)
			# random vflp
			probability = 0.5
			if random.random() < probability:
				image = F.vflip(image)
				segmentation_map = F.vflip(segmentation_map)
			# random hflp
			if random.random() < probability:
				image = F.hflip(image)
				segmentation_map = F.hflip(segmentation_map)
			


		encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
		for k,v in encoded_inputs.items():
			encoded_inputs[k].squeeze_() # remove batch dimension

		return encoded_inputs, image_path