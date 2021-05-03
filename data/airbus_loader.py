from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import numpy as np
import torch
import PIL
import cv2
import os


class AirbusDataset(Dataset):
    def __init__(self, im_dir, label_path, input_im_size=(768, 768),
                 train_im_size=(128, 128), min_objects_per_image=1,
                 max_objects_per_image=8, normalize_images=True):

        super(Dataset, self).__init__()

        self.im_dir = im_dir
        self.input_im_size = input_im_size
        self.train_im_size = train_im_size
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.normalize_images = normalize_images
        self.set_image_size(train_im_size)

        self.labels = self.generate_labels(label_path)
        self.idx_to_im_id = self.labels['ImageId'].unique()

    def __len__(self):
        return len(self.idx_to_im_id)

    def __getitem__(self, index):

        if index > len(self.idx_to_im_id):
            print('Error index too large')

        im_id = self.idx_to_im_id[index]
        im_path = os.path.join(self.im_dir, im_id)

        with open(im_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = self.transform(image.convert('RGB'))

        #image = PIL.Image.open(im_path)
        #image = self.transform(image)

        labels = self.labels[self.labels['ImageId'] == im_id]
        boxes = list(labels['BoundingBox'])
        objects = list(labels['ObjectClass'])

        for _ in range(len(objects), self.max_objects_per_image):
            objects.append(0)
            boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))

        boxes = np.array(boxes)
        objects = torch.LongTensor(objects)

        return image, objects, boxes

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def generate_labels(self, label_path):
        labels = pd.read_csv(label_path)
        num_objects = labels.pivot_table(index=['ImageId'], aggfunc='size')
        valid_images = num_objects[(self.min_objects_per_image < num_objects) & 
                              (num_objects < self.max_objects_per_image)].keys()
        valid_labels = labels[labels['ImageId'].isin(valid_images)].reset_index(drop=True)
        valid_labels['BoundingBox'] = valid_labels['EncodedPixels'].apply(self.rle_to_bbox)
        valid_labels['ObjectClass'] = valid_labels[['ImageId', 'BoundingBox']].apply(self.label_object, axis=1)
        return valid_labels

    def rle_to_bbox(self, mask_rle):
        """Decodes run length encoded image mask"""
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int)
                          for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(np.prod(self.input_im_size), dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        img = img.reshape(self.input_im_size).T
        mask_pixels = np.where(img == 1)
        x_range = [np.min(mask_pixels[1]), np.max(mask_pixels[1])]
        y_range = [np.min(mask_pixels[0]), np.max(mask_pixels[0])]
        x0 = x_range[0] / self.input_im_size[1]
        y0 = y_range[0] / self.input_im_size[0]
        width = (x_range[1] - x_range[0]) / self.input_im_size[1]
        height = (y_range[1] - y_range[0]) / self.input_im_size[0]

        return (x0, y0, width, height)

    def label_object(self, label_data):
        im_name = label_data[0]
        bbox = label_data[1]
        im_path = os.path.join(self.im_dir, im_name)
        # We can split ships into different object class based on size or color,
        # for now, all ships are class 1
        return 1

    def show_bbxs(self, image_id, line_width=2):
        im_path = os.path.join(self.im_dir, image_id)
        im = np.array(PIL.Image.open(im_path).convert('RGB'))
        im_w, im_h = im.shape[:2]
        bbxs = self.labels[self.labels['ImageId'] == image_id]['BoundingBox']
        for bbx in bbxs:
            x0 = int(bbx[0] * im_w)
            x1 = int(x0 + bbx[2] * im_w)
            y0 = int(bbx[1] * im_h)
            y1 = int(y0 + bbx[3] * im_h)
            im[y0: y1, x0: x0 + line_width] = [255, 0, 0]
            im[y0: y1, x1 - line_width: x1] = [255, 0, 0]
            im[y0: y0 + line_width, x0: x1] = [255, 0, 0]
            im[y1 - line_width: y1, x0: x1] = [255, 0, 0]
        return PIL.Image.fromarray(im)

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
              H, W = size
              self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)

def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)

def imagenet_deprocess_batch(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de
