import os
import os.path
import numpy as np                                                              
import torch


def find_classes(dir, valid_styles):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d in valid_styles]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        if target in class_to_idx.keys():
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

    return images


def make_target(train, class_to_idx):
    style_to_target = {}
    for row in train.iterrows():
        style = row[1].style
        if style in class_to_idx:
            # Convert from pandas to Tensor
            target = row[1][1:].as_matrix()
            target = np.asfarray(target, dtype = 'float32')
            target = torch.from_numpy(target)
            # Normalize (since we want this to mean probability)
            target = target - 1. # zeros should be zeros, not 1s
            normalize = target.sum()
            target = target/normalize
            # To do this for a vector, do this
            #target = torch.div(target, normalize.view(-1, 1).expand_as(target))
            style_to_target[class_to_idx[style]] = target
    return style_to_target


import torch.utils.data as data
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            try:
                return img.convert('RGB')
            except OSError:
                print('corrupt file', path)
                raise

class StyleLoader(data.Dataset):
    def __init__(self, root, targets, transform=None):
        valid_styles = targets['style'].tolist()
        self.found_classes, self.style_to_idx = find_classes(root, valid_styles)
        self.imgs = make_dataset(root, self.style_to_idx)
        self.idx_to_target = make_target(targets, self.style_to_idx)
        self.transform = transform
        print('found', len(self.found_classes), 'style images on disk out of ', len(valid_styles), ' given')
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        target_vec = self.idx_to_target[target]
        return img, target_vec, path 
        #return img, target_vec
    
    def __len__(self):
        return len(self.imgs)



