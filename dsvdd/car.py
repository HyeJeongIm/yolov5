import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import argparse

def global_contrast_normalization(x, scale='l1'):
    mean = torch.mean(x)
    x = x - mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    else:
        x_scale = torch.sqrt(torch.mean(x ** 2))
    x = x / x_scale
    return x

class CarDataset(Dataset):
    def __init__(self, root, target_class, label, transform=None):
        self.root = root
        self.target_class = target_class
        self.label = label
        self.transform = transform
        self.filepaths = []
        self.labels = []

        classes = sorted(os.listdir(root))
        target_dir = os.path.join(root, classes[target_class])

        if os.path.isdir(target_dir):
            for filename in os.listdir(target_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    self.filepaths.append(os.path.join(target_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def calculate_min_max(data_dir, target_class):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1'))
    ])

    dataset = CarDataset(root=data_dir, target_class=target_class, label=0, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    min_val = float('inf')
    max_val = float('-inf')

    for images, _ in loader:
        min_val = min(min_val, images.min().item())
        max_val = max(max_val, images.max().item())

    return min_val, max_val
    
# def calculate_min_max_for_all_classes(data_dir):
#     num_classes = len(os.listdir(data_dir))
#     min_max_values = {}

#     for class_idx in range(num_classes):
#         min_val, max_val = calculate_min_max(data_dir, class_idx)
#         min_max_values[class_idx] = (min_val, max_val)
#         print(f"Class {class_idx}: min = {min_val}, max = {max_val}")

#     return min_max_values
    
# Function to get transformations with min-max normalization
def get_transforms(args):
    with open('/home/cal-05/hj/0726/yolov5/dsvdd/mydata/min_max/bmw1_min_max.json', 'r') as f:
        min_max = json.load(f)
    min_val, max_val = min_max[args.normal_class]

    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x)),
        transforms.Normalize([min_val] * 3, [max_val - min_val] * 3)
    ])

def get_car(args):
    data_dir='/home/cal-05/hj/0726/yolov5/dsvdd/mydata'
    # min_max_values = calculate_min_max_for_all_classes(os.path.join(data_dir,'train'))

    transform = get_transforms(args)

    train_dataset = CarDataset(
        root=os.path.join(data_dir, 'train'),
        target_class=args.normal_class,
        label=0,
        transform=transform
    )

    test_dataset_normal = CarDataset(
        root=os.path.join(data_dir, 'test'),
        target_class=args.normal_class,
        label=0,
        transform=transform
    )

    test_dataset_abnormal = CarDataset(
        root=os.path.join(data_dir, 'test'),
        target_class=args.abnormal_class,
        label=1,
        transform=transform
    )

    test_dataset = ConcatDataset([test_dataset_normal, test_dataset_abnormal])
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader_train, dataloader_test
