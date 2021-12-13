import os
import math
from paddle.io import Dataset, DataLoader, DistributedBatchSampler
from paddle.vision import transforms, datasets, image_load

class ImageNet2012Dataset(Dataset):

    def __init__(self, file_folder, mode="train", transform=None):
        """Init ImageNet2012 Dataset with dataset file path, mode(train/val), and transform"""
        super(ImageNet2012Dataset, self).__init__()
        assert mode in ["train", "val"]
        self.file_folder = file_folder
        self.transform = transform
        self.img_path_list = []
        self.label_list = []

        if mode == "train":
            self.list_file = "lit_data/train.txt"
        else:
            self.list_file = "lit_data/val.txt"

        with open(self.list_file, 'r') as infile:
            for line in infile:
                img_path = line.strip().split()[0]
                img_label = int(line.strip().split()[1])
                self.img_path_list.append(os.path.join(self.file_folder, img_path))
                self.label_list.append(img_label)
        print(f'----- Imagenet2012 image {mode} list len = {len(self.label_list)}')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = image_load(self.img_path_list[index]).convert('RGB')
        data = self.transform(data)
        label = self.label_list[index]

        return data, label



def get_train_transforms():
 

    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224),
                                     scale=(0.05, 1.0)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms_train


vals_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

def get_dataset(mode='train'):
  

    assert mode in ['train', 'val']
    
    if mode == 'train':
        dataset = ImageNet2012Dataset('lit_data/train',
                                          mode=mode,
                                          transform=get_train_transforms())
    else:
        dataset = ImageNet2012Dataset('lit_data/val',
                                          mode=mode,
                                          transform=vals_transform)
    return dataset

