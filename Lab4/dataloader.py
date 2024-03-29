import pandas as pd
from torch.utils import data
import numpy as np

from PIL import Image, ImageFile
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        if self.mode == 'train':
            path = self.root + 'preprocessed_train/' + self.img_name[index] + '.jpeg'
        else:
            path = self.root + 'preprocessed_test/' + self.img_name[index] + '.jpeg'
        
        label = self.label[index]
        img = Image.open(path)
        
        transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])

        img = transformations(img)
        return img, label

def create_dataset(batch):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train_data = RetinopathyLoader(root = '/home/penny644/DL/', mode = 'train')
    test_data = RetinopathyLoader(root = '/home/penny644/DL/', mode = 'test')

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
    test_loader  = data.DataLoader(dataset=test_data, batch_size=batch, shuffle=False)
    return train_loader, test_loader