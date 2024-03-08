import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class iclevr_dataset(Dataset):
    def __init__(self, device, mode="train"):
        self.mode = mode
        self.device = device

        self.object = json.load(open('objects.json'))
        if mode == 'train':
            self.train = list(json.load(open('train.json')).items())
        elif mode == 'test':
            self.test = json.load(open('test.json'))
        elif mode == 'new test':
            self.test = json.load(open('new_test.json'))

        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        else:
            return len(self.test)
        
    def __getitem__(self, index):
        if self.mode == 'train':
            filename = '/home/penny644/DL_Lab7/iclevr/' + str(self.train[index][0])
            image = Image.open(filename).convert("RGB")
            image = self.transforms(image)

            onehot = np.zeros(24)
            for label in self.train[index][1]:
                onehot[self.object[label]] = 1

            return image, onehot
        else:
            onehot = np.zeros(24)
            for label in self.test[index]:
                onehot[self.object[label]] = 1
            
            return onehot
