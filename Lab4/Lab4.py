import torch.nn as nn
from torchvision import models
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataloader import create_dataset
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class BasicBlock(nn.Module):
    def __init__(self, input_channel, block_channel, strides):
        super(BasicBlock, self).__init__()

        self.input_channel = input_channel
        self.output_channel = block_channel
        self.activation = nn.ReLU(inplace = True)
        self.strides = strides

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, block_channel, 3, stride = strides, padding=1, bias=False),
            nn.BatchNorm2d(block_channel),
            self.activation,
            nn.Conv2d(block_channel, self.output_channel, 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(self.output_channel),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel, 1, stride = strides, bias=False),
            nn.BatchNorm2d(self.output_channel)
        )
    def forward(self, x):
        output = self.block(x)
        if(self.strides != 1) | (self.input_channel != self.output_channel):
            residual = self.downsample(x)
        else:
            residual = x
        output += residual
        output = self.activation(output)
        return output

class BottleNeck(nn.Module):
    def __init__(self, input_channel, block_channel, strides):
        super(BottleNeck, self).__init__()

        self.input_channel = input_channel
        self.output_channel = block_channel * 4
        self.activation = nn.ReLU(inplace = True)
        self.strides = strides

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, block_channel, 1, stride = 1, bias=False),
            nn.BatchNorm2d(block_channel),
            self.activation,
            nn.Conv2d(block_channel, block_channel, 3, stride = strides, padding=1, bias=False),
            nn.BatchNorm2d(block_channel),
            self.activation,
            nn.Conv2d(block_channel, self.output_channel, 1, stride = 1, bias=False),
            nn.BatchNorm2d(self.output_channel)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel, 1, stride = strides, bias=False),
            nn.BatchNorm2d(self.output_channel)
        )
    def forward(self, x):
        output = self.block(x)
        if(self.strides != 1) | (self.input_channel != self.output_channel):
            residual = self.downsample(x)
        else:
            residual = x
        output += residual
        output = self.activation(output)
        return output

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 2)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 2)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        output = self.first(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            BottleNeck(64, 64, 1),
            BottleNeck(256, 64, 1),
            BottleNeck(256, 64, 1)
        )

        self.layer2 = nn.Sequential(
            BottleNeck(256, 128, 2),
            BottleNeck(512, 128, 2),
            BottleNeck(512, 128, 2),
            BottleNeck(512, 128, 2),
        )

        self.layer3 = nn.Sequential(
            BottleNeck(512, 256, 2),
            BottleNeck(1024, 256, 2),
            BottleNeck(1024, 256, 2),
            BottleNeck(1024, 256, 2),
            BottleNeck(1024, 256, 2),
            BottleNeck(1024, 256, 2)
        )

        self.layer4 = nn.Sequential(
            BottleNeck(1024, 512, 2),
            BottleNeck(2048, 512, 2),
            BottleNeck(2048, 512, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        output = self.first(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output
    
class resnet(nn.Module):
    def __init__(self, model, pretrain):
        super(resnet, self).__init__()
        if model == 'resnet18':
            if pretrain:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 5)
            else:
                self.resnet = models.resnet18(weights=None)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 5)
            self.epoch = 10
        elif model == 'resnet50':
            if pretrain:
                self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 5)
            else:
                self.resnet = models.resnet50(weights=None)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 5)
            self.epoch = 5
    
    def forward(self, x):
        x = self.resnet(x)
        return x

def test(test_data, model, device):
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            prediction = torch.argmax(output, dim=1)
            
            for i in range(len(labels)):
                all_labels.append(labels[i].item())
                all_predictions.append(prediction[i].item())
        test_acc = np.sum((np.array(all_labels) == np.array(all_predictions)).astype(int)) / len(all_labels)
    
    return all_labels, all_predictions, test_acc * 100.0

def train(lr, train_data, test_data, model_name, model, device):
    model.to(device)
    best_dict = {"epoch":0, "train_loss":0.0, "train_acc":0.0, "test_acc":0.0}
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    all_train_acc = []
    all_test_acc = []

    for epoch in range(model.epoch):
        total_data = 0
        correct = 0
        total_loss = 0.0
        model.train()
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            output = model(inputs)
            prediction = torch.argmax(output, dim=1)
            loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(len(labels)):
                if(labels[i].item()==prediction[i].item()):
                    correct += 1
                total_data += 1
            total_loss += loss.item()

        train_acc = correct / total_data * 100.0
        test_labels, test_predict, test_acc = test(test_data, model, device)

        all_train_acc.append(train_acc)
        all_test_acc.append(test_acc)

        if(test_acc > best_dict["test_acc"]):
            best_dict["epoch"] = epoch
            best_dict["train_loss"] = total_loss/len(train_data)
            best_dict["train_acc"] = train_acc
            best_dict["test_acc"] = test_acc
            best_labels = test_labels
            best_predict = test_predict
            torch.save(model.state_dict(), model_name + '.pth')

        if (epoch % 1) == 0:
            print("Epoch: {:4d}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(epoch, total_loss/len(train_data), train_acc, test_acc))
    print("The highest accuracy is")
    print("Epoch: {:4d}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(best_dict["epoch"], best_dict["train_loss"], best_dict['train_acc'], best_dict['test_acc']))
    return best_labels, best_predict, all_train_acc, all_test_acc


def plot_confusion_matrix(predict, label, model, filename, normalized):
    plt.clf()
    plt.cla()
    confusion_matrix = np.zeros((5, 5))
    for i in range(len(predict)):
        confusion_matrix[label[i]][predict[i]] += 1
    if(normalized):
        for i in range(5):
            confusion_matrix[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])
    
    fig, ax = plt.subplots()
    img = ax.imshow(confusion_matrix, cmap = plt.cm.Blues)

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            if(normalized):
                if(confusion_matrix[i,j]>0.5):
                    text = ax.text(j, i, "{:.2f}".format(confusion_matrix[i, j]), ha="center", va="center", color="white")
                else:
                    text = ax.text(j, i, "{:.2f}".format(confusion_matrix[i, j]), ha="center", va="center", color="black")
            else:
                if(confusion_matrix[i,j]>2523):
                    text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="white")
                else:
                    text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar(img)
    if model == 'resnet18':
        if(normalized):
            plt.title("Normalized Confusion Matrix (ResNet18)")
        else:
            plt.title("Confusion Matrix (ResNet18)")
    else:
        if(normalized):
            plt.title("Normalized Confusion Matrix (ResNet50)")
        else:
            plt.title("Confusion Matrix (ResNet50)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(filename)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs='?', default=123)
    parser.add_argument("lr", type=float, nargs='?', default=0.001)
    parser.add_argument("batch", type=int, nargs='?', default=4)
    parser.add_argument("demo", type=bool, nargs='?', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    seed_everything(args.seed)
    train_data, test_data = create_dataset(args.batch)
    if(args.demo == False):
        print('Use ResNet18 with pretraining')
        model = resnet('resnet18', True)
        test_labels, test_predict, train_acc_pretrain, test_acc_pretrain = train(args.lr, train_data, test_data, 'resnet18_pretrained', model, device)
        plot_confusion_matrix(test_predict, test_labels, 'resnet18', 'confusion_matrix_resnet18_pretrained.png', True)
        plot_confusion_matrix(test_predict, test_labels, 'resnet18', 'confusion_matrix_resnet18_pretrained_unnormalized.png', False)
        
        print('Use ResNet18 without pretraining')
        model = resnet('resnet18', False)
        test_labels, test_predict, train_acc, test_acc = train(args.lr, train_data, test_data, 'resnet18', model, device)
        plot_confusion_matrix(test_predict, test_labels, 'resnet18', 'confusion_matrix_resnet18.png', True)
        plot_confusion_matrix(test_predict, test_labels, 'resnet18', 'confusion_matrix_resnet18_unnormalized.png', False)

        x = np.zeros(10)
        for i in range(10):
            x[i] = i + 1
        plt.clf()
        plt.cla()
        plt.plot(x, test_acc, label = 'Test(w/o pretraining)', linestyle="-", marker=".", color='blue')
        plt.plot(x,test_acc_pretrain, label = 'Test(with pretraining)', linestyle="-", color='orange')
        plt.plot(x,train_acc, label = 'Train(w/o pretraining)', linestyle="-", marker=".", color='green')
        plt.plot(x,train_acc_pretrain, label = 'Train(with pretraining)', linestyle="-", color='red')
        plt.xlabel('Accuracy(%)')
        plt.xlabel('Epochs')
        plt.grid()
        plt.legend()
        plt.title('Result Comparison(ResNet18)')
        plt.savefig('compare_figure_ResNet18.jpeg')

        print('Use ResNet50 with pretraining')
        model = resnet('resnet50', True)
        test_labels, test_predict, train_acc_pretrain, test_acc_pretrain = train(args.lr, train_data, test_data, 'resnet50_pretrained', model, device)
        plot_confusion_matrix(test_predict, test_labels, 'resnet18', 'confusion_matrix_resnet50_pretrained.png', True)
        plot_confusion_matrix(test_predict, test_labels, 'resnet18', 'confusion_matrix_resnet50_pretrained_unnormalized.png', False)
        
        print('Use ResNet50 without pretraining')
        model = resnet('resnet50', False)
        test_labels, test_predict, train_acc, test_acc = train(args.lr, train_data, test_data, 'resnet50', model, device)
        plot_confusion_matrix(test_predict, test_labels, 'resnet50', 'confusion_matrix_resnet50.png', True)
        plot_confusion_matrix(test_predict, test_labels, 'resnet50', 'confusion_matrix_resnet50_unnormalized.png', False)

        x = np.zeros(5)
        for i in range(5):
            x[i] = i + 1
        plt.clf()
        plt.cla()
        plt.plot(x, test_acc, label = 'Test(w/o pretraining)', linestyle="-", marker=".", color='blue')
        plt.plot(x,test_acc_pretrain, label = 'Test(with pretraining)', linestyle="-", color='orange')
        plt.plot(x,train_acc, label = 'Train(w/o pretraining)', linestyle="-", marker=".", color='green')
        plt.plot(x,train_acc_pretrain, label = 'Train(with pretraining)', linestyle="-", color='red')
        plt.xlabel('Accuracy(%)')
        plt.xlabel('Epochs')
        plt.grid()
        plt.legend()
        plt.title('Result Comparison(resnet50)')
        plt.savefig('compare_figure_resnet50.jpeg')
    else:
        model = resnet('resnet18', False)
        model.load_state_dict(torch.load('resnet18_pretrained.pth'))
        model.to(device)
        all_labels, all_preds, test_acc = test(test_data, model, device)
        print('model:resnet18_pretrained, test accuracy:{:.2f}%'.format(test_acc))
        
        model = resnet('resnet18', False)
        model.load_state_dict(torch.load('resnet18.pth'))
        model.to(device)
        all_labels, all_preds, test_acc = test(test_data, model, device)
        print('model:resnet18, test accuracy:{:.2f}%'.format(test_acc))

        model = resnet('resnet50', False)
        model.load_state_dict(torch.load('resnet50_pretrained.pth'))
        model.to(device)
        all_labels, all_preds, test_acc = test(test_data, model, device)
        print('model:resnet50_pretrained, test accuracy:{:.2f}%'.format(test_acc))
        
        model = resnet('resnet50', False)
        model.load_state_dict(torch.load('resnet50.pth'))
        model.to(device)
        all_labels, all_preds, test_acc = test(test_data, model, device)
        print('model:resnet50, test accuracy:{:.2f}%'.format(test_acc))

    


    