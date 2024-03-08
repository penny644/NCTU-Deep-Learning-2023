import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataloader import read_bci_data
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

class EEGNet(nn.Module):
    def __init__(self, activation="ReLU"):
        super(EEGNet, self).__init__()

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "ELU":
            self.activation = nn.ELU()
        
        self.firstconv = nn.Sequential(
                            nn.Conv2d(1, 16, kernel_size = (1, 51), stride = (1, 1), padding = (0, 25), bias = False),
                            nn.BatchNorm2d(16, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
                        )
        
        self.deptwiseConv = nn.Sequential(
                                nn.Conv2d(16, 32, kernel_size = (2, 1), stride = (1, 1), groups = 16, bias = False),
                                nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
                                self.activation,
                                nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0),
                                nn.Dropout(p = 0.25)
                            )
        
        self.separableConv = nn.Sequential(
                                nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1, 1), padding = (0, 7), bias = False),
                                nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
                                self.activation,
                                nn.AvgPool2d(kernel_size = (1, 8), stride = (1, 8), padding = 0),
                                nn.Dropout(p = 0.25)
                            )
        
        self.classify = nn.Linear(in_features = 736, out_features = 2, bias = True)
    
    def forward(self, x):
        tmp = self.firstconv(x)
        tmp = self.deptwiseConv(tmp)
        tmp = self.separableConv(tmp)

        tmp = tmp.view(tmp.shape[0], -1)
        output = self.classify(tmp)
        return output
    
class DeepConvNet(nn.Module):
    def __init__(self, activation="ReLu"):
        super(DeepConvNet, self).__init__()

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "ELU":
            self.activation = nn.ELU()
        
        self.firstconv = nn.Conv2d(1, 25, kernel_size = (1, 5))
        
        namelist = ['secondconv', 'thirdconv', 'fourthconv', 'fifthconv']
        channel = [25, 25, 50, 100, 200]
        kernel = [(2,1), (1,5), (1,5), (1,5)]

        for i in range(1, 5):
            setattr(self, namelist[i-1], 
                    nn.Sequential(
                        nn.Conv2d(channel[i-1], channel[i], kernel_size = kernel[i-1]),
                        nn.BatchNorm2d(channel[i], eps = 1e-05, momentum = 0.1),
                        self.activation,
                        nn.MaxPool2d(kernel_size = (1, 2)),
                        nn.Dropout(p = 0.5)
                    )
            )
        
        self.classify = nn.Linear(in_features = 8600, out_features = 2, bias = True)
    
    def forward(self, x):
        tmp = self.firstconv(x)
        tmp = self.secondconv(tmp)
        tmp = self.thirdconv(tmp)
        tmp = self.fourthconv(tmp)
        tmp = self.fifthconv(tmp)

        tmp = tmp.view(tmp.shape[0], -1)
        output = self.classify(tmp)
        return output

class square_activation(nn.Module):
    def __init__(self):
        super(square_activation, self).__init__()

    def forward(self, x):
        return torch.square(x)
    
class log_activation(nn.Module):
    def __init__(self):
        super(log_activation, self).__init__()

    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))

class ShallowConvNet(nn.Module):
    def __init__(self):
        super(ShallowConvNet, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(1, 40, kernel_size = (1, 25), stride = 1),
                        nn.Conv2d(40, 40, kernel_size = (2, 1), bias = False),
                        nn.BatchNorm2d(40),
                        square_activation(),
                        nn.AvgPool2d(kernel_size = (1, 75), stride = (1, 15)),
                        log_activation(),
                        nn.Dropout(p = 0.5)
                    )
        self.classify = nn.Linear(in_features = 1760, out_features = 2, bias = True)
    def forward(self, x):
        tmp = self.conv(x)
        tmp = tmp.view(tmp.shape[0], -1)
        output = self.classify(tmp)
        return output

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def data_batch(args, train_data, train_label, test_data, test_label, device):
    train_dataset = TensorDataset(torch.Tensor(train_data).to(device), torch.LongTensor(train_label).to(device))
    test_dataset = TensorDataset(torch.Tensor(test_data).to(device), torch.LongTensor(test_label).to(device))
    batch_train = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True)
    batch_test = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False)
    return batch_train, batch_test

def model_evaluation(model, test_data):
    model.eval()
    test_acc = 0
    total = 0
    for data, label in test_data:
        prediction = torch.argmax(model(data), dim = 1)
        for i in range(len(prediction)):
            total += 1
            if(prediction[i]==label[i]):
                test_acc += 1
    return test_acc / total * 100.0

def train(args, model, train_data, test_data):
    best_dict = {"epoch":0, "train_loss":0.0, "train_acc":0.0, "test_acc":0.0}
    train_accuracy_all = []
    test_accuracy_all = []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0.01)
    #optimizer = optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(args.train_epoch):
        train_acc = 0
        total = 0
        total_loss = 0.0
        average_loss = 0.0
        model.train()
        for data, label in train_data:
            model_output = model(data)
            prediction = torch.argmax(model_output, dim = 1)

            loss = loss_fn(model_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(len(prediction)):
                total += 1
                if(prediction[i]==label[i]):
                    train_acc += 1

            total_loss += loss.item()
        train_acc = train_acc / total * 100.0
        test_acc = model_evaluation(model, test_data)
        train_accuracy_all.append(train_acc)
        test_accuracy_all.append(test_acc)
        average_loss = total_loss / len(train_data)
        
        if(test_acc > best_dict["test_acc"]):
            best_dict["epoch"] = epoch
            best_dict["train_loss"] = average_loss
            best_dict["train_acc"] = train_acc
            best_dict["test_acc"] = test_acc

        if (epoch % 10) == 0:
            print("Epoch: {:4d}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(epoch, average_loss, train_acc, test_acc))
    print("Epoch: {:4d}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(epoch, average_loss, train_acc, test_acc))
    print("The highest accuracy is")
    print("Epoch: {:4d}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(best_dict["epoch"], best_dict["train_loss"], best_dict['train_acc'], best_dict['test_acc']))
    return train_accuracy_all, test_accuracy_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs='?', default=777)
    parser.add_argument("lr", type=float, nargs='?', default=0.0005)
    parser.add_argument("batch", type=int, nargs='?', default=64)
    parser.add_argument("train_epoch", type=int, nargs='?', default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    seed_everything(args.seed)

    train_data, train_label, test_data, test_label = read_bci_data()
    batch_train, batch_test = data_batch(args, train_data, train_label, test_data, test_label, device)

    activation_list = ["ReLU", "LeakyReLU", "ELU"]

    train_accuracy = []
    test_accuracy = []
    
    print("Use SallowConvNet")
    model = ShallowConvNet()
    model.to(device)
    train_tmp, test_tmp = train(args, model, batch_train, batch_test)
    train_accuracy.append(train_tmp)
    test_accuracy.append(test_tmp)

    plt.cla()
    plt.plot(train_accuracy[0], label ='train')
    plt.plot(test_accuracy[0], label = 'test')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Activation function comparision(SallowConvNet)")
    plt.legend()
    plt.show()
    
    train_accuracy = []
    test_accuracy = []

    for activation in activation_list:
        print("Use EEGNet and " + activation)
        model = EEGNet(activation)
        model.to(device)
        train_tmp, test_tmp = train(args, model, batch_train, batch_test)
        train_accuracy.append(train_tmp)
        test_accuracy.append(test_tmp)
    plt.cla()
    for i in range(len(activation_list)):
        plt.plot(train_accuracy[i], label = activation_list[i] + '_train')
        plt.plot(test_accuracy[i], label = activation_list[i] + '_test')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Activation function comparision(EEGNet)")
    plt.legend()
    plt.show()
    
    train_accuracy = []
    test_accuracy = []

    for activation in activation_list:
        print("Use DeepConvNet and " + activation)
        model = DeepConvNet(activation)
        model.to(device)
        train_tmp, test_tmp = train(args, model, batch_train, batch_test)
        train_accuracy.append(train_tmp)
        test_accuracy.append(test_tmp)

    plt.cla()
    for i in range(len(activation_list)):
        plt.plot(train_accuracy[i], label = activation_list[i] + '_train')
        plt.plot(test_accuracy[i], label = activation_list[i] + '_test')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Activation function comparision(DeepConvNet)")
    plt.legend()
    plt.show()