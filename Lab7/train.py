from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
from dataset import iclevr_dataset
from evaluator import evaluation_model
import argparse
import warnings

warnings.filterwarnings("ignore")

class ResidualConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, residual = False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        # for resnet, change the channel of input x
        self.shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1))

        self.residual = residual

        self.input_channel = input_channel
        self.output_channel = output_channel

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)

        if(self.residual == True):
            if(self.input_channel == self.output_channel):
                return x + output
            else:
                x = self.shortcut(x)
                return x + output
        
        else:
            return output
        
class UnetDown(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UnetDown, self).__init__()

        self.conv = ResidualConvBlock(input_channel, output_channel)
        self.maxpool = nn.MaxPool2d((2, 2))


    def forward(self, x):
        x = self.conv(x)
        maxpool = self.maxpool(x)
        return x, maxpool
    
class UnetUp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UnetUp, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, 2, 2)
        self.conv = ResidualConvBlock(in_channel, out_channel)

    def forward(self, x, skip):

        x = self.up_conv(x)

        # by U-net architecture, we need to copy and crop the result of down named as skip
        x_diff = skip.shape[2] - x.shape[2]
        y_diff = skip.shape[3] - x.shape[3]

        x_start = x_diff // 2
        x_end = x.shape[2] + (x_diff // 2)
        y_start = y_diff // 2
        y_end = x.shape[3] + (y_diff // 2)
        crop_skip = skip[:, :, x_start:x_end, y_start:y_end]

        x = torch.cat([crop_skip, x], dim=1)
        return self.conv(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class time_embedding(nn.Module):
    def __init__(self, input_channel, output_channel, total_timesteps):
        super(time_embedding, self).__init__()
        self.input_channel = input_channel
        self.total_timestep = total_timesteps
        
        self.model = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            swish(),
            nn.Linear(output_channel, output_channel)
        )

    def forward(self, x):
        x = x / self.total_timestep
        x = x.view(-1, self.input_channel)
        return self.model(x)
    
class condition_embedding(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(condition_embedding, self).__init__()
        self.input_channel = input_channel
        
        self.model = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            swish(),
            nn.Linear(output_channel, output_channel)
        )

    def forward(self, x):
        x = x.view(-1, self.input_channel)
        return self.model(x)
    
class Unet(nn.Module):
    def __init__(self, input_channel, num_classes=24, total_timesteps = 300):
        super(Unet, self).__init__()
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.total_timesteps = total_timesteps

        self.down1 = UnetDown(input_channel, 64)
        self.down2 = UnetDown(64, 128)
        self.down3 = UnetDown(128, 256)
        self.down4 = UnetDown(256, 512)
        self.down5 = ResidualConvBlock(512, 1024)

        self.up1 = UnetUp(1024, 512)
        self.up2 = UnetUp(512, 256)
        self.up3 = UnetUp(256, 128)
        self.up4 = UnetUp(128, 64)
        self.out = nn.Conv2d(64, input_channel, kernel_size=(1, 1))

        self.time_embedding1 = time_embedding(1, 1024, total_timesteps)
        self.time_embedding2 = time_embedding(1, 512, total_timesteps)
        self.time_embedding3 = time_embedding(1, 256, total_timesteps)
        self.time_embedding4 = time_embedding(1, 128, total_timesteps)

        self.condition_embedding1 = condition_embedding(num_classes, 1024)
        self.condition_embedding2 = condition_embedding(num_classes, 512)
        self.condition_embedding3 = condition_embedding(num_classes, 256)
        self.condition_embedding4 = condition_embedding(num_classes, 128)

    def forward(self, x, condition, t):
        skip1, down_output1 = self.down1(x)
        skip2, down_output2 = self.down2(down_output1)
        skip3, down_output3 = self.down3(down_output2)
        skip4, down_output4 = self.down4(down_output3)
        down_output5 = self.down5(down_output4)

        condition = condition.float()

        c_embedding1 = self.condition_embedding1(condition).view(-1, 1024, 1, 1)
        t_embedding1 = self.time_embedding1(t).view(-1, 1024, 1, 1)
        up_output1 = self.up1(c_embedding1 * down_output5 + t_embedding1, skip4)

        c_embedding2 = self.condition_embedding2(condition).view(-1, 512, 1, 1)
        t_embedding2 = self.time_embedding2(t).view(-1, 512, 1, 1)
        up_output2 = self.up2(c_embedding2 * up_output1 + t_embedding2, skip3)

        c_embedding3 = self.condition_embedding3(condition).view(-1, 256, 1, 1)
        t_embedding3 = self.time_embedding3(t).view(-1, 256, 1, 1)
        up_output3 = self.up3(c_embedding3 * up_output2 + t_embedding3, skip2)

        c_embedding4 = self.condition_embedding4(condition).view(-1, 128, 1, 1)
        t_embedding4 = self.time_embedding4(t).view(-1, 128, 1, 1)
        up_output4 = self.up4(c_embedding4 * up_output3 + t_embedding4, skip1)

        output = self.out(up_output4)

        return output
    
def ddpm_schedules(beta_start = 0.0001, beta_end = 0.02, total_timesteps = 300, device = 'cuda:2'):
    beta_t = torch.linspace(beta_start, beta_end, steps = total_timesteps+1).to(device)
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_sub_alpha_bar = torch.sqrt(1 - alpha_bar_t)

    one_over_sqrt_alpha_t = 1 / torch.sqrt(alpha_t)
    one_sub_alpha_over_sqrt_one_sub_alpha_bar = (1 - alpha_t) / sqrt_one_sub_alpha_bar
    sigma_t = torch.sqrt(beta_t)

    return sqrt_alpha_bar, sqrt_one_sub_alpha_bar, one_over_sqrt_alpha_t, one_sub_alpha_over_sqrt_one_sub_alpha_bar, sigma_t

class DDPM(nn.Module):
    def __init__(self, beta_start, beta_end, total_timesteps, device):
        super(DDPM, self).__init__()
        self.model = Unet(3, 24, total_timesteps).to(device)
        self.sqrt_alpha_bar, self.sqrt_one_sub_alpha_bar, self.one_over_sqrt_alpha_t, self.one_sub_alpha_over_sqrt_one_sub_alpha_bar, self.sigma_t = ddpm_schedules(beta_start, beta_end, total_timesteps, device)

        self.device = device
        self.mse_criterion = nn.MSELoss()

        self.total_timesteps = total_timesteps
    def forward(self, x, condition):
        t = torch.randint(1, self.total_timesteps + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1) * x + self.sqrt_one_sub_alpha_bar[t].view(-1, 1, 1, 1) * noise
        predict_noise = self.model(x_t, condition, t)

        loss = self.mse_criterion(noise, predict_noise)
        return loss
    
    def sampling(self, condition, image_channel, image_H, image_W):
        x = torch.randn(condition.shape[0], image_channel, image_H, image_W).to(self.device)
        for t in range(self.total_timesteps, 0, -1):
            t_input = torch.ones(condition.shape[0], 1, 1, 1) * t
            t_input = t_input.to(self.device)
            if t>1:
                z = torch.randn_like(x).to(self.device)
            else:
                z = torch.zeros_like(x).to(self.device)
            x = self.one_over_sqrt_alpha_t[t] * (x - self.one_sub_alpha_over_sqrt_one_sub_alpha_bar[t] * self.model(x, condition, t_input)) + self.sigma_t[t] * z
        return x
    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--total_timesteps', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--total_epoch', default=400, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--save_result_path', default='./result/', type=str)
    parser.add_argument('--test_only', default=False, action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    ddpm = DDPM(0.0001, 0.02, args.total_timesteps, args.device).to(args.device)
    eval_model = evaluation_model()
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch : 1.0 - epoch / args.total_epoch * 1.0)

    dataset = iclevr_dataset(args.device, 'train')
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)
    dataset = iclevr_dataset(args.device, 'test')
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=5)
    dataset = iclevr_dataset(args.device, 'new test')
    new_test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=5)

    if args.test_only == False:
        for epoch in range(args.total_epoch):
            ddpm.train()
            tmp = tqdm(train_dataloader)
            loss_ema = None
            for x, condition in tmp:
                x = x.to(args.device)
                condition = condition.to(args.device)
                loss = ddpm(x, condition)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                tmp.set_description('loss: {:.4f}'.format(loss_ema))
            scheduler.step()

            if epoch % 5 == 0:
                ddpm.eval()
                with torch.no_grad():
                    for condition in test_dataloader:
                        condition = condition.to(args.device)
                        x_gen = ddpm.sampling(condition, 3, 64, 64)
                        x_save = x_gen.mul(0.5).add_(0.5)
                        image_name = args.save_result_path + 'test/image' + str(epoch) + '.png'
                        save_image(x_save, image_name)
                        print('test accuracy:{:.4f}'.format(eval_model.eval(x_gen, condition)))
                    
                ddpm.eval()
                with torch.no_grad():
                    for condition in new_test_dataloader:
                        condition = condition.to(args.device)
                        x_gen = ddpm.sampling(condition, 3, 64, 64)
                        x_save = x_gen.mul(0.5).add_(0.5)
                        image_name = args.save_result_path + 'new_test/image' + str(epoch) + '.png'
                        save_image(x_save, image_name)
                        print('new test accuracy:{:.4f}'.format(eval_model.eval(x_gen, condition)))
            
            if epoch % 20 == 0:
                model_name = args.save_result_path + 'model.pth'
                torch.save(ddpm.state_dict(), model_name)
    
    else:
        model_name = args.save_result_path + 'model.pth'
        ddpm.load_state_dict(torch.load(model_name))
        ddpm = ddpm.to(args.device)
        ddpm.eval()
        with torch.no_grad():
            for condition in test_dataloader:
                condition = condition.to(args.device)
                x_gen = ddpm.sampling(condition, 3, 64, 64)
                x_save = x_gen.mul(0.5).add_(0.5)
                image_name = args.save_result_path + 'test/image.png'
                save_image(x_save, image_name)
                print('test accuracy:{:.4f}'.format(eval_model.eval(x_gen, condition)))
            
        ddpm.eval()
        with torch.no_grad():
            for condition in new_test_dataloader:
                condition = condition.to(args.device)
                x_gen = ddpm.sampling(condition, 3, 64, 64)
                x_save = x_gen.mul(0.5).add_(0.5)
                image_name = args.save_result_path + 'new_test/image.png'
                save_image(x_save, image_name)
                print('new test accuracy:{:.4f}'.format(eval_model.eval(x_gen, condition)))







