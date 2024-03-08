import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from PIL import Image, ImageDraw
import imageio

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred
from torchvision.utils import save_image, make_grid

torch.backends.cudnn.benchmark = True
mse_criterion = nn.MSELoss()

def plot_gif(x, cond, modules, epoch, args, device):
    # posterior
    x = x.to(device)
    cond = cond.to(device)
    gen_posterior = []
    h_sequence = [modules['encoder'](x[i]) for i in range(args.n_past + args.n_future)]
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    gen_posterior.append(x[0])
    x_t_hat = x[0]
    with torch.no_grad():
        for i in range(1, args.n_past + args.n_future):
            
            if((args.last_frame_skip) | (i < args.n_past)):
                h_t_1, skip = h_sequence[i-1]

            else:
                h_t_1 = modules['encoder'](x_t_hat)[0]
            
            action_and_position = cond[i-1]

            if i < args.n_past:
                h_t = h_sequence[i][0]
                z_t, mu, logvar = modules['posterior'](h_t)
                frame_pred_input = torch.cat([h_t_1, z_t, action_and_position], dim=1)
                modules['frame_predictor'](frame_pred_input)
                x_t_hat = x[i]
            else:
                h_t = h_sequence[i][0]
                z_t, mu, logvar = modules['posterior'](h_t)
                frame_pred_input = torch.cat([h_t_1, z_t, action_and_position], dim=1)
                g_t = modules['frame_predictor'](frame_pred_input)
                x_t_hat = modules['decoder']([g_t, skip])
            gen_posterior.append(x_t_hat)
    gen_posterior = torch.stack(gen_posterior)
    gen_posterior = gen_posterior[:, 0, :, :, :].detach()

    gif = []
    all_prediction = []
    all_psnr = []
    ground_truth = []
    for i in range(3):
        pred_seq = pred(x, cond, modules, args, device)
        _, _, psnr = finn_eval_seq(x[args.n_past:args.n_past+args.n_future,:,:,:,:], pred_seq[args.n_past:args.n_past+args.n_future,:,:,:,:])
        #print(psnr[0])
        all_psnr.append(np.mean(psnr[0]))
        all_prediction.append(pred_seq[:, 0, :, :, :].detach()) 
    ground_truth_x = x[:, 0, :, :, :].detach()
    max_idx = np.asarray(all_psnr).argmax()
    for i in range(args.n_past+args.n_future):
        if i < args.n_past:
            gif_tmp = Image.new('RGB', (396, 95), 'green')
        else:
            gif_tmp = Image.new('RGB', (396, 95), 'green')
            red_tmp = Image.new('RGB', (330, 95), 'red')
            gif_tmp.paste(red_tmp, (66,0))
        save_image(ground_truth_x[i], 'tmp.png')
        tmp = Image.open('tmp.png')
        gif_tmp.paste(tmp, (1,1))
        os.remove('tmp.png')
        draw = ImageDraw.Draw(gif_tmp)
        draw.text((2,66), 'Ground\ntruth', fill=(0,0,0))

        save_image(gen_posterior[i], 'tmp.png')
        tmp = Image.open('tmp.png')
        gif_tmp.paste(tmp, (67,1))
        os.remove('tmp.png')
        draw.text((68,66), 'Approx.\nposterior', fill=(0,0,0))

        save_image(all_prediction[max_idx][i], 'tmp.png')
        tmp = Image.open('tmp.png')
        gif_tmp.paste(tmp, (133,1))
        os.remove('tmp.png')
        draw.text((134,66), 'Best\nPSNR', fill=(0,0,0))

        save_image(all_prediction[0][i], 'tmp.png')
        tmp = Image.open('tmp.png')
        gif_tmp.paste(tmp, (199,1))
        os.remove('tmp.png')
        draw.text((200,66), 'Random\nsample 1', fill=(0,0,0))

        save_image(all_prediction[1][i], 'tmp.png')
        tmp = Image.open('tmp.png')
        gif_tmp.paste(tmp, (265,1))
        os.remove('tmp.png')
        draw.text((266,66), 'Random\nsample 2', fill=(0,0,0))

        save_image(all_prediction[2][i], 'tmp.png')
        tmp = Image.open('tmp.png')
        gif_tmp.paste(tmp, (330,1))
        os.remove('tmp.png')
        draw.text((331,66), 'Random\nsample 3', fill=(0,0,0))
        gif.append(gif_tmp)
    imageio.mimsave('result.gif', gif, format='GIF', duration=500)

def plot_pred_grid(x, cond, modules, epoch, args, device):
    gen_x = pred(x, cond, modules, args, device)

    prediction = []
    ground_truth = []

    prediction_x = gen_x[:, 0, :, :, :].detach()
    ground_truth_x = x[:, 0, :, :, :].detach()

    for i in range(len(gen_x)):
        prediction.append(prediction_x[i])
        ground_truth.append(ground_truth_x[i])
    
    prediction_combine = make_grid(prediction, nrow = 12)
    filename = "prediction_combine.png"
    save_image(prediction_combine, filename)
    ground_truth_combine = make_grid(ground_truth, nrow = 12)
    filename = "ground_truth_combine.png"
    save_image(ground_truth_combine, filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=30, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='/home/penny644/DL_Lab5/logs/fp/monotonic50_finish/', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=150, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.01, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=6, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=777, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--demo', default=False, action='store_true')
    

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    demo = args.demo
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    
    saved_model = torch.load('%s/model.pth' % args.model_dir)
    optimizer = args.optimizer
    model_dir = args.model_dir
    niter = args.niter
    args = saved_model['args']
    args.optimizer = optimizer
    args.model_dir = model_dir
    start_epoch = saved_model['last_epoch']

    args.seed = 1
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print(args)

    # ------------ build the models  --------------


    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']

    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------    
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)

    test_iterator = iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    
    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------
    print(start_epoch)
    psnr_list = []
    for epoch in range(start_epoch, start_epoch + 1):
        for data_len in range(len(test_data) // args.batch_size):
            try:
                test_seq, test_cond = next(test_iterator)
                test_seq.transpose_(0, 1)
                test_cond.transpose_(0, 1)
            except StopIteration:
                test_iterator = iter(test_loader)
                test_seq, test_cond = next(test_iterator)
                test_seq.transpose_(0, 1)
                test_cond.transpose_(0, 1)
            if((data_len==0)&(demo==False)):
                plot_gif(test_seq, test_cond, modules, epoch, args, device)
                plot_pred_grid(test_seq, test_cond, modules, epoch, args, device)
                
            pred_seq = pred(test_seq, test_cond, modules, args, device)
            _, _, psnr = finn_eval_seq(test_seq[args.n_past:args.n_past+args.n_future], pred_seq[args.n_past:args.n_past+args.n_future])
            psnr_list.append(psnr)
            
        ave_psnr = np.mean(np.concatenate(psnr_list))

        print(('====================== test psnr = {:.5f} ========================\n'.format(ave_psnr)))

if __name__ == '__main__':
    main()
        
