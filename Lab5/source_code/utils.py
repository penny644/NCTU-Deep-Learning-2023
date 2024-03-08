import math
import os
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from matplotlib.ticker import MaxNLocator
from mpl_axes_aligner import align

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pred(x, cond, modules, args, device):
    x = x.to(device)
    cond = cond.to(device)
    gen_x = []
    h_sequence = [modules['encoder'](x[i]) for i in range(args.n_past + args.n_future)]
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    gen_x.append(x[0])
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
                z_t = torch.FloatTensor(args.batch_size, args.z_dim).normal_().to(device)
                frame_pred_input = torch.cat([h_t_1, z_t, action_and_position], dim=1)
                g_t = modules['frame_predictor'](frame_pred_input)
                x_t_hat = modules['decoder']([g_t, skip])
            gen_x.append(x_t_hat)
    gen_x = torch.stack(gen_x)
    return gen_x

def plot_pred(x, cond, modules, epoch, args, device):
    gen_x = pred(x, cond, modules, args, device)
    os.makedirs("{}/gen/pred_{}".format(args.log_dir, epoch), exist_ok=True)

    gif = []
    prediction = []
    ground_truth = []

    prediction_x = gen_x[:, 0, :, :, :].detach()
    ground_truth_x = x[:, 0, :, :, :].detach()

    for i in range(len(gen_x)):
        filename = "{}/gen/pred_{}/frame{}.png".format(args.log_dir, epoch, i)
        save_image(prediction_x[i], filename)
        gif.append(imageio.imread(filename))
        prediction.append(prediction_x[i])
        ground_truth.append(ground_truth_x[i])
    
    prediction_combine = make_grid(prediction, nrow = 12)
    filename = "{}/gen/pred_{}/prediction_combine.png".format(args.log_dir, epoch)
    save_image(prediction_combine, filename)
    ground_truth_combine = make_grid(ground_truth, nrow = 12)
    filename = "{}/gen/pred_{}/ground_truth_combine.png".format(args.log_dir, epoch)
    save_image(ground_truth_combine, filename)

    imageio.mimsave("{}/gen/pred_{}/result.gif".format(args.log_dir, epoch), gif)

def plot_loss(kld, mse, total_loss, tfr, beta, args):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 0.05)
    x = range(0,args.niter)
    kld_line = ax.plot(x, kld, label='KLD loss')
    total_line = ax.plot(x, total_loss, label='Total loss')
    mse_line = ax.plot(x, mse, label='MSE loss')

    ax2 = ax.twinx()
    tfr_line = ax2.plot(x, tfr, linestyle='--', label='Teacher Forcing ratio')
    beta_line = ax2.plot(x, beta, linestyle='--', label='KLD weight')
    all_line = kld_line + mse_line + total_line + tfr_line + beta_line
    labels = [l.get_label() for l in all_line]
    ax.legend(all_line, labels)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Teacher Forcing ratio/KLD weight')
    ax.set_title('Training loss/ratio curve')
    align.yaxes(ax, -0.0, ax2, 0.0, 0.05)
    filename = './{}/Training_loss_ratio_curve.jpeg'.format(args.log_dir)
    plt.savefig(filename)

def plot_psnr(psnr, args):
    plt.clf()
    x = range(0, len(psnr)*5, 5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, psnr, label='Average PSNR')
    plt.title('Learning Curve of PSNR')
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.legend()
    filename = './{}/PSNR.jpeg'.format(args.log_dir)
    plt.savefig(filename)

    


