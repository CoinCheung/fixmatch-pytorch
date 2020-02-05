import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WideResnet
from cifar import get_train_loader, get_val_loader, OneHot
from label_guessor import LabelGuessor
#  from loss import CrossEntropyLoss
from lr_scheduler import WarmupCosineLrScheduler
from ema import EMA


## some hyper-parameters are borrowed from the official repository
wresnet_k = 2
wresnet_n = 28
n_classes = 10
n_labeled = 40
lr = 0.03
n_epoches = 1024
batchsize = 64
mu = 7
thr = 0.95
n_imgs_per_epoch = 64 * 1024
lam_u = 1
ema_alpha = 0.999
weight_decay = 5e-4
momentum = 0.9
n_iters_per_epoch = n_imgs_per_epoch // batchsize
n_iters_all = n_iters_per_epoch * n_epoches


## settings
torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True


def set_model():
    model = WideResnet(n_classes, k=wresnet_k, n=wresnet_n) # wide resnet-28
    model.train()
    model.cuda()
    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss().cuda()
    return model, criteria_x, criteria_u


def train_one_epoch(
        model,
        criteria_x,
        criteria_u,
        optim,
        lr_schdlr,
        ema,
        dltrain_x,
        dltrain_u,
        lb_guessor,
        lambda_u,
    ):
    loss_avg, loss_x_avg, loss_u_avg, loss_u_real_avg = [], [], [], []
    n_correct_lbs = []
    st = time.time()
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    n_strong = 0
    for it in range(n_iters_per_epoch):
        ims_x_weak, ims_x_strong, lbs_x = next(dl_x)
        ims_u_weak, ims_u_strong, lbs_u_real = next(dl_u)

        ims_x_strong = ims_x_strong.cuda()
        ims_x_weak = ims_x_weak.cuda()
        lbs_x = lbs_x.cuda()
        ims_u_weak = ims_u_weak.cuda()
        ims_u_strong = ims_u_strong.cuda()

        #  n_x, n_u = ims_x_weak.size(0), ims_u_weak.size(0)
        #  imgs = torch.cat([ims_x_weak, ims_u_strong, ims_u_weak], dim=0)
        #  logits = model(imgs)
        #  logits_x, logits_u, logits_guess = logits[:n_x], logits[n_x:n_x+n_u], logits[n_x+n_u:]
        #  probs_u = torch.softmax(logits_guess.detach(), dim=1)
        #  scores_u, lbs_u = torch.max(probs_u, dim=1)
        #  lbs_u[scores_u < thr] = discard_idx
        #  lbs_u = lbs_u.detach()
        #  loss_x = criteria_x(logits_x, lbs_x)
        #  loss_u = criteria_u(logits_u, lbs_u)
        #  loss = loss_x + lambda_u * loss_u
        #  n_u = (scores_u < thr).sum()

        lbs_u, valid_u = lb_guessor(model, ims_u_weak)
        ims_u_strong = ims_u_strong[valid_u]
        n_x, n_u = ims_x_weak.size(0), ims_u_strong.size(0)
        if n_u != 0:
            ims_x_u = torch.cat([ims_x_weak, ims_u_strong], dim=0).detach()
            lbs_x_u = torch.cat([lbs_x, lbs_u], dim=0).detach()
            logits_x_u = model(ims_x_u)
            logits_x, logits_u = logits_x_u[:n_x], logits_x_u[n_x:]
            loss_x = criteria_x(logits_x, lbs_x)
            loss_u = criteria_u(logits_u, lbs_u)
            loss = loss_x + lambda_u * loss_u
            with torch.no_grad():
                lbs_u_real = lbs_u_real[valid_u].cuda()
                corr_lb = lbs_u_real == lbs_u
                loss_u_real = F.cross_entropy(logits_u, lbs_u_real)
        else:
            logits_x = model(ims_x_weak)
            loss_x = criteria_x(logits_x, lbs_x)
            loss_u = torch.tensor(0)
            loss_u_real = torch.tensor(0)
            corr_lb = torch.tensor(0)
            loss = loss_x

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_avg.append(loss.item())
        loss_x_avg.append(loss_x.item())
        loss_u_avg.append(loss_u.item())
        loss_u_real_avg.append(loss_u_real.item())
        n_correct_lbs.append(corr_lb.sum().item())
        n_strong += n_u

        if (it+1) % 512 == 0:
            ed = time.time()
            t = ed -st
            loss_avg = sum(loss_avg) / len(loss_avg)
            loss_x_avg = sum(loss_x_avg) / len(loss_x_avg)
            loss_u_avg = sum(loss_u_avg) / len(loss_u_avg)
            loss_u_real_avg = sum(loss_u_real_avg) / len(loss_u_real_avg)
            n_correct_lbs = sum(n_correct_lbs) / len(n_correct_lbs)
            lr_log = lr_schdlr.get_lr_ratio() * lr
            n_strong /= 512
            msg = ', '.join([
                'iter: {}',
                'loss: {:.4f}',
                'loss_u: {:.4f}',
                'loss_x: {:.4f}',
                'loss_u_real_lb: {:.4f}',
                'n_correct_u: {}/{}',
                'lr: {:.4f}',
                'time: {:.2f}',
            ]).format(
                it+1, loss_avg, loss_u, loss_x, loss_u_real_avg,
                int(n_correct_lbs), int(n_strong), lr_log, t
            )
            loss_avg, loss_x_avg, loss_u_avg, loss_u_real_avg = [], [], [], []
            n_correct_lbs = []
            st = ed
            n_strong = 0
            print(msg)

    ema.update_buffer()


def evaluate(ema):
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()

    dlval = get_val_loader(batch_size=128, num_workers=0, root='cifar10')
    matches = []
    for ims, lbs in dlval:
        ims = ims.cuda()
        lbs = lbs.cuda()
        with torch.no_grad():
            logits = ema.model(ims)
            scores = torch.softmax(logits, dim=1)
            _, preds = torch.max(scores, dim=1)
            match = lbs == preds
            matches.append(match)
    matches = torch.cat(matches, dim=0).float()
    acc = torch.mean(matches)
    ema.restore()
    return acc



def train():
    model, criteria_x, criteria_u = set_model()

    dltrain_x, dltrain_u = get_train_loader(
        batchsize, mu, n_iters_per_epoch, L=n_labeled)
    lb_guessor = LabelGuessor(thresh=thr)

    ema = EMA(model, ema_alpha)

    wd_params, non_wd_params = [], []
    for param in model.parameters():
        if len(param.size()) == 1:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=lr, weight_decay=weight_decay,
        momentum=momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(
        optim, max_iter=n_iters_all, warmup_iter=0
    )

    train_args = dict(
        model=model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        ema=ema,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        lb_guessor=lb_guessor,
        lambda_u=lam_u,
    )
    best_acc = -1
    print('start to train')
    for e in range(n_epoches):
        model.train()
        print('epoch: {}'.format(e))
        train_one_epoch(**train_args)
        torch.cuda.empty_cache()

        acc = evaluate(ema)
        best_acc = acc if best_acc < acc else best_acc
        log_msg = [
            'epoch: {}'.format(e),
            'acc: {:.4f}'.format(acc),
            'best_acc: {:.4f}'.format(best_acc)]
        print(', '.join(log_msg))


if __name__ == '__main__':
    train()

