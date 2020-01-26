import random
import time

import numpy as np

import torch
import torch.nn as nn

from model import WideResnet
from cifar import get_train_loader, get_val_loader, OneHot
from label_guessor import LabelGuessor
from mixup import MixUp
from loss import CrossEntropyLoss
from ema import EMA


## some hyper-parameters are borrowed from the official repository
wresnet_k = 2
wresnet_n = 28
n_classes = 10
n_workers = 0
lr = 0.002
n_epoches = 1024
batchsize = 64
n_imgs_per_epoch = 64 * 1024
n_guesses = 2
temperature = 0.5
mixup_alpha = 0.75
lam_u = 75
ema_alpha = 0.999
weight_decay = 0.02


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
    criteria_x = CrossEntropyLoss().cuda()
    criteria_u = nn.MSELoss().cuda()
    return model, criteria_x, criteria_u


def train_one_epoch(
        model,
        criteria_x,
        criteria_u,
        optim,
        ema,
        wd,
        dltrain_x,
        dltrain_u,
        lb_guessor,
        mixuper,
        lambda_u,
        lambda_u_once,
    ):
    n_iters_per_epoch = n_imgs_per_epoch // batchsize
    one_hot = OneHot(n_classes)
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    loss_avg, loss_x_avg, loss_u_avg = [], [], []
    st = time.time()
    for it in range(n_iters_per_epoch):
        try:
            ims_x, lbs_x = next(dl_x)
        except StopIteration:
            dl_x = iter(dltrain_x)
            ims_x, lbs_x = next(dl_x)
        try:
            ims_u, _ = next(dl_u)
        except StopIteration:
            dl_u = iter(dltrain_u)
            ims_u, _ = next(dl_u)
        with torch.no_grad():
            ims_x, lbs_x = ims_x[0].cuda(), one_hot(lbs_x).cuda()
            ims_u = [im.cuda() for im in ims_u]
            lbs_u = lb_guessor(model, ims_u).cuda()
            ims = torch.cat([ims_x]+ims_u, dim=0)
            lbs = torch.cat([lbs_x]+[lbs_u for _ in range(n_guesses)], dim=0)
            ims, lbs = mixuper(ims, lbs)

        optim.zero_grad()
        logits = model(ims)
        logits_x = logits[:batchsize]
        lbs_x = lbs[:batchsize]
        logits_u = logits[batchsize:]
        preds_u = torch.softmax(logits_u, dim=1)
        lbs_u = lbs[batchsize:]
        loss_x = criteria_x(logits_x, lbs_x)
        loss_u = criteria_u(preds_u, lbs_u)
        lam_u = lambda_u + lambda_u_once * it
        loss = loss_x + lam_u * loss_u
        loss.backward()
        optim.step()
        do_weight_decay(model, wd)
        ema.update_params()

        loss_avg.append(loss.item())
        loss_x_avg.append(loss_x.item())
        loss_u_avg.append(loss_u.item())

        if (it+1) % 512 == 0:
            ed = time.time()
            t = ed -st
            loss_avg = sum(loss_avg) / len(loss_avg)
            loss_x_avg = sum(loss_x_avg) / len(loss_x_avg)
            loss_u_avg = sum(loss_u_avg) / len(loss_u_avg)
            msg = ', '.join([
                'iter: {}',
                'loss_avg: {:.4f}',
                'loss_u: {:.4f}',
                'loss_x: {:.4f}',
                'lam_u: {:.4f}',
                'time: {:.2f}',
            ]).format(
                it+1, loss_avg, loss_u, loss_x, lam_u, t
            )
            loss_avg, loss_x_avg, loss_u_avg = [], [], []
            st = ed
            print(msg)

    ema.update_buffer()


def evaluate(ema):
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()

    dlval = get_val_loader(
        batch_size=128, num_workers=n_workers, root='cifar10'
    )
    matches = []
    for ims, lbs in dlval:
        ims = ims[0].cuda()
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


@torch.no_grad()
def do_weight_decay(model, decay):
    for param in model.parameters():
        param.copy_(param * decay)


def train():
    model, criteria_x, criteria_u = set_model()

    dltrain_x, dltrain_u = get_train_loader(
        batchsize, L=250, K=n_guesses, num_workers=n_workers
    )
    lb_guessor = LabelGuessor(model, T=temperature)
    mixuper = MixUp(mixup_alpha)

    ema = EMA(model, ema_alpha)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    n_iters_per_epoch = n_imgs_per_epoch // batchsize
    lam_u_epoch = float(lam_u) / n_epoches
    lam_u_once = lam_u_epoch / n_iters_per_epoch

    train_args = dict(
        model=model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        ema=ema,
        wd = 1 - weight_decay * lr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        lb_guessor=lb_guessor,
        mixuper=mixuper,
        lambda_u=0,
        lambda_u_once=lam_u_once,
    )
    best_acc = -1
    print('start to train')
    for e in range(n_epoches):
        model.train()
        print('epoch: {}'.format(e))
        train_args['lambda_u'] = e * lam_u_epoch
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

