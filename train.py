import random
import time

import numpy as np

import torch
import torch.nn as nn

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
discard_idx = 1001
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
    criteria_u = nn.CrossEntropyLoss(ignore_index=discard_idx).cuda()
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
    one_hot = OneHot(n_classes)
    loss_avg, loss_x_avg, loss_u_avg = [], [], []
    st = time.time()
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters_per_epoch):
        ims_x_weak, _, lbs_x = next(dl_x)
        ims_u_weak, ims_u_strong, _ = next(dl_u)

        ims_x_weak = ims_x_weak.cuda()
        lbs_x = lbs_x.cuda()
        ims_u_weak = ims_u_weak.cuda()
        ims_u_strong = ims_u_strong.cuda()

        ## TODO: try only one forward
        logits_x = model(ims_x_weak)
        loss_x = criteria_x(logits_x, lbs_x)
        lbs_u = lb_guessor(model, ims_u_weak)
        logits_u = model(ims_u_strong)
        loss_u = criteria_u(logits_u, lbs_u)
        loss = loss_x + lambda_u * loss_u

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()
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
            lr_log = lr_schdlr.get_lr_ratio() * lr
            msg = ', '.join([
                'iter: {}',
                'loss_avg: {:.4f}',
                'loss_u: {:.4f}',
                'loss_x: {:.4f}',
                'lr: {:.4f}',
                'time: {:.2f}',
            ]).format(
                it+1, loss_avg, loss_u, loss_x, lr_log, t
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
        batch_size=128, num_workers=0, root='cifar10'
    )
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
        batchsize, mu, n_iters_per_epoch, L=250)
    lb_guessor = LabelGuessor(thresh=thr, discard_idx=discard_idx)

    ema = EMA(model, ema_alpha)

    wd_params, non_wd_params = [], []
    for param in model.parameters():
        if len(param.size()) == 1:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=lr, weight_decay=weight_decay)
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

