import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, lbs):
        log_scores = self.log_softmax(logits)
        loss = - log_scores * lbs
        return loss.sum(dim=1).mean()
