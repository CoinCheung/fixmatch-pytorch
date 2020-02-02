import torch

class LabelGuessor(object):

    def __init__(self, thresh, discard_idx=1001):
        self.discard_idx = discard_idx
        self.thresh = thresh

    @torch.no_grad()
    def __call__(self, logits):
        probs = torch.softmax(logits, dim=1)
        scores, lbs = torch.max(probs, dim=1)
        lbs[scores < self.thresh] = self.discard_idx

        return lbs.detach()

