import torch

class LabelGuessor(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, model, ims):
        org_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        is_train = model.training
        with torch.no_grad():
            model.train()
            all_probs = []
            logits = model(ims)
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
            idx = scores > self.thresh
            lbs = lbs[idx]

        model.load_state_dict(org_state)
        if is_train:
            model.train()
        else:
            model.eval()
        return lbs.detach(), idx

