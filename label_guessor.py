import torch

class LabelGuessor(object):

    def __init__(self, model, T):
        self.T = T
        self.guessor = model

    @torch.no_grad()
    def __call__(self, model, ims):
        org_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        is_train = self.guessor.training
        self.guessor.train()
        all_probs = []
        for im in ims:
            im = im.cuda()
            logits = self.guessor(im)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
        qb = sum(all_probs)/len(all_probs)
        lbs_tem = torch.pow(qb, 1./self.T)
        lbs = lbs_tem / torch.sum(lbs_tem, dim=1, keepdim=True)
        self.guessor.load_state_dict(org_state)
        if is_train:
            self.guessor.train()
        else:
            self.guessor.eval()
        return lbs.detach()

