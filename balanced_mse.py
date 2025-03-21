import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import joblib
from sklearn.mixture import GaussianMixture
import time
from tqdm import tqdm

def train_gmm(train_graph):
    start = time.time()
    all_labels = train_graph.edge_label
    # print('train_loader', train_loader)
    # assert 0
    # for batch in train_loader:
    # for i, batch in enumerate(tqdm(train_loader)):
        # print("batch", batch)
        # all_labels.append(batch.edge_label)
    
    print('all_labels', all_labels)
    print('Training labels curated')
    print('Fitting GMM...')
    gmm = GaussianMixture(n_components=8, random_state=0, verbose=2).fit(
        all_labels.reshape(-1, 1).cpu().numpy())
    
    gmm_dict = {}
    gmm_dict['means'] = gmm.means_
    gmm_dict['weights'] = gmm.weights_
    gmm_dict['variances'] = gmm.covariances_
    gmm_path = 'gmm.pkl'

    joblib.dump(gmm_dict, gmm_path)

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Train gmm took {timestr}")
    return gmm_path


class GAILoss(_Loss):
    def __init__(self, init_noise_sigma, gmm, device):
        super(GAILoss, self).__init__()
        self.gmm = joblib.load(gmm)
        self.gmm = {k: torch.tensor(self.gmm[k]).to(device) for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(float(init_noise_sigma), device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss(pred, target, self.gmm, noise_var)
        return loss


def gai_loss(pred, target, gmm, noise_var):
    gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
    # print('gmm', gmm)
    # print('pred', pred.shape)
    # assert 0
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var + 0.5 * noise_var.log()
    # print('mse_term', mse_term)
    sum_var = gmm['variances'] + noise_var
    balancing_term = - 0.5 * sum_var.log() - 0.5 * (pred.view(-1, 1) - gmm['means']).pow(2) / sum_var + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()

    return loss.mean()


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(float(init_noise_sigma), device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], dtype=torch.float32, device=pred.device))
    loss = loss * (2 * noise_var).detach()

    return loss


class BNILoss(_Loss):
    def __init__(self, init_noise_sigma, bucket_centers, bucket_weights):
        super(BNILoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
        self.bucket_centers = torch.tensor(bucket_centers).cuda()
        self.bucket_weights = torch.tensor(bucket_weights).cuda()

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
        return loss


def bni_loss(pred, target, noise_var, bucket_centers, bucket_weights):
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()