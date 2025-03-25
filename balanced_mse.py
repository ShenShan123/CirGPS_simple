import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import joblib
from sklearn.mixture import GaussianMixture
import time
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np

def kl_divergence(p: torch.Tensor, q: torch.Tensor, bins: int = 10, 
                  epsilon: float = 1e-8, save_path: str = "logs/histograms.png") -> float:
    """
    Compute KL divergence and plot histograms of the distributions.
    
    Args:
        p (torch.Tensor): True distribution [N, 1]
        q (torch.Tensor): Pred distribution [N, 1]
        bins (int): Number of histogram bins
        epsilon (float): Smoothing factor
        save_path (str): Path to save histogram plot
        
    Returns:
        float: KL divergence value
    """
    # Convert to 1D numpy arrays for plotting
    p_np = p.flatten().cpu().numpy()
    q_np = q.flatten().cpu().numpy()
    
    # Create figure with subplots
    plt.figure(figsize=(12, 6))
    
    # Plot first distribution
    plt.subplot(1, 2, 1)
    plt.hist(p_np, bins=bins, alpha=0.5, density=True, label='Distribution P')
    plt.title("Distribution P")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    
    # Plot second distribution
    plt.subplot(1, 2, 2)
    plt.hist(q_np, bins=bins, alpha=0.5, density=True, label='Distribution Q', color='orange')
    plt.title("Distribution Q")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    
    # Save and close plot
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    # Compute KL divergence (original calculation)
    min_val = min(p.min(), q.min()).item()
    max_val = max(p.max(), q.max()).item()
    
    p_counts = torch.histc(p, bins=bins, min=min_val, max=max_val) + epsilon
    q_counts = torch.histc(q, bins=bins, min=min_val, max=max_val) + epsilon
    
    p_probs = p_counts / p_counts.sum()
    q_probs = q_counts / q_counts.sum()
    
    kl = (p_probs * (torch.log(p_probs) - torch.log(q_probs))).sum().item()
    
    return kl

def train_gmm(dataset):
    start = time.time()
    graph_idx = 0
    train_labels = dataset[graph_idx].edge_label

    for i in range(graph_idx+1, len(dataset.names)):
        test_labels = dataset[i].edge_label
        # Compute KL divergence and save histograms
        kl_value = kl_divergence(
            test_labels, train_labels, bins=20, 
            save_path=f"logs/{dataset.names[i]}_distribution_comparison.png"
        )
        print(f"KL Divergence for {dataset.names[i]}: {kl_value:.4f}")

    print('Training labels curated')
    print('Fitting GMM...')
    gmm = GaussianMixture(n_components=8, random_state=0, verbose=2).fit(
        train_labels.reshape(-1, 1).cpu().numpy())
    
    gmm_dict = {}
    gmm_dict['means'] = gmm.means_
    gmm_dict['weights'] = gmm.weights_
    gmm_dict['variances'] = gmm.covariances_
    gmm_path = 'pkl/gmm/gmm.pkl'

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