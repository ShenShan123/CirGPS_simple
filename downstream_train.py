import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)

# from torch.utils.data.sampler import SubsetRandomSampler
# from sram_dataset import LinkPredictionDataset
# from sram_dataset import collate_fn, adaption_for_sgrl
# from torch_geometric.data import Batch

import time
from tqdm import tqdm
# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler
from model import GraphHead
from sampling_and_pe import dataset_sampling_and_pe_calculation

class Logger (object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification'):
        super().__init__()
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task

    def _get_pred_int(self, pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > 0.5).astype(int)
        else:
            return pred_score.max(dim=1)[1]

    def update_stats(self, true, pred, batch_size, loss):
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._iter += 1

    def write_epoch(self, split=""):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        true = true.numpy()
        pred_score = pred_score.numpy()
        reformat = lambda x: round(float(x), 4)

        if self.task == 'classification':
            pred_int = self._get_pred_int(pred_score)

            try:
                r_a_score = roc_auc_score(true, pred_score)
            except ValueError:
                r_a_score = 0.0

            # performance metrics to be printed
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'accuracy': reformat(accuracy_score(true, pred_int)),
                'precision': reformat(precision_score(true, pred_int)),
                'recall': reformat(recall_score(true, pred_int)),
                'f1': reformat(f1_score(true, pred_int)),
                'auc': reformat(r_a_score),
            }
        else:
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mean_squared_error(true, pred_score)),
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
            }

        # Just print the results to screen
        print(split, res)
        return res

def compute_loss(pred, true, task):
    """Compute loss and prediction score. 
    This version only supports binary classification.
    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        task (str): The task type, 'classification' or 'regression'
    Returns: Loss, normalized prediction score
    """

    ## default manipulation for pred and true
    ## can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if task == 'classification':
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        ## multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        ## binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
        
    elif task == 'regression':
        mse_loss = torch.nn.MSELoss(reduction='mean')
        return mse_loss(pred, true), pred
    
    else:
        raise ValueError(f"Task type {task} not supported!")

@torch.no_grad()
def eval_epoch(loader, batched_dspd, model, device, 
               split='val', task='classification'):
    """ 
    evaluate the model on the validation or test set
    Args:
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
        task (str): The edge-level task type, 'classification' or 'regression'
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        ## copy dspd tensor to the batch
        batch.dspd = batched_dspd[i]
        pred, true = model(batch.to(device))
        loss, pred_score = compute_loss(pred, true, task)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.squeeze().size(0),
                            loss=loss.detach().cpu().item(),
                            )
    logger.write_epoch(split)

def train(args, model, optimizier, 
          train_loader, val_loader, test_loaders, 
          train_batched_dspd, val_batched_dspd, 
          test_batched_dspd_dict, device):
    """
    Train the head model for link prediction task
    Args:
        args (argparse.Namespace): The arguments
        head_model (torch.nn.Module): The head model
        optimizier (torch.optim.Optimizer): The optimizer
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        train_batched_dspd (list): The list of batched DSPD tensors for training
        val_batched_dspd (list): The list of batched DSPD tensors for validation
        test_batched_dspd_dict (dict): The dictionary of batched DSPD tensors for test datasets
        device (torch.device): The device to train the model on
    """
    optimizier.zero_grad()
    
    for epoch in range(args.epochs):
        logger = Logger(task=args.task)
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizier.zero_grad()
            ## copy dspd tensor to the data batch
            batch.dspd = train_batched_dspd[i]
            
            ## Get the prediction from the model
            y_pred, y = model(batch.to(device))
            # print("y_pred", y_pred.squeeze()[:10])
            # print("y_true", y.squeeze()[:10])
            loss, pred = compute_loss(y_pred, y, args.task)
            _true = y.detach().to('cpu', non_blocking=True)
            _pred = y_pred.detach().to('cpu', non_blocking=True)

            loss.backward()
            optimizier.step()
            
            ## Update the logger and print message to the screen
            logger.update_stats(
                true=_true, pred=_pred, 
                batch_size=_true.squeeze().size(0), 
                loss=loss.detach().cpu().item()
            )

        logger.write_epoch(split='train')
        ## ========== validation ========== ##
        eval_epoch(
            val_loader, val_batched_dspd, 
            model, device, split='val', task=args.task
        )

        ## ========== testing on other datasets ========== ##
        for test_name in test_batched_dspd_dict.keys():
            eval_epoch(
                test_loaders[test_name], test_batched_dspd_dict[test_name], 
                model, device, split='test', task=args.task
            )
NET = 0
DEV = 1
PIN = 2

def downstream_link_pred(args, dataset, device):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        batch_index (torch.tensor): The batch index for all_node_embeds
        all_node_embeds (torch.tensor): The node embeddings come from the contrastive learning
        device (torch.device): The device to train the model on
    """
    if args.task == 'regression':
        ## normalize the circuit statistics
        dataset.norm_nfeat([NET, DEV])
    ## Subgraph sampling for each dataset graph & PE calculation
    (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    ) = dataset_sampling_and_pe_calculation(args, dataset)
    
    model = GraphHead(
        args.hid_dim, 1, num_layers=args.num_gnn_layers, 
        num_head_layers=args.num_head_layers, 
        use_bn=args.use_bn, drop_out=args.dropout, activation=args.act_fn, 
        src_dst_agg=args.src_dst_agg, use_pe=args.use_pe, max_dist=args.max_dist,
        task=args.task, use_stats=args.use_stats, 
    )

    model = model.to(device)
    
    optimizier = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    start = time.time()

    ## Start training, go go go!
    train(args, model, optimizier, 
          train_loader, val_loader, test_loaders, 
          train_dspd_list, 
          valid_dspd_list, 
          test_dspd_dict,
          device)
    
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")