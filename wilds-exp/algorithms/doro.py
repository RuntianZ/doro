import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

import math
import scipy.optimize as sopt
import torch.nn.functional as F

class DORO(SingleModelAlgorithm):
    """
    DORO algorithm
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        # initialize model
        model = initialize_model(config, d_out).to(config.device)
        if config.data_parallel:
            assert config.device == 'cuda'
            model = torch.nn.DataParallel(model)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.alpha = config.alpha
        self.eps = config.eps
        self.alg = config.doro_alg

    def objective(self, results):
        """
        Takes an output of SingleModelAlgorithm.process_batch() and computes the
        optimized objective. For group DRO, the objective is the weighted average
        of losses, where groups have weights groupDRO.group_weights.
        Args:
            - results (dictionary): output of SingleModelAlgorithm.process_batch()
        Output:
            - objective (Tensor): optimized objective; size (1,).
        """

        loss = self.loss.compute_element_wise(
            results['y_pred'],
            results['y_true'],
            return_dict=False
        )
        batch_size = len(loss)
        if self.alg == 'erm':
            # ERM
            loss = loss.mean()
        elif self.alg == 'cvar':
            # CVaR
            n = int(self.alpha * batch_size)
            rk = torch.argsort(loss, descending=True)
            loss = loss[rk[:n]].mean()
        elif self.alg == 'cvar_doro':
            # CVaR-DORO
            gamma = self.eps + self.alpha * (1 - self.eps)
            n1 = int(gamma * batch_size)
            n2 = int(self.eps * batch_size)
            rk = torch.argsort(loss, descending=True)
            loss = loss[rk[n2:n1]].sum() / self.alpha / (batch_size - n2)
        elif self.alg == 'chisq':
            # Chi^2
            max_l = 10.
            C = math.sqrt(1 + (1 / self.alpha - 1) ** 2)
            foo = lambda eta: C * math.sqrt((F.relu(loss - eta) ** 2).mean().item()) + eta
            opt_eta = sopt.brent(foo, brack=(0, max_l))
            loss = C * torch.sqrt((F.relu(loss - opt_eta) ** 2).mean()) + opt_eta
        elif self.alg == 'chisq_doro':
            # Chi^2-DORO
            max_l = 10.
            C = math.sqrt(1 + (1 / self.alpha - 1) ** 2)
            n = int(self.eps * batch_size)
            rk = torch.argsort(loss, descending=True)
            l0 = loss[rk[n:]]
            foo = lambda eta: C * math.sqrt((F.relu(l0 - eta) ** 2).mean().item()) + eta
            opt_eta = sopt.brent(foo, brack=(0, max_l))
            loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
        else:
            raise NotImplementedError
        return loss
