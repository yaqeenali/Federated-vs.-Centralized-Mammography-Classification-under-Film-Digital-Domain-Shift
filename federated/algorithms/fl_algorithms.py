"""
Federated Learning algorithms: FedAvg, FedProx, SCAFFOLD, FedBN.

All four algorithms evaluated in the paper (Section 2.4):
    - FedAvg:   sample-size-weighted averaging of local updates
    - FedProx:  proximal regularizer (μ=0.01) to reduce client drift
    - SCAFFOLD: control-variate drift correction; local SGD with zero momentum
    - FedBN:    aggregate non-BN params only; BN stats stay client-specific

Training config (Section 2.6):
    - SGD, momentum=0.9 (0.0 for SCAFFOLD), weight decay=1e-4
    - lr=0.01, batch_size=32
    - E=3 local epochs per round, 100 FL rounds

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
    FedAvg: McMahan et al., AISTATS 2017
    FedProx: Li et al., MLSys 2020
    SCAFFOLD: Karimireddy et al., ICML 2020
    FedBN: Li et al., ICLR 2021
"""

import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from models.backbones import get_non_bn_params, get_bn_params


# --------------------------------------------------------------------------- #
#  Optimiser factory                                                           #
# --------------------------------------------------------------------------- #

def make_sgd(model, lr=0.01, momentum=0.9, weight_decay=1e-4):
    return SGD(model.parameters(), lr=lr,
               momentum=momentum, weight_decay=weight_decay)


# --------------------------------------------------------------------------- #
#  Local training helpers                                                      #
# --------------------------------------------------------------------------- #

def local_train_epoch(model, loader, optimizer, criterion, device,
                      proximal_mu=None, global_params=None):
    """
    One local epoch.

    For FedProx: adds proximal term  (μ/2) * ||w - w_global||²
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        # FedProx proximal term (Section 2.4)
        if proximal_mu is not None and global_params is not None:
            prox = sum(
                ((p - gp.detach()) ** 2).sum()
                for p, gp in zip(model.parameters(), global_params)
            )
            loss = loss + (proximal_mu / 2.0) * prox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------------------------------- #
#  FedAvg                                                                      #
# --------------------------------------------------------------------------- #

class FedAvg:
    """
    FedAvg — sample-size-weighted averaging of local model updates.
    A strong, widely used baseline in medical FL (McMahan et al., 2017).
    """

    @staticmethod
    def aggregate(global_model, client_states, client_sizes):
        """
        Weighted average of client state dicts.

        Args:
            global_model:   current global model (updated in place)
            client_states:  list of client model state_dicts
            client_sizes:   list of client dataset sizes (for weighting)
        """
        total     = sum(client_sizes)
        weights   = [s / total for s in client_sizes]
        new_state = copy.deepcopy(client_states[0])

        for key in new_state:
            new_state[key] = sum(
                w * s[key].float() for w, s in zip(weights, client_states)
            )
        global_model.load_state_dict(new_state)
        return global_model

    @staticmethod
    def local_update(model, loader, device, local_epochs=3,
                     lr=0.01, momentum=0.9, weight_decay=1e-4):
        """Standard local training — no proximal term."""
        criterion = nn.CrossEntropyLoss()
        optimizer = make_sgd(model, lr, momentum, weight_decay)

        for _ in range(local_epochs):
            local_train_epoch(model, loader, optimizer, criterion, device)

        return model.state_dict(), len(loader.dataset)


# --------------------------------------------------------------------------- #
#  FedProx                                                                     #
# --------------------------------------------------------------------------- #

class FedProx:
    """
    FedProx — adds proximal regularizer to reduce client drift under non-IID data.
    Aggregation follows FedAvg. (Li et al., MLSys 2020)

    μ=0.01 as used in paper.
    """

    def __init__(self, mu=0.01):
        self.mu = mu

    def aggregate(self, global_model, client_states, client_sizes):
        """Same weighted aggregation as FedAvg."""
        return FedAvg.aggregate(global_model, client_states, client_sizes)

    def local_update(self, model, loader, device, global_model,
                     local_epochs=3, lr=0.01, momentum=0.9, weight_decay=1e-4):
        """Local training with proximal term w.r.t. global model params."""
        criterion    = nn.CrossEntropyLoss()
        optimizer    = make_sgd(model, lr, momentum, weight_decay)
        global_params = list(global_model.parameters())

        for _ in range(local_epochs):
            local_train_epoch(model, loader, optimizer, criterion, device,
                              proximal_mu=self.mu, global_params=global_params)

        return model.state_dict(), len(loader.dataset)


# --------------------------------------------------------------------------- #
#  SCAFFOLD                                                                    #
# --------------------------------------------------------------------------- #

class SCAFFOLD:
    """
    SCAFFOLD — control-variate correction for client drift.
    Local SGD uses zero momentum (Section 2.4 paper note).
    Aggregation remains sample-weighted. (Karimireddy et al., ICML 2020)
    """

    def __init__(self):
        # Server and client control variates (initialised lazily)
        self.server_cv = None
        self.client_cvs = {}

    def _init_cv(self, model):
        """Initialise zero control variates matching model parameters."""
        return {name: torch.zeros_like(p.data)
                for name, p in model.named_parameters()}

    def aggregate(self, global_model, client_states, client_sizes,
                  client_cv_deltas):
        """
        Update global model and server control variate.

        client_cv_deltas: list of {name: delta_tensor} from each client
        """
        # Standard weighted model aggregation
        FedAvg.aggregate(global_model, client_states, client_sizes)

        # Update server control variate: c ← c + Σ Δci / num_clients
        if self.server_cv is None:
            self.server_cv = self._init_cv(global_model)

        n = len(client_cv_deltas)
        for name in self.server_cv:
            self.server_cv[name] += sum(
                d[name] for d in client_cv_deltas
            ) / n

        return global_model

    def local_update(self, model, loader, device, client_id,
                     global_model, local_epochs=3, lr=0.01, weight_decay=1e-4):
        """
        SCAFFOLD local update with control-variate gradient correction.
        Zero momentum SGD as noted in Section 2.4.
        """
        if self.server_cv is None:
            self.server_cv = self._init_cv(model)
        if client_id not in self.client_cvs:
            self.client_cvs[client_id] = self._init_cv(model)

        criterion = nn.CrossEntropyLoss()
        # Zero momentum for SCAFFOLD (Section 2.4)
        optimizer = make_sgd(model, lr, momentum=0.0, weight_decay=weight_decay)

        client_cv = self.client_cvs[client_id]
        server_cv = self.server_cv

        w_old     = copy.deepcopy(model.state_dict())
        n_steps   = 0

        for _ in range(local_epochs):
            model.train()
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                loss   = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()

                # Apply SCAFFOLD correction to gradients
                for name, param in model.named_parameters():
                    if param.grad is not None and name in server_cv:
                        param.grad.data += (
                            server_cv[name].to(device) - client_cv[name].to(device)
                        )
                optimizer.step()
                n_steps += 1

        # Update client control variate
        w_new    = model.state_dict()
        cv_delta = {}
        for name in client_cv:
            # Δci = ci - c + (wi_old - wi_new) / (K * lr)
            delta = (
                client_cv[name]
                - server_cv[name]
                + (w_old[name].float() - w_new[name].float()) / (n_steps * lr)
            )
            cv_delta[name]               = delta
            self.client_cvs[client_id][name] = client_cv[name] - server_cv[name] + delta

        return model.state_dict(), len(loader.dataset), cv_delta


# --------------------------------------------------------------------------- #
#  FedBN                                                                       #
# --------------------------------------------------------------------------- #

class FedBN:
    """
    FedBN — aggregate only non-Batch Normalization parameters.
    BN parameters/statistics remain client-specific to preserve
    domain-dependent feature statistics. (Li et al., ICLR 2021)

    For evaluation: inject each client's cached BN stats before testing
    that client's domain (Section 2.5.5).
    """

    def __init__(self):
        self.client_bn_states = {}    # {client_id: bn_state_dict}

    def aggregate(self, global_model, client_states, client_sizes, client_ids):
        """
        Aggregate only non-BN parameters.
        Each client's BN params are stored locally and NOT aggregated.
        """
        total   = sum(client_sizes)
        weights = [s / total for s in client_sizes]

        # Weighted average of non-BN params only
        global_state = global_model.state_dict()
        non_bn_keys  = [k for k in global_state
                        if not any(t in k.lower()
                                   for t in ["bn", "batch_norm", "norm"])]

        for key in non_bn_keys:
            global_state[key] = sum(
                w * s[key].float()
                for w, s in zip(weights, client_states)
            )

        global_model.load_state_dict(global_state)

        # Cache each client's BN state
        for cid, state in zip(client_ids, client_states):
            bn_state = {k: v for k, v in state.items()
                        if any(t in k.lower()
                               for t in ["bn", "batch_norm", "norm"])}
            self.client_bn_states[cid] = bn_state

        return global_model

    def get_client_model(self, global_model, client_id):
        """
        Inject client-specific BN stats into a copy of the global model.
        Used for evaluation of each domain's test set (Section 2.5.5).
        """
        model     = copy.deepcopy(global_model)
        state     = model.state_dict()
        bn_state  = self.client_bn_states.get(client_id, {})
        state.update(bn_state)
        model.load_state_dict(state)
        return model

    def local_update(self, model, loader, device, local_epochs=3,
                     lr=0.01, momentum=0.9, weight_decay=1e-4):
        """Standard local training — BN params stay local."""
        criterion = nn.CrossEntropyLoss()
        optimizer = make_sgd(model, lr, momentum, weight_decay)

        for _ in range(local_epochs):
            local_train_epoch(model, loader, optimizer, criterion, device)

        return model.state_dict(), len(loader.dataset)
