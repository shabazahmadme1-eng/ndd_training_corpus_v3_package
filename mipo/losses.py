import torch
from torch.nn import functional as F


def contrast_term(mu, batch):
    """Rank the gap between two assays that measured the same substitution.

    For a variant measured by both assays, d = y_second-y_first separates the mechanisms:
    a folding-driven loss moves both readouts together and leaves d near zero, while a
    catalytic-site loss moves only one. Ranking d across variants asks the model to
    reproduce that separation, and drops out the per-assay offsets that a direct
    regression on d would otherwise have to fit. Same variant, same batch only.
    """
    same = (batch["variant_id"][:, None] == batch["variant_id"][None, :]) & (batch["group"][:, None] != batch["group"][None, :])
    first, second = torch.nonzero(same & torch.triu(torch.ones_like(same), diagonal=1).bool(), as_tuple=True)
    if len(first) < 2:
        return mu.sum()*0
    gap_y = (batch["y"].float()[second]-batch["y"].float()[first])
    gap_mu = mu[second]-mu[first]
    delta = gap_y[:, None]-gap_y[None, :]
    mask = (delta.abs() > 1e-6) & torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1)
    if not mask.any():
        return mu.sum()*0
    order = F.softplus(-(gap_mu[:, None]-gap_mu[None, :])*delta.sign())
    return order[mask].mean()


def pairwise_ranking(pred, batch, weight):
    """Within-assay pairwise ranking loss; shared by the joint head and branch probes."""
    y = batch["y"].float()
    delta = y[:, None]-y[None, :]
    mask = (batch["group"][:, None] == batch["group"][None, :]) & (delta.abs() > 1e-6)
    mask &= batch["row_id"][:, None] != batch["row_id"][None, :]
    mask &= torch.triu(torch.ones_like(mask), diagonal=1).bool()
    ranks = F.softplus(-(pred[:, None]-pred[None, :])*delta.sign())
    pair_weight = torch.sqrt(weight[:, None]*weight[None, :])
    return (ranks*pair_weight)[mask].mean() if mask.any() else pred.sum()*0


def branch_term(branch_mu, batch, weight):
    """Mean pairwise ranking loss over the four fusion branches, each scored standalone.

    Gives a dominated branch (e.g. the structural field) its own gradient signal so it
    must learn ranking even while the joint head is carried by a stronger branch.
    """
    return torch.stack([pairwise_ranking(branch_mu[:, i], batch, weight) for i in range(branch_mu.shape[1])]).mean()


def load_balance(gates):
    """Switch-style load-balancing pressure toward uniform fusion-gate usage.

    4 * sum(mean(gates, 0)^2): 1.0 at uniform, 4.0 at one-hot. Bounded scale
    keeps the weight interpretable next to the other loss terms.
    """
    density = gates.float().mean(0)
    return 4*(density*density).sum()


def loss_function(output, batch, ranking_weight=.2, field_weight=1e-5, auxiliary_weight=.1, contrast_weight=0., balance_weight=0., branch_weight=0.):
    # FP32 likelihood math even when the encoder runs under mixed precision.
    mu, sigma, y = output["mu"].float(), output["sigma"].float(), batch["y"].float()
    weight = batch.get("sample_weight", torch.ones_like(y)).float()
    nll = ((.5*((mu-y)/sigma).square()+sigma.log())*weight).mean()
    huber = (F.huber_loss(mu, y, delta=1., reduction='none')*weight).mean()
    # All within-assay pairs; exclude self-pairs, ties, and repeated samples.
    ranking = pairwise_ranking(mu, batch, weight)
    field = output["field"].float().square().mean(-1)
    regularizer = (field*batch["node_mask"]).sum()/batch["node_mask"].sum().clamp_min(1)
    aux_mask = batch["aux_mask"] & batch["node_mask"][..., None]
    aux = F.huber_loss(output["physical_proxies"].float()[aux_mask], batch["aux_target"].float()[aux_mask]) if aux_mask.any() else mu.sum()*0
    contrast = contrast_term(mu, batch) if contrast_weight else mu.sum()*0
    gates = output.get("gates")
    balance = load_balance(gates) if balance_weight and gates is not None else mu.sum()*0
    branch_mu = output.get("branch_mu")
    branch = branch_term(branch_mu, batch, weight) if branch_weight and branch_mu is not None else mu.sum()*0
    total = (nll+.1*huber+ranking_weight*ranking+field_weight*regularizer
             + auxiliary_weight*aux+contrast_weight*contrast+balance_weight*balance+branch_weight*branch)
    return total, {"nll": nll.detach(), "huber": huber.detach(), "ranking": ranking.detach(),
                   "field": regularizer.detach(), "auxiliary": aux.detach(), "contrast": contrast.detach(),
                   "balance": balance.detach(), "branch": branch.detach()}
