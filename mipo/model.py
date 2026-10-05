"""Scalar response fields with E(3)-equivariant vector messages.

The finite graph implementation is an operator-inspired discretization. It does
not establish continuum resolution independence or identify causal mechanisms.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import AA, TASKS


def mlp(inp, hidden, out, dropout=0.):
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, out))


def gather_nodes(x, edges):
    return x[torch.arange(len(x), device=x.device)[:, None, None], edges]


def chemistry_table():
    # Kyte-Doolittle hydropathy, approximate residue volume, charge at neutral pH,
    # polar side-chain indicator. These are descriptors, not physical energies.
    hydro = [1.8, 2.5, -3.5, -3.5, 2.8, -.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -.8, -.7, 4.2, -.9, -1.3]
    vol = [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89., 116.1, 140., 227.8, 193.6]
    return torch.tensor([[hydro[i]/5, vol[i]/200, {"D": -1, "E": -1, "K": 1, "R": 1, "H": .1}.get(a, 0),
                          float(a in "STNQDEKRHY")] for i, a in enumerate(AA)], dtype=torch.float32)


class PerturbationLayer(nn.Module):
    def __init__(self, dim, anchors=16, use_global=True, use_structure=True):
        super().__init__()
        self.use_global, self.use_structure = use_global, use_structure
        self.gate = mlp(dim*2+5, dim, 1)
        self.message = nn.Linear(dim, dim, bias=False)
        self.vector_coeff = nn.Linear(dim, 1, bias=False)
        self.update = nn.Linear(dim*3+1, dim, bias=False)
        self.step = nn.Parameter(torch.tensor(-1.))
        self.anchor_logits = nn.Linear(dim, anchors)

    def forward(self, h, context, b):
        edges, mask = b["edge"], b["edge_mask"]
        src = gather_nodes(context, edges)
        dst = context[:, :, None].expand_as(src)
        geometry = b["edge_features"]
        if not self.use_structure:
            geometry = geometry.clone()
            geometry[..., [0, 1, 2, 4]] = 0
        logits = self.gate(torch.cat([src, dst, geometry], -1)).squeeze(-1)
        weights = torch.softmax(logits.masked_fill(~mask, -1e4), -1)*mask
        neighbor_h = gather_nodes(h, edges)
        local = (weights[..., None]*self.message(neighbor_h)).sum(2)
        vector = torch.zeros((*b["unit"].shape[:3], 3), device=h.device, dtype=h.dtype)
        if self.use_structure:
            coeff = self.vector_coeff(neighbor_h).squeeze(-1)*weights
            vector = (b["unit"]*(coeff[:, None]*b["edge_weight"])[..., None]).sum(3)
        conformer_mask = b["conformer_mask"][..., None]
        magnitude = ((torch.linalg.vector_norm(vector.float(), dim=-1)*conformer_mask).sum(1)
                     / conformer_mask.sum(1).clamp_min(1)).to(h.dtype).unsqueeze(-1)
        global_message = torch.zeros_like(h)
        if self.use_global:
            # Learned latent landmarks: gather and redistribute in O(N*r*d).
            assign = self.anchor_logits(context)
            pool = torch.softmax(assign.masked_fill(~b["node_mask"][..., None], -1e4), dim=1)
            anchors = torch.einsum("bnr,bnd->brd", pool, h)
            global_message = torch.einsum("bnr,brd->bnd", torch.softmax(assign, -1), anchors)
        delta = torch.tanh(self.update(torch.cat([h, local, global_message, magnitude], -1)))
        h = (h+torch.sigmoid(self.step)*delta)*b["node_mask"][..., None]
        return h, vector


class MIPO(nn.Module):
    def __init__(self, esm_dim, metadata_size, config, seen_tasks=None):
        super().__init__()
        self.config = config
        dim = int(config.get("hidden", 128))
        dropout = config.get("dropout", .1)
        self.mode = config.get("model", "mipo")
        self.use_chemistry = config.get("chemistry", True)
        self.project = nn.Sequential(nn.Linear(esm_dim, dim), nn.LayerNorm(dim))
        self.aa = nn.Embedding(20, 24)
        self.register_buffer("chemistry", chemistry_table())
        self.impulse = nn.Linear(28, dim, bias=False)
        self.impulse_gate = mlp(dim, dim, dim)
        self.llr_proj = mlp(2, dim, dim)
        radius = config.get("window_radius", 25)
        self.relative = nn.Parameter(torch.randn(2*radius+1, dim)*.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim*2, dropout=dropout, batch_first=True, norm_first=True)
        self.sequence = nn.TransformerEncoder(layer, num_layers=config.get("sequence_layers", 2), enable_nested_tensor=False)
        self.layers = nn.ModuleList([PerturbationLayer(dim, config.get("anchors", 16), config.get("global_kernel", True), config.get("structure", True))
                                     for _ in range(config.get("field_layers", 3))])
        self.fusion_gate = mlp(dim*4+1, dim, 4)
        if config.get("fusion_gate_init") == "switch":
            # Switch-style router init: small truncated-normal weights so no piece
            # starts dominant, zero bias. Default PyTorch init is kept otherwise.
            final = self.fusion_gate[-1]
            torch.nn.init.trunc_normal_(final.weight, std=math.sqrt(0.1/final.in_features))
            torch.nn.init.zeros_(final.bias)
        self.fusion = mlp(dim*4, dim*2, dim, dropout)
        self.mechanisms = mlp(dim, dim, config.get("mechanisms", 8))
        self.physical_readout = nn.Linear(dim, 2, bias=False)
        self.task_embedding = nn.Embedding(len(TASKS)+1, 24)
        self.meta_embedding = nn.Embedding(metadata_size, 16, padding_idx=0)
        zdim = config.get("mechanisms", 8)
        self.hyper = mlp(40, 64, 2*(zdim+1))
        self.fixed = nn.ModuleList([nn.Linear(zdim, 2) for _ in range(len(TASKS)+1)])
        # Optional zero-shot anchor: the network predicts a correction to a per-task affine
        # function of the cached LLR, so fitting the training assays cannot drive the
        # zero-shot signal out of the prediction. Held-out assays inherit their task's
        # coefficient, which assumes the task's training assays share its score direction.
        # Independent of use_llr (which adds LLR as a fusion input): the residual anchors the
        # output on the cached LLR and zeroes itself per-row via the presence mask b["llr"][:,1],
        # so it works even when LLR is not a fusion modality -- the case these configs run
        # (use_llr: false). Gating it on use_llr silently disabled the correction-on-top-of-LLR head.
        self.llr_residual = bool(config.get("llr_residual", False))
        self.llr_weight = nn.Parameter(torch.zeros(len(TASKS)+1))
        self.llr_bias = nn.Parameter(torch.zeros(len(TASKS)+1))
        # Assay-agnostic linear readout. The per-assay hypernetwork head is the main
        # extrapolation liability on held-out genes; blending it toward this single shared
        # head trades training-assay expressivity for cross-gene transfer (a near-linear head
        # is what has been shown to transfer to unseen genes). head_shrinkage in [0,1]:
        # 0 keeps the conditioned head unchanged, 1 uses the shared head alone. The per-task
        # LLR residual anchor still supplies per-task direction, so shrinkage=1 is not zero-shot.
        self.head_shrinkage = min(1., max(0., float(config.get("head_shrinkage", 0.))))
        self.shared_readout = nn.Linear(zdim, 2)
        # Per-branch ranking probes for the auxiliary branch loss. Created only when the
        # loss is on, so older checkpoints restore unchanged and the default path is untouched.
        self.branch_probes = nn.ModuleList([nn.Linear(dim, 1) for _ in range(4)]) if config.get("branch_weight", 0.) else None
        self.register_buffer("seen_tasks", torch.tensor(seen_tasks if seen_tasks is not None else [True]*len(TASKS), dtype=torch.bool))

    def forward(self, b):
        B = len(b["wt"])
        context = self.project(b["esm"])
        indices = torch.arange(B, device=context.device)
        site_context = context[indices, b["site"]]
        chem = self.chemistry[b["mut"]]-self.chemistry[b["wt"]]
        if not self.use_chemistry:
            chem = torch.zeros_like(chem)
        difference = torch.cat([self.aa(b["mut"])-self.aa(b["wt"]), chem], -1)
        impulse = torch.tanh(self.impulse(difference))*torch.sigmoid(self.impulse_gate(site_context))
        h = torch.zeros_like(context)
        # Under autocast, project() ends in LayerNorm and stays fp32 while impulse comes from a
        # Linear in fp16. Index assignment is the one op here that will not promote, so cast.
        h[indices, b["site"]] = impulse.to(h.dtype)
        vector = torch.zeros((*b["unit"].shape[:3], 3), device=h.device, dtype=h.dtype)
        if self.mode != "esm_mlp":
            for layer in self.layers:
                h, vector = layer(h, context, b)
        mask = b["node_mask"][..., None]
        field_pool = (h*mask).sum(1)/mask.sum(1).clamp_min(1)
        local_field = h[indices, b["site"]]
        window = self.project(b["seq_window"])+self.relative
        if self.mode == "esm_mlp":
            local_seq = site_context
            field_pool = impulse
        else:
            local_seq = self.sequence(window, src_key_padding_mask=~b["seq_mask"])[:, self.config.get("window_radius", 25)]
        global_context = self.project(b["global"])
        llr = self.llr_proj(b["llr"]) if self.config.get("use_llr", True) else torch.zeros_like(impulse)
        pieces = torch.stack([local_seq+local_field, field_pool, global_context, impulse+llr], 1)
        quality = b["confidence"][indices, b["site"]][:, None] if self.config.get("structure", True) else torch.zeros((B, 1), device=h.device)
        gates = torch.softmax(self.fusion_gate(torch.cat([pieces.flatten(1), quality], -1)), -1)
        if self.training and self.config.get("modality_dropout", .1) > 0:
            keep = torch.rand((B, 4), device=h.device) >= self.config.get("modality_dropout", .1)
            keep[:, 3] = True
            gates = gates*keep
            gates = gates/gates.sum(-1, keepdim=True)
        branch_mu = (torch.stack([probe(pieces[:, i]).squeeze(-1) for i, probe in enumerate(self.branch_probes)], -1)
                     if self.branch_probes is not None else None)
        fused = self.fusion((pieces*gates[..., None]).flatten(1))
        z = torch.tanh(self.mechanisms(fused))
        task = b["task"].clone()
        task[~self.seen_tasks[task]] = len(TASKS)
        if self.training:
            task[torch.rand(B, device=h.device) < .15] = len(TASKS)
        if self.mode == "fixed_heads":
            pred = torch.stack([head(z) for head in self.fixed], 1)[indices, task]
        else:
            assay = torch.cat([self.task_embedding(task), self.meta_embedding(b["meta"]).mean(1)], -1)
            params = self.hyper(assay).reshape(B, 2, -1)
            pred = (params[..., :-1]*z[:, None]).sum(-1)/math.sqrt(z.shape[-1])+params[..., -1]
            if self.head_shrinkage:
                pred = (1-self.head_shrinkage)*pred+self.head_shrinkage*self.shared_readout(z)
        mu = pred[:, 0]
        if self.llr_residual:
            # b["llr"] is [signed LLR/10, present]; an absent score contributes nothing.
            anchor = self.llr_weight[task]*b["llr"][:, 0]+self.llr_bias[task]
            mu = mu+anchor*b["llr"][:, 1]
        sigma = F.softplus(pred[:, 1])+.05
        return {"mu": mu, "sigma": sigma, "field": h, "vector": vector, "latent": z, "gates": gates,
                "branch_mu": branch_mu,
                "physical_proxies": self.physical_readout(h)}
