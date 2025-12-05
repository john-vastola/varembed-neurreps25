#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 07:50:05 2025

@author: vastola
"""


import math, torch
from torch import nn


try:
    from torch.func import vmap, jacrev
except Exception:
    from functorch import vmap, jacrev
    
    
    

# Sample from Gaussian
def sample_q_gaussian(batch, d, sigma0=1.0, device='cpu', dtype=torch.double):
    return sigma0 * torch.randn(batch, d, device=device, dtype=dtype)


# Sample from uniform
def sample_q_uniform(batch, bounds, device='cpu', dtype=torch.double):
    b = torch.as_tensor(bounds, dtype=dtype, device=device)
    low, high = b[:, 0], b[:, 1]                    # [d], [d]
    u = torch.rand(batch, b.shape[0], device=device, dtype=dtype)
    
    return low + (high - low) * u                   # [B, d]







# Define Gaussian mixture given point cloud
class GaussianMixture:
    """Simple isotropic Gaussian mixture p_data in R^D (example)."""
    def __init__(self, weights, mus, sigmas):
        assert len(weights)==len(mus)==len(sigmas)
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.weights = self.weights / self.weights.sum()
        self.mus = [torch.as_tensor(m, dtype=torch.double) for m in mus]
        self.sigmas = [float(s) for s in sigmas]
        self.D = self.mus[0].numel()

    def log_prob(self, x):  # x: (..., D)
        x = x.to(dtype=torch.double)
        comps = []
        for w, m, s in zip(self.weights, self.mus, self.sigmas):
            diff2 = (x - m).pow(2).sum(dim=-1)
            logp = -0.5*diff2/(s*s) - 0.5*self.D*math.log(2*math.pi*s*s)
            comps.append(torch.log(w) + logp)
        return torch.logsumexp(torch.stack(comps, dim=-1), dim=-1)



# Make MLP
def make_mlp(in_dim, out_dim, width=128, depth=3, dtype=torch.double):
    layers = []
    last = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(last, width, dtype=dtype))
        last = width
    layers.append(nn.Linear(last, out_dim, dtype=dtype))
    net = nn.Sequential(*layers)

    # Light init for smooth derivatives (esp. for sine)
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                bound = math.sqrt(6.0 / fan_in)
                m.weight.uniform_(-bound, bound)
                m.bias.zero_()
    return net


# Jacobian utility func
def batch_jacobian(f, z):
    """
    Return J per-sample: [B, D, d], where f: R^d -> R^D, z: [B, d].
    """
    def f_single(z1):  # z1: [d]
        out = f(z1.unsqueeze(0)).squeeze(0)   # [D]
        return out
    return vmap(jacrev(f_single))(z)  # [B, D, d]



# Construct and train embedding
class NeuralEmbedding:
    def __init__(
        self, d, D, hidden=128, depth=3, 
        q_mode='gaussian',            # 'gaussian' or 'uniform'
        sigma0=1.0,                   # used if q_mode='gaussian'
        bounds=None,                  # used if q_mode='uniform'; shape [d, 2]
        device='cpu', dtype=torch.double, seed=0
    ):
        """
        - If q_mode='gaussian': z ~ N(0, sigma0^2 I_d)
        - If q_mode='uniform' : z ~ Uniform([low_i, high_i]) for each dim i (finite box)
        """
        torch.manual_seed(seed)
        self.d, self.D = d, D
        self.device, self.dtype = torch.device(device), dtype
        self.q_mode = q_mode
        self.sigma0 = float(sigma0)
        if q_mode == 'uniform':
            if bounds is None:
                # default symmetric box [-L, L]^d with L=1
                L = 1.0
                bounds = torch.tensor([[-L, L]] * d, dtype=dtype)
            self.bounds = torch.as_tensor(bounds, dtype=dtype, device=self.device)
            assert self.bounds.shape == (d, 2), "bounds must be [d, 2] for uniform q"
        else:
            self.bounds = None

        self.net = make_mlp(d, D, width=hidden, depth=depth, dtype=dtype).to(self.device)

    def forward(self, z):
        return self.net(z)

    def _sample_z(self, batch):
        if self.q_mode == 'gaussian':
            return sample_q_gaussian(batch, self.d, self.sigma0, device=self.device, dtype=self.dtype)
        else:
            return sample_q_uniform(batch, self.bounds, device=self.device, dtype=self.dtype)

    def objective_terms(self, z, log_p_data_fn, eps=1e-10):
        """
        z: [B, d] samples from q
        Returns kin (B,), logp (B,), cont (B,)
          kin  = 0.5 * log det(J^T J)
          logp = log p_data(phi(z))
          cont = trace(J^T J)
        """
        z = z.requires_grad_(True)
        phi = self.forward(z)                           # [B, D]
        J = batch_jacobian(lambda zz: self.forward(zz.unsqueeze(0)).squeeze(0), z)  # [B, D, d]

        if self.d == 1:
            g11 = (J[..., 0].pow(2).sum(dim=-1))                    # [B]
            logdet = torch.log(g11.clamp_min(eps))                  # [B]
            trG = g11
        else:
            Jt = J.transpose(-2, -1)                                # [B, d, D]
            G = Jt @ J                                              # [B, d, d]
            if self.d == 2:
                g11, g22 = G[:,0,0], G[:,1,1]
                g12 = G[:,0,1]
                detG = (g11 * g22 - g12.pow(2)).clamp_min(eps)
                logdet = torch.log(detG)
                trG = (g11 + g22)
            else:
                L = torch.linalg.cholesky(G + eps * torch.eye(self.d, device=G.device, dtype=G.dtype))
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
                trG = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)

        kin = 0.5 * logdet
        logp = log_p_data_fn(phi)                                   # [B]
        cont = trG
        return kin, logp, cont

    def train(
        self, log_p_data_fn, steps=5000, batch=1024, lr=1e-3,
        cont_lambda=1e-3, weight_decay=0.0, print_every=500, clip_grad=1.0
    ):
        """
        Stochastic ascent on:
          E_q[ 0.5 log det(J^T J) + log p_data(phi(z)) ] - cont_lambda * E_q[ tr(J^T J) ].
        Note: The -E_q[log q] term is phi-independent and omitted for optimization.
        """
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        for t in range(1, steps+1):
            z = self._sample_z(batch)
            kin, logp, cont = self.objective_terms(z, log_p_data_fn)

            J_obj = (kin + logp).mean()
            loss = -(J_obj - cont_lambda * cont.mean())   # ascent on J_obj, penalize continuity

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=clip_grad)
            opt.step()

            if (t % print_every) == 0 or t == 1:
                print(f"[{t:5d}] J={J_obj.item():.4f}  kin={kin.mean().item():.4f}  "
                      f"logp={logp.mean().item():.4f}  cont={cont.mean().item():.4f}")

        return self.net
    
    
    
    