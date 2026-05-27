"""Pretrain optimizer helpers."""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper around a base optimizer."""

    def __init__(self, params, base_optimizer_cls, *, rho: float = 0.05, **kwargs):
        if rho <= 0:
            raise ValueError("SAM rho must be > 0")
        defaults = dict(rho=float(rho), **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                e_w = param.grad * scale.to(param.device)
                param.add_(e_w)
                self.state[param]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.sub_(self.state[param]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires explicit first_step/second_step or a closure")
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            param.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)
