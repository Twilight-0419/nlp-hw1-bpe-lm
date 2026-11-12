import torch
import torch.nn as nn

torch.manual_seed(42)

def Cross_Entropy_Loss(
        x:torch.Tensor,
        logits:torch.Tensor
):
    logits=logits-logits.max(dim=-1,keepdim=True).values
    log_probs=logits-torch.log(torch.exp(logits).sum(dim=-1,keepdim=True))
    vocab_size=log_probs.shape[-1]
    log_probs=log_probs.reshape(-1,vocab_size)
    x=x.reshape(-1,1).long()
    return -log_probs.gather(1,x).mean()



from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取与参数 p 相关的状态。

                t = state.get("t", 0)
                # 获取迭代次数，如果不存在则使用初始值 0。

                grad = p.grad.data
                # 获取损失函数相对于参数 p 的梯度。

                p.data -= lr / math.sqrt(t + 1) * grad
                # 原地更新权重张量。

                state["t"] = t + 1
                # 迭代次数加 1。


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        assert lr >= 0, f"Invalid learning rate: {lr}"
        assert 0 <= betas[0] < 1, f"Invalid beta1: {betas[0]}"
        assert 0 <= betas[1] < 1, f"Invalid beta2: {betas[1]}"
        assert eps >= 0, f"Invalid epsilon: {eps}"
        assert weight_decay >= 0, f"Invalid weight decay: {weight_decay}"
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if "t" not in state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t = state["t"] + 1
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t
        return loss



def Lr_Cos_Schedule(
        t,alpha_max,alpha_min,T_w,T_c
):
    if t<T_w:
        alpha_t=t/T_w*alpha_max
    elif t>=T_w and t<=T_c:
        alpha_t=alpha_min+1/2*(1+math.cos((t-T_w)/(T_c-T_w)*torch.pi))*(alpha_max-alpha_min)
    elif t>T_c:
        alpha_t=alpha_min
    return alpha_t



def Gridient_Clipping(
        paramlist,max_norm,epsilon=1e-6
):
    norm_sum=0
    for param in paramlist:
        if param.grad is not None:
            norm_sum=norm_sum+(param.grad.data**2).sum()
    norm_sum=torch.sqrt(norm_sum)
    if norm_sum>=max_norm:
        for param in paramlist:
            if param.grad is not None:
                param.grad.data=param.grad.data*max_norm/(norm_sum+epsilon)