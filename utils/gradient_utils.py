import torch

def compute_forward_pass_info(model):
    weight_sum = 0.0
    grad_sum = 0.0
    for name, p in model.named_parameters():
        if torch.isfinite(p).all().item() and p.requires_grad and type(p.grad) != type(None):
            weight_sum += torch.sum(torch.abs(p.data))
            grad_sum += torch.sum(torch.abs(p.grad))

    return weight_sum, grad_sum