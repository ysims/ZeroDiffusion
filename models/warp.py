import torch


def WARP(output, target, delta=1.0):
    """
    :param output: (num_samples, num_classes)
    :param target: (num_samples)
    :param weights: (num_samples)
    :param delta: default 1.0
    :return:
    """
    device = output.device
    num_samples, num_classes = output.size()

    delta = torch.tensor(data=delta, dtype=torch.float32, device=device)

    target_mask = torch.nn.functional.one_hot(target, num_classes=num_classes).to(
        device=device
    )  # (num_samples, num_classes)
    inverse_target_mask = torch.ones_like(target_mask, device=device).sub(target_mask)

    target = output.mul(target_mask)
    target = target.matmul(
        torch.ones(num_classes, num_classes, device=device)
    )  # (num_samples, num_classes)

    loss = (output - target + delta).mul(
        inverse_target_mask
    )  # (num_samples, num_classes)
    loss = loss.clamp_min(0.0)

    ranks = (loss > 0.0).sum(dim=1).clamp_min(1).sub(1)  # (num_samples)

    betas = 1.0 / torch.arange(1, num_classes, dtype=torch.float32)  # (num_classes - 1)
    betas = betas.cumsum(dim=0) / torch.arange(1, num_classes, dtype=torch.float32)
    betas = betas.to(device=device)
    betas = betas[ranks]  # (num_samples)

    obj = loss.sum(dim=1, dtype=torch.float32).mul(betas)  # (num_samples)
    obj = obj.mean()

    return obj
