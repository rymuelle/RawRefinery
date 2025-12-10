import torch

def match_colors_linear(
    src: torch.Tensor, 
    tgt: torch.Tensor, 
    sample_fraction: float = 0.05
):
    """
    Fit per-channel affine color transforms:
        tgt ≈ scale * src + bias

    Args:
        src: [B, C, H, W] source tensor
        tgt: [B, C, H, W] target tensor
        sample_fraction: fraction of pixels to use for fitting

    Returns:
        transformed_src: source after color matching
        scale: [B, C]
        bias:  [B, C]
    """

    B, C, H, W = src.shape
    device = src.device

    # Flatten spatial dims
    src_flat = src.view(B, C, -1)
    tgt_flat = tgt.view(B, C, -1)

    # Sample subset of pixels
    N = src_flat.shape[-1]
    k = max(64, int(N * sample_fraction))

    idx = torch.randint(0, N, (k,), device=device)

    src_s = src_flat[..., idx]  # [B, C, k]
    tgt_s = tgt_flat[..., idx]

    # Compute scale and bias using least squares
    # scale = cov(src, tgt) / var(src)
    src_mean = src_s.mean(-1, keepdim=True)
    tgt_mean = tgt_s.mean(-1, keepdim=True)

    src_centered = src_s - src_mean
    tgt_centered = tgt_s - tgt_mean

    var_src = (src_centered ** 2).mean(-1)
    cov = (src_centered * tgt_centered).mean(-1)

    scale = cov / (var_src + 1e-8)            # [B, C]
    bias = tgt_mean.squeeze(-1) - scale * src_mean.squeeze(-1)

    # Apply correction
    scale_ = scale.view(B, C, 1, 1)
    bias_ = bias.view(B, C, 1, 1)
    transformed = src * scale_ + bias_

    return transformed, scale, bias
