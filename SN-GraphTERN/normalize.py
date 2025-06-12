import torch


def translate_and_rotate(S_obs, S_trgt, unit_scale=True):
    # [step, node, 2]
    hist_abs, hist_rel = S_obs.squeeze(0)
    fut_abs, fut_rel = S_trgt.squeeze(0)
    origin = hist_abs[-1].mean(dim=0)
    initial = hist_abs[0].mean(dim=0)
    angle = torch.atan2(origin[1] - initial[1], origin[0] - initial[0])
    mat = torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]).to(S_obs.device)
    scale = torch.linalg.norm(hist_abs[1:] - hist_abs[:-1], dim=-1).mean().item() / 7 * 19
    scale = min(scale, 1.)
    if unit_scale:
        scale = 1.
    hist_translated = hist_abs - origin.unsqueeze(0).unsqueeze(0)
    fut_translated = fut_abs - origin.unsqueeze(0).unsqueeze(0)
    hist_rotated = (mat[None, None, ...] @ hist_translated[..., None]).squeeze(-1)
    fut_rotated = (mat[None, None, ...] @ fut_translated[..., None]).squeeze(-1)
    hist_rel_rotated = (mat[None, None, ...] @ hist_rel[..., None]).squeeze(-1)
    fut_rel_rotated = (mat[None, None, ...] @ fut_rel[..., None]).squeeze(-1)
    return torch.stack([hist_rotated, hist_rel_rotated]).unsqueeze(0) / scale, torch.stack([fut_rotated, fut_rel_rotated]).unsqueeze(0) / scale, origin, angle, scale


def inv_translate_and_rotate(S_obs_, S_trgt_, pred, origin, angle, scale=1.):
    hist_abs, hist_rel = S_obs_.squeeze(0) * scale
    fut_abs, fut_rel = S_trgt_.squeeze(0) * scale
    mat = torch.tensor([[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]]).to(S_obs_.device)
    hist_rotated = (mat[None, None, ...] @ hist_abs[..., None]).squeeze(-1)
    fut_rotated = (mat[None, None, ...] @ fut_abs[..., None]).squeeze(-1)
    hist_rel_rotated = (mat[None, None, ...] @ hist_rel[..., None]).squeeze(-1)
    fut_rel_rotated = (mat[None, None, ...] @ fut_rel[..., None]).squeeze(-1)
    hist_translated = hist_rotated + origin.unsqueeze(0).unsqueeze(0)
    fut_translated = fut_rotated + origin.unsqueeze(0).unsqueeze(0)
    pred = pred * scale
    pred_rotated = (mat[None, None, ...] @ pred[..., None]).squeeze(-1)
    pred_translated = pred_rotated + origin.unsqueeze(0).unsqueeze(0)
    return torch.stack([hist_translated, hist_rel_rotated]).unsqueeze(0), torch.stack([fut_translated, fut_rel_rotated]).unsqueeze(0), pred_translated