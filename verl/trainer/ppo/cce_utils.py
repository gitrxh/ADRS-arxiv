"""
CCE: Counterfactual Credit from privileged self-teachers (bootstrap estimator).

Thesis: privileged signals (skill text) should be *verified by intervention*, not
*trusted by likelihood*. The interventional quantity we want at a state s is the
value gain from switching the action to the teacher (self + skill) policy:

    CCE(s) = E_{a ~ pi_teacher}[ Q^pi(s, a) ] - E_{a ~ pi_student}[ Q^pi(s, a) ]

The gold estimator (Stage 1) measures this with on-policy do()-rollouts. This file
implements the *bootstrap* estimator (Stage 0): a pure off-policy estimate computed
from already-logged group data, with zero extra rollouts.

Key identity: the per-action importance log-ratio between teacher and student is
exactly delta, the quantity STAR already computes:

    log [ pi_teacher(a|s) / pi_student(a|s) ] = sum_t ( teacher_lp_t - student_lp_t ) = delta_sum

So, within a GiGPO anchor-state group (trajectories that visited the *same* state s),
a self-normalized importance sampling (SNIS) estimate of the teacher-policy value is:

    E_teacher[Q] ~= sum_m w_m G_m / sum_m w_m,   w_m = exp(delta_sum_m / T)
    E_student[Q] ~= mean_m G_m
    CCE_boot(s)  =  E_teacher[Q] - E_student[Q]

where G_m = step_rewards[m] is the GiGPO discounted return-to-go achieved after the
action actually taken at s (an on-policy Monte-Carlo estimate of Q(s, a_m)).

This is distinct from:
  * TVA L3 (compute_tva_vectorized): E[R | teacher-confident] - E[R | teacher-divergent],
    a correlational split on the teacher's *absolute* confidence, not an IS reweighting.
  * SC-TCA (TRACE): Pearson Corr(delta, reward | group), also correlational.

The discriminative claim this enables: there exist high-delta states (skill strongly
changes the action) where CCE_boot ~= 0 (the change does not improve value). A purely
delta-driven credit signal rewards those tokens; CCE does not.
"""

import numpy as np
import torch


def aggregate_per_row(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sum / count masked values over the token axis for each row.

    In the flat multi-turn batch each row IS one step (turn) of one trajectory,
    so a per-row aggregate is a per-(state, action-taken) aggregate.

    Args:
        values: (bs, response_length)
        response_mask: (bs, response_length)

    Returns:
        row_sum:   (bs,) sum of masked values per row.
        row_count: (bs,) number of valid (action) tokens per row, clamped to >= 1.
    """
    masked = values * response_mask
    row_sum = masked.sum(dim=-1)
    row_count = response_mask.sum(dim=-1).clamp(min=1)
    return row_sum, row_count


def _pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    if x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=eps)
    return (x @ y / denom).item()


def compute_cce_bootstrap(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_rewards: torch.Tensor,
    group_uids,
    is_temperature: float = 1.0,
    clip_logw: float = 8.0,
    min_group_size: int = 2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Bootstrap (off-policy SNIS) estimate of counterfactual credit per row.

    Args:
        teacher_log_probs: (bs, response_length) — teacher = self + skill.
        student_log_probs: (bs, response_length) — student (old_log_probs).
        response_mask:     (bs, response_length)
        step_rewards:      (bs,) — GiGPO discounted return-to-go per row (=Q(s, a_taken)).
        group_uids:        sequence of length bs — anchor-state group id per row
                           (rows sharing a uid visited the same state s).
        is_temperature:    T in w = exp(delta_sum / T). T=1 is exact SNIS; larger T
                           softens the weights (lower variance, more bias).
        clip_logw:         clip the (group-centered) log-weight to [-clip, +clip]
                           before exp, to bound IS variance.
        min_group_size:    groups smaller than this get CCE = 0 (no counterfactual mass).

    Returns:
        cce_per_row:        (bs,) CCE_boot(group(row)), broadcast to every row in the group.
        delta_mean_per_row: (bs,) mean per-token delta on the action (skill intervention).
        delta_sum_per_row:  (bs,) summed delta on the action (= IS log-ratio of the action).
        ess_per_row:        (bs,) Kish effective sample size of the IS estimate of row's group.
        metrics:            dict of diagnostics.
    """
    device = teacher_log_probs.device
    bs = teacher_log_probs.shape[0]

    rmask = response_mask.float()
    delta = (teacher_log_probs.detach().float() - student_log_probs.detach().float()) * rmask
    delta_sum, row_count = aggregate_per_row(delta, rmask)
    delta_mean = delta_sum / row_count

    G = step_rewards.detach().to(device).float().view(-1)
    if G.shape[0] != bs:
        # defensive: truncate / pad to bs
        if G.shape[0] > bs:
            G = G[:bs]
        else:
            G = torch.cat([G, torch.zeros(bs - G.shape[0], device=device)])

    # Map string/object group uids -> contiguous integer ids.
    uids = np.asarray(group_uids, dtype=object)
    unique_uids, inverse = np.unique(uids, return_inverse=True)
    gid = torch.as_tensor(inverse, device=device, dtype=torch.long)
    num_groups = len(unique_uids)

    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, gid, torch.ones(bs, device=device))

    # Group-wise max of delta_sum for numerically stable softmax / SNIS.
    group_max = torch.full((num_groups,), float("-inf"), device=device)
    group_max = group_max.scatter_reduce(0, gid, delta_sum / is_temperature, reduce="amax", include_self=True)
    logw = (delta_sum / is_temperature) - group_max[gid]
    logw = logw.clamp(min=-clip_logw, max=clip_logw)
    w = torch.exp(logw)

    # SNIS teacher value:   sum_m w_m G_m / sum_m w_m   (per group)
    wsum = torch.zeros(num_groups, device=device)
    wGsum = torch.zeros(num_groups, device=device)
    Gsum = torch.zeros(num_groups, device=device)
    w2sum = torch.zeros(num_groups, device=device)
    wsum.scatter_add_(0, gid, w)
    wGsum.scatter_add_(0, gid, w * G)
    Gsum.scatter_add_(0, gid, G)
    w2sum.scatter_add_(0, gid, w * w)

    teacher_val = wGsum / wsum.clamp(min=eps)
    student_val = Gsum / group_count.clamp(min=1)
    group_cce = teacher_val - student_val

    # Effective sample size of the IS estimate per group (Kish): (sum w)^2 / sum w^2.
    group_ess = (wsum * wsum) / w2sum.clamp(min=eps)

    # Zero out groups without enough counterfactual mass.
    valid_group = group_count >= min_group_size
    group_cce = torch.where(valid_group, group_cce, torch.zeros_like(group_cce))

    cce_per_row = group_cce[gid]
    ess_per_row = group_ess[gid]

    with torch.no_grad():
        valid_rows = valid_group[gid]
        vc = cce_per_row[valid_rows]
        vd = delta_mean[valid_rows]
        metrics = {
            "cce/mean": vc.mean().item() if vc.numel() > 0 else 0.0,
            "cce/std": vc.std().item() if vc.numel() > 1 else 0.0,
            "cce/abs_mean": vc.abs().mean().item() if vc.numel() > 0 else 0.0,
            "cce/positive_ratio": (vc > 0).float().mean().item() if vc.numel() > 0 else 0.0,
            "cce/delta_mean": vd.mean().item() if vd.numel() > 0 else 0.0,
            "cce/delta_std": vd.std().item() if vd.numel() > 1 else 0.0,
            # The headline number: how much does the (cheap) skill-intervention magnitude
            # actually predict causal value? Low |corr| motivates the whole paper.
            "cce/corr_delta_cce": _pearson(vd, vc) if vc.numel() > 1 else 0.0,
            "cce/ess_mean": group_ess[valid_group].mean().item() if valid_group.any() else 0.0,
            "cce/num_groups": float(num_groups),
            "cce/num_valid_groups": float(valid_group.sum().item()),
        }
        # High-delta-low-CCE mass: among the top-quartile |delta| decisions,
        # what fraction have near-zero causal value? (the discriminative claim)
        if vd.numel() >= 4:
            thr_delta = torch.quantile(vd.abs(), 0.75)
            cce_scale = vc.abs().mean().clamp(min=eps)
            high_delta = vd.abs() >= thr_delta
            low_cce = vc.abs() < 0.25 * cce_scale
            denom = high_delta.float().sum().clamp(min=1)
            metrics["cce/highdelta_lowcce_frac"] = (
                (high_delta & low_cce).float().sum() / denom
            ).item()
        else:
            metrics["cce/highdelta_lowcce_frac"] = 0.0

    return cce_per_row, delta_mean, delta_sum, ess_per_row, metrics


def dump_cce_records(
    path: str,
    global_step: int,
    group_uids,
    delta_mean: torch.Tensor,
    delta_sum: torch.Tensor,
    returns: torch.Tensor,
    cce_per_row: torch.Tensor,
    ess_per_row: torch.Tensor,
    won=None,
    tag: str = "real_skill",
) -> None:
    """
    Append one JSONL record per row (turn) for offline analysis (scripts/analyze_cce.py).

    Each record is a single decision point with its skill-intervention magnitude and
    its bootstrap counterfactual credit, so the offline script can draw the
    delta-vs-CCE scatter without any GPU / model.
    """
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    uids = np.asarray(group_uids, dtype=object)
    dm = delta_mean.detach().float().cpu().tolist()
    ds = delta_sum.detach().float().cpu().tolist()
    g = returns.detach().float().cpu().view(-1).tolist()
    c = cce_per_row.detach().float().cpu().tolist()
    e = ess_per_row.detach().float().cpu().tolist()
    if won is not None and not isinstance(won, (list, tuple)):
        won = list(won)

    n = len(dm)
    with open(path, "a") as f:
        for i in range(n):
            rec = {
                "global_step": int(global_step),
                "group_uid": str(uids[i]) if i < len(uids) else f"row_{i}",
                "delta_mean": dm[i],
                "delta_sum": ds[i],
                "return": g[i] if i < len(g) else 0.0,
                "cce": c[i],
                "ess": e[i],
                "tag": tag,
            }
            if won is not None and i < len(won):
                rec["won"] = float(won[i])
            f.write(json.dumps(rec) + "\n")
