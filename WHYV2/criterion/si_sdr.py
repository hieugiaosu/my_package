import itertools

import torch
from torch.nn.modules.loss import _Loss


class SingleSrcNegSDRScaledSrc(_Loss):
    r"""Base class for single-source negative SI-SDR, SD-SDR and SNR.
        (Scaled source)
    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target and
            estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, zero_mean=True, take_log=True, reduction="none", EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_target, target):
        if target.size() != est_target.size() or target.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
            # [batch, 1]
        dot = torch.sum(est_target * target, dim=1, keepdim=True)
        # [batch, 1]
        s_target_energy = torch.sum(target**2, dim=1, keepdim=True) + self.EPS
        # [batch, time]
        scaled_target = dot * target / s_target_energy

        e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target**2, dim=1) / (torch.sum(e_noise**2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses
   
class SingleSrcNegSDRScaledEst(_Loss):
    def __init__(self, zero_mean=True, take_log=True, reduction="none", EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_target, target):

        if target.size() != est_target.size() or target.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
            # [batch, 1]
        dot = torch.sum(est_target*target, dim = 1, keepdim = True)
        # [batch, 1]
        s_est_energy = torch.sum(est_target**2, dim=1, keepdim=True) + self.EPS
        # [batch, time]
        scaled_est =  dot* est_target / s_est_energy

        e_noise = target - scaled_est
        # [batch]
        losses = torch.sum(target**2, dim=1) / (torch.sum(e_noise**2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses
    
class SISDRLoss(_Loss):
    def __init__(self, zero_mean=True, scale_source = False, take_log=True, reduction="none", EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        if scale_source:
            self.loss = SingleSrcNegSDRScaledSrc(zero_mean, take_log, reduction, EPS)
        else:
            self.loss = SingleSrcNegSDRScaledEst(zero_mean, take_log, reduction, EPS)
    
    def forward(self, est_target, target):
        return self.loss(est_target, target)

class SISDRLossWithPIT(_Loss):
    def __init__(self, zero_mean=True, scale_source = False, take_log=True, reduction="none", EPS=1e-8, n_src=2):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)
        self.n_src = n_src
        if scale_source:
            self.loss = SingleSrcNegSDRScaledSrc(zero_mean, take_log, reduction, EPS)
        else:
            self.loss = SingleSrcNegSDRScaledEst(zero_mean, take_log, reduction, EPS)
    
    def forward(self, est_sources, src_sources):
        """
        Compute SI-SDR with PIT.
        Args:
            est_sources (torch.Tensor): Estimated sources. Shape: (batch, n_src, time)
            src_sources (torch.Tensor): Reference sources. Shape: (batch, n_src, time)
        """
        batch_size, n_src, time = est_sources.size()
        assert n_src == self.n_src, "Number of estimated sources must match the configured n_src."
        assert src_sources.size() == est_sources.size(), "Shape mismatch between estimated and reference sources."

        # Generate all possible permutations
        permutations = list(itertools.permutations(range(n_src)))
        loss_matrix = torch.zeros(batch_size, len(permutations), device=est_sources.device)

        # Compute loss for each permutation
        for perm_idx, perm in enumerate(permutations):
            permuted_src = src_sources[:, perm, :]  # Permute the sources
            loss_per_perm = torch.stack(
                [self.loss(est_sources[:, i, :], permuted_src[:, i, :]) for i in range(n_src)],
                dim=-1,
            )  # (batch, n_src)
            loss_matrix[:, perm_idx] = loss_per_perm.sum(dim=-1)

        # Select the permutation with the minimum loss
        min_loss, _ = loss_matrix.min(dim=-1)  # (batch,)

        # Apply reduction
        if self.reduction == "mean":
            return min_loss.mean()
        elif self.reduction == "none":
            return min_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")