from __future__ import annotations

import torch
import torch.nn as nn
import math

class MSELoss(nn.Module):
    def forward(self, per_sample_loss, y_true=None):
        return per_sample_loss.mean()
    
    def set_epoch(self, epoch):
        pass

class CVaRLoss(nn.Module):
    def __init__(self, alpha=0.1, learnable_t=True):
        super().__init__()
        self.alpha = alpha
        self.t = nn.Parameter(torch.tensor(0.0)) if learnable_t else None

    def forward(self, per_sample_loss, y_true=None):
        B = per_sample_loss.numel()
        if B == 0:
            return torch.tensor(0.0, device=per_sample_loss.device, dtype=per_sample_loss.dtype,)

        denom = max(float(self.alpha) * float(B), 1e-12)
        return self.t + torch.relu(per_sample_loss - self.t).sum() / denom
    
    def set_epoch(self, epoch):
        pass

class AnnealedLoss(nn.Module):
    def __init__(
        self,
        mse_loss = MSELoss(),
        cvar_loss = CVaRLoss(),
        lambda_start=1.0, # weight of MSE at the start of training
        lambda_final=0.0, # weight of MSE at the end of training
        alpha_start=0.2, # weight of CVaR at the start of training
        alpha_final=0.0005, # weight of CVaR at the end of training
        warmup_epochs=10, # number of epochs to train with only MSE before annealing
        anneal_epochs=80, # number of epochs to anneal from lambda_start to lambda_final and alpha_start to alpha_final
    ):
        super().__init__()

        self.mse_loss = mse_loss
        self.cvar_loss = cvar_loss

        self.lambda_start = lambda_start
        self.lambda_final = lambda_final

        self.alpha_start = alpha_start
        self.alpha_final = alpha_final

        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

        self.current_lambda = lambda_start
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch <= self.warmup_epochs:
            self.current_lambda = 1.0
            self.cvar_loss.alpha = self.alpha_start
            return

        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.anneal_epochs))

        self.current_lambda = (self.lambda_start * (1.0 - progress) + self.lambda_final * progress)

        self.cvar_loss.alpha = (self.alpha_start * (1.0 - progress) + self.alpha_final * progress)

    def forward(self, per_sample_loss, y_true=None):
        mse = self.mse_loss(per_sample_loss)
        cvar = self.cvar_loss(per_sample_loss)

        if self.current_epoch <= self.warmup_epochs:
            return mse
        return self.current_lambda * mse + (1.0 - self.current_lambda) * cvar

class ExtremeMSELoss(nn.Module):
    """
    MSE evaluated only on the upper tail of y_true.

    rho = fraction of samples to retain.
        rho=1.0 -> all samples -> ordinary MSE
        rho=0.2 -> top 20% of y_true
        rho=0.01 -> top 1% of y_true

    The threshold is computed from y_true, so the selection is
    independent of the model's predictions.
    """

    def __init__(self, rho=0.1):
        super().__init__()
        self.rho = rho

    def forward(self, per_sample_loss, y_true):
        B = y_true.numel()

        if B == 0:
            return torch.tensor(
                0.0,
                device=per_sample_loss.device,
                dtype=per_sample_loss.dtype,
            )

        # rho = fraction of samples to keep
        if self.rho >= 1.0:
            return per_sample_loss.mean()

        if self.rho <= 0.0:
            raise ValueError("rho must be > 0")

        # Quantile corresponding to upper rho fraction.
        q = torch.quantile(
            y_true.detach().flatten(),
            1.0 - self.rho,
        )

        mask = y_true > q

        # Quantile can produce ties, potentially giving no samples.
        # Fall back to top-k in that case.
        if mask.sum() == 0:
            k = max(1, int(torch.ceil(
                torch.tensor(self.rho * B)
            ).item()))

            _, indices = torch.topk(
                y_true.flatten(),
                k=k,
            )

            mask = torch.zeros(
                B,
                dtype=torch.bool,
                device=y_true.device,
            )
            mask[indices] = True

        return per_sample_loss.flatten()[mask].mean()

    def set_epoch(self, epoch):
        pass

class AnnealedExtremeMSELoss(nn.Module):
    """
    Anneals from ordinary MSE to MSE on the extreme upper tail of y_true.

    rho_start = fraction of samples used at the beginning.
    rho_final = fraction of samples used at the end.

    Example:
        rho_start = 1.0   -> use 100% of samples
        rho_final = 0.01  -> use top 1% of samples

    The annealing is logarithmic, which is more appropriate when
    moving across several orders of magnitude in the tail fraction.
    """

    def __init__(
        self,
        rho_start=1.0,
        rho_final=0.05,
        warmup_epochs=10,
        anneal_epochs=80,
    ):
        super().__init__()

        if not (0.0 < rho_final <= rho_start <= 1.0):
            raise ValueError(
                "Require 0 < rho_final <= rho_start <= 1."
            )

        self.rho_start = rho_start
        self.rho_final = rho_final

        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

        self.current_rho = rho_start
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

        # Warmup: ordinary MSE
        if epoch <= self.warmup_epochs:
            self.current_rho = self.rho_start
            return

        # Progress through annealing
        progress = min(
            1.0,
            (epoch - self.warmup_epochs)
            / max(1, self.anneal_epochs),
        )

        # Logarithmic interpolation:
        # rho_start -> rho_final
        log_start = math.log(self.rho_start)
        log_final = math.log(self.rho_final)

        self.current_rho = math.exp(
            log_start
            + progress * (log_final - log_start)
        )

    def forward(self, per_sample_loss, y_true):
        """
        Args:
            per_sample_loss:
                Tensor of shape [B], containing squared errors.

            y_true:
                Tensor of shape [B], containing target values.

        Returns:
            Scalar loss.
        """

        B = y_true.numel()

        if B == 0:
            return per_sample_loss.sum() * 0.0

        # rho = 1 -> ordinary MSE
        if self.current_rho >= 1.0:
            return per_sample_loss.mean()

        # Number of extreme samples to keep
        k = max(
            1,
            math.ceil(self.current_rho * B),
        )

        y_flat = y_true.flatten()
        loss_flat = per_sample_loss.flatten()

        # Select samples with largest y_true
        _, indices = torch.topk(
            y_flat,
            k=k,
            largest=True,
        )

        # MSE only on the extreme samples
        return loss_flat[indices].mean()


class AnnealedSymmetricTailMSELoss(nn.Module):
    def __init__(
        self,
        rho_start=1.0,
        rho_final=0.05,
        warmup_epochs=10,
        anneal_epochs=80,
    ):
        super().__init__()

        if not (0.0 < rho_final <= rho_start <= 1.0):
            raise ValueError(
                "Require 0 < rho_final <= rho_start <= 1."
            )

        self.rho_start = rho_start
        self.rho_final = rho_final

        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

        self.current_rho = rho_start
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

        if epoch <= self.warmup_epochs:
            self.current_rho = self.rho_start
            return

        progress = min(
            1.0,
            (epoch - self.warmup_epochs)
            / max(1, self.anneal_epochs),
        )

        log_start = math.log(self.rho_start)
        log_final = math.log(self.rho_final)

        self.current_rho = math.exp(
            log_start
            + progress * (log_final - log_start)
        )

    @staticmethod
    def tail_key(per_sample_loss, y_true):
        """y + |error|, per sample. Works for [B] or [B, T] targets."""
        y = y_true.reshape(per_sample_loss.shape[0], -1).mean(dim=1)
        return (y + per_sample_loss.detach().clamp_min(0.0).sqrt()).flatten()

    def forward(self, per_sample_loss, y_true):
        """
        Args:
            per_sample_loss: Tensor [B], squared errors.
            y_true:          Tensor [B] (or [B, T]), targets.
        """
        B = per_sample_loss.numel()

        if B == 0:
            return per_sample_loss.sum() * 0.0

        if self.current_rho >= 1.0:
            return per_sample_loss.mean()

        k = max(1, math.ceil(self.current_rho * B))

        _, indices = torch.topk(
            self.tail_key(per_sample_loss, y_true),
            k=k,
            largest=True,
        )

        return per_sample_loss.flatten()[indices].mean()

class AnnealedTopPredTopTrueMSELoss(nn.Module):
    def __init__(
        self,
        rho_start=1.0,
        rho_final=0.05,
        warmup_epochs=10,
        anneal_epochs=80,
        pred_weight=0.5,
    ):
        super().__init__()

        if not (0.0 < rho_final <= rho_start <= 1.0):
            raise ValueError("Require 0 < rho_final <= rho_start <= 1.")
        if not (0.0 <= pred_weight <= 1.0):
            raise ValueError("Require 0 <= pred_weight <= 1.")

        self.rho_start = rho_start
        self.rho_final = rho_final
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.pred_weight = pred_weight
        self.needs_prediction = True


        self.current_rho = rho_start
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

        if epoch <= self.warmup_epochs:
            self.current_rho = self.rho_start
            return

        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.anneal_epochs))
        log_start = math.log(self.rho_start)
        log_final = math.log(self.rho_final)
        self.current_rho = math.exp(log_start + progress * (log_final - log_start))

    def forward(self, per_sample_loss, y_true, prediction):
        """
        Args:
            per_sample_loss: Tensor [B], squared errors.
            y_true:          Tensor [B] (or [B, T]), targets.
            prediction:      Tensor [B] (or [B, T]), model outputs.
        """
        B = per_sample_loss.numel()

        if B == 0:
            return per_sample_loss.sum() * 0.0

        if self.current_rho >= 1.0:
            return per_sample_loss.mean()

        k = max(1, math.ceil(self.current_rho * B))
        losses = per_sample_loss.flatten()

        y = y_true.reshape(B, -1).mean(dim=1)
        p = prediction.detach().reshape(B, -1).mean(dim=1)

        top_true = torch.topk(y, k=k, largest=True).indices
        top_pred = torch.topk(p, k=k, largest=True).indices

        return (
            self.pred_weight * losses[top_pred].mean()
            + (1.0 - self.pred_weight) * losses[top_true].mean()
        )

class AnnealedRandomMSELoss(nn.Module):
    """
    Anneals from ordinary MSE to MSE on a random subset of y_true.

    rho_start = fraction of samples used at the beginning.
    rho_final = fraction of samples used at the end.

    Example:
        rho_start = 1.0   -> use 100% of samples
        rho_final = 0.01  -> use random 1% of samples

    The annealing is logarithmic.s
    """

    def __init__(
        self,
        rho_start=1.0,
        rho_final=0.01,
        warmup_epochs=10,
        anneal_epochs=80,
    ):
        super().__init__()

        if not (0.0 < rho_final <= rho_start <= 1.0):
            raise ValueError(
                "Require 0 < rho_final <= rho_start <= 1."
            )

        self.rho_start = rho_start
        self.rho_final = rho_final

        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

        self.current_rho = rho_start
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

        # Warmup: ordinary MSE
        if epoch <= self.warmup_epochs:
            self.current_rho = self.rho_start
            return

        # Progress through annealing
        progress = min(
            1.0,
            (epoch - self.warmup_epochs)
            / max(1, self.anneal_epochs),
        )

        # Logarithmic interpolation:
        # rho_start -> rho_final
        log_start = math.log(self.rho_start)
        log_final = math.log(self.rho_final)

        self.current_rho = math.exp(
            log_start
            + progress * (log_final - log_start)
        )

    def forward(self, per_sample_loss, y_true):
        """
        Args:
            per_sample_loss:
                Tensor of shape [B], containing squared errors.

            y_true:
                Tensor of shape [B], containing target values.

        Returns:
            Scalar loss.
        """

        B = y_true.numel()

        if B == 0:
            return per_sample_loss.sum() * 0.0

        # rho = 1 -> ordinary MSE
        if self.current_rho >= 1.0:
            return per_sample_loss.mean()

        # Number of extreme samples to keep
        k = max(
            1,
            math.ceil(self.current_rho * B),
        )

        indices = torch.randperm(B, device=per_sample_loss.device)[:k]

        return per_sample_loss[indices].mean()

class MiddleMSELoss(nn.Module):
    """
    MSE on a fixed fraction of samples from the middle of the
    target distribution.

    rho = fraction of samples retained.

    The selected samples are centered around the median of y_true.
    """

    def __init__(self, rho=0.01):
        super().__init__()

        if not (0.0 < rho <= 1.0):
            raise ValueError("rho must satisfy 0 < rho <= 1.")

        self.rho = rho

    def forward(self, per_sample_loss, y_true):

        B = y_true.numel()

        if B == 0:
            return per_sample_loss.sum() * 0.0

        if self.rho >= 1.0:
            return per_sample_loss.mean()

        k = max(
            1,
            math.ceil(self.rho * B),
        )

        y_flat = y_true.flatten()
        loss_flat = per_sample_loss.flatten()

        # Distance from median
        median = torch.median(y_flat)

        distance = torch.abs(y_flat - median)

        # Select samples closest to the median
        indices = torch.topk(
            distance,
            k=k,
            largest=False,
        ).indices

        return loss_flat[indices].mean()

    def set_epoch(self, epoch):
        pass