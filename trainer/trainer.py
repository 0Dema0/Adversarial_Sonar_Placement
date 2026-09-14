import math
import torch

from trainer.loss import CVaRLoss, AnnealedLoss, AnnealedExtremeMSELoss, AnnealedRandomMSELoss, MiddleMSELoss

class Trainer:
    def __init__(
        self,
        model,
        optimizer=None,
        loss=None,
        device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0

        self.loss.set_epoch(epoch)

        for local, global_context, target in loader:
            local = local.to(self.device)
            global_context = global_context.to(self.device)
            target = target.to(self.device)

            prediction = self.model(local, global_context)

            loss_map = (prediction - target) ** 2

            # (B,)
            per_sample = loss_map.mean(dim=tuple(range(1, loss_map.ndim)))

            loss = self.loss(per_sample, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader, epoch, return_stats=False):
        self.model.eval()
        total_loss = 0.0
        per_sample_all = []
        target_all = []

        self.loss.set_epoch(epoch)

        for local, global_context, target in loader:
            local = local.to(self.device)
            global_context = global_context.to(self.device)
            target = target.to(self.device)

            prediction = self.model(local, global_context)

            loss_map = (prediction - target) ** 2

            per_sample = loss_map.mean(dim=tuple(range(1, loss_map.ndim)))
            per_sample_all.append(per_sample.detach().cpu())
            target_all.append(target.detach().cpu())

            loss = self.loss(per_sample, target)

            total_loss += loss.item()

        val_obj = total_loss / len(loader)

        if not per_sample_all:
            stats = {
                "val_obj": val_obj,
                "val_mse": 0.0,
                "val_extreme": None,
                "val_random": None,
                "val_middle": None,
                "val_cvar": None,
                "val_select": val_obj,
                "lambda": None,
                "alpha": None,
                "rho": None,
            }
            return stats if return_stats else float(stats["val_select"])

        all_val = torch.cat(per_sample_all)
        all_target = torch.cat(target_all)
        val_mse = float(all_val.mean().item())

        # EXTREME ANNEALING
        if isinstance(self.loss, AnnealedExtremeMSELoss):

            rho = float(self.loss.current_rho)

            k_val = max(
                1,
                math.ceil(rho * all_val.numel()),
            )

            extreme_indices = torch.topk(
                all_target.flatten(),
                k_val,
                largest=True,
            ).indices

            val_extreme = float(
                all_val.flatten()[extreme_indices]
                .mean()
                .item()
            )

            val_select = val_extreme

            stats = {
                "val_obj": val_obj,
                "val_mse": val_mse,
                "val_extreme": val_extreme,
                "val_random": None,
                "val_middle": None,
                "val_cvar": None,
                "val_select": val_select,
                "lambda": None,
                "alpha": None,
                "rho": rho,
            }

            return (
                stats
                if return_stats
                else float(stats["val_select"])
            )

        # Use empirical top-k CVaR for model selection (same principle as trainer_single_CVaR).
        if isinstance(self.loss, AnnealedLoss):
            alpha = float(self.loss.cvar_loss.alpha)
            k_val = max(1, int(math.ceil(alpha * all_val.numel())))
            val_cvar = float(torch.topk(all_val, k_val).values.mean().item())
            if epoch <= int(self.loss.warmup_epochs):
                val_select = val_mse
            else:
                lam = float(self.loss.current_lambda)
                val_select = lam * val_mse + (1.0 - lam) * val_cvar
            stats = {
                "val_obj": val_obj,
                "val_mse": val_mse,
                "val_extreme": None,
                "val_random": None,
                "val_cvar": val_cvar,
                "val_select": float(val_select),
                "lambda": float(self.loss.current_lambda),
                "alpha": alpha,
                "rho": None,                
            }
            return stats if return_stats else float(stats["val_select"])

        # RANDOM ANNEALING
        if isinstance(self.loss, AnnealedRandomMSELoss):

            rho = float(self.loss.current_rho)

            k_val = max(
                1,
                math.ceil(rho * all_val.numel()),
            )

            random_indices = torch.randperm(
                all_val.numel(),
                device=all_val.device,
            )[:k_val]

            val_random = float(
                all_val.flatten()[random_indices]
                .mean()
                .item()
            )

            val_select = val_random

            stats = {
                "val_obj": val_obj,
                "val_mse": val_mse,
                "val_extreme": None,
                "val_random": val_random,
                "val_middle": None,
                "val_cvar": None,
                "val_select": val_select,
                "lambda": None,
                "alpha": None,
                "rho": rho,
            }

            return (
                stats
                if return_stats
                else float(stats["val_select"])
            )

        # MIDDLE-MSE ANNEALING
        if isinstance(self.loss, MiddleMSELoss):

            rho = float(self.loss.current_rho)

            k_val = max(
                1,
                math.ceil(rho * all_val.numel()),
            )

            y_flat = all_target.flatten()
            loss_flat = all_val.flatten()

            # Distance from median target
            median = torch.median(y_flat)

            distance = torch.abs(y_flat - median)

            # Deterministically select samples closest to the median
            middle_indices = torch.topk(
                distance,
                k=k_val,
                largest=False,
            ).indices

            val_middle = float(
                loss_flat[middle_indices]
                .mean()
                .item()
            )

            val_select = val_middle

            stats = {
                "val_obj": val_obj,
                "val_mse": val_mse,
                "val_extreme": None,
                "val_random": None,
                "val_middle": val_middle,
                "val_cvar": None,
                "val_select": val_select,
                "lambda": None,
                "alpha": None,
                "rho": rho,
            }

            return (
                stats
                if return_stats
                else float(stats["val_select"])
            )

        if isinstance(self.loss, CVaRLoss):
            alpha = float(self.loss.alpha)
            k_val = max(1, int(math.ceil(alpha * all_val.numel())))
            val_cvar = float(torch.topk(all_val, k_val).values.mean().item())
            stats = {
                "val_obj": val_obj,
                "val_mse": val_mse,
                "val_extreme": None,
                "val_random": None,
                "val_middle": None,
                "val_cvar": val_cvar,
                "val_select": val_cvar,
                "lambda": None,
                "alpha": alpha,
                "rho": None,
            }
            return stats if return_stats else float(stats["val_select"])

        stats = {
            "val_obj": val_obj,
            "val_mse": val_mse,
            "val_extreme": None,
            "val_random": None,
            "val_middle": None,
            "val_cvar": None,
            "val_select": val_mse,
            "lambda": None,
            "alpha": None,
            "rho": None,
        }
        return stats if return_stats else float(stats["val_select"])

    def fit(self, train_loader, val_loader, epochs, save_path="best_model.pt", last_save_path=None):
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_stats = self.validate(val_loader, epoch, return_stats=True)
            val_loss = float(val_stats["val_select"])

            msg = (
                f"{epoch+1:3d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"val_mse={val_stats['val_mse']:.6f}"
            )
            if val_stats["val_extreme"] is not None:
                msg += f" val_extreme={val_stats['val_extreme']:.6f}"
            if val_stats["val_random"] is not None:
                msg += f" val_random={val_stats['val_random']:.6f}"
            if val_stats["val_middle"] is not None:
                msg += f" val_middle={val_stats['val_middle']:.6f}"
            if val_stats["val_cvar"] is not None:
                msg += f" val_cvar={val_stats['val_cvar']:.6f}"
            if val_stats["lambda"] is not None:
                msg += f" lambda={val_stats['lambda']:.4f}"
            if val_stats["alpha"] is not None:
                msg += f" alpha={val_stats['alpha']:.4f}"
            if val_stats["rho"] is not None:
                msg += f" rho={val_stats['rho']:.4f}"
            print(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

        if last_save_path is not None:
            torch.save(self.model.state_dict(), last_save_path)
            print(f"Saved last model to {last_save_path}")