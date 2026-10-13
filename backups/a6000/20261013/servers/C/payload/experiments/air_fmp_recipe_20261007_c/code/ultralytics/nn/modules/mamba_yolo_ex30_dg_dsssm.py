import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_yolo import PFIM, USE_CUDA_SCAN, selective_scan_fn


class DGDeSpeckleSSM(nn.Module):
    """
    Decoupled-gated DS-SSM.

    This keeps the ex27 DS-SSM/PFIM backbone intact and only separates the
    retain/erase gate, write gate, and delta gate for controlled SSDD ablations.
    """

    def __init__(
        self,
        dim,
        state_dim=16,
        sigma_n=0.1,
        lambda_init=0.1,
        alpha_init=0.5,
        kernel_size=3,
        mode="erase_write",
        use_pfim=True,
        beta_init=0.1,
        rho_init=0.1,
        eta_init=0.02,
        min_retain=0.9,
        write_scale=0.1,
        write_min=0.75,
        write_max=1.25,
        pfim_hidden_dim=None,
        pfim_use_2d_std=True,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.sigma_n = sigma_n
        self.eps = 1e-5
        self.mode = str(mode)
        self.use_pfim = use_pfim
        self.min_retain = float(min_retain)
        self.write_scale = float(write_scale)
        self.write_min = float(write_min)
        self.write_max = float(write_max)
        self.output_clip = 20.0

        self.use_erase_gate = self.mode in {"erase", "erase_write", "erase_write_delta", "full"}
        self.use_write_gate = self.mode in {"write", "erase_write", "erase_write_delta", "full"}
        self.use_delta_gate = self.mode in {"delta", "erase_write_delta", "full"}

        self.lambda_s = nn.Parameter(torch.tensor(float(lambda_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta_e = nn.Parameter(torch.tensor(float(beta_init)))
        self.rho_w = nn.Parameter(torch.tensor(float(rho_init)))
        self.eta_d = nn.Parameter(torch.tensor(float(eta_init)))

        pad = kernel_size // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=False)
        nn.init.constant_(self.dw.weight, 1.0 / kernel_size)

        self.pfim = (
            PFIM(dim=dim, kernel_size=kernel_size, hidden_dim=pfim_hidden_dim, eps=self.eps, use_2d_std=pfim_use_2d_std)
            if self.use_pfim
            else None
        )

        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.delta_proj = nn.Linear(dim, dim, bias=True)
        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        self.C_proj = nn.Linear(dim, state_dim, bias=False)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, hw_shape=None, return_importance=False):
        input_dtype = x.dtype
        bsz, n_tokens, channels = x.shape

        x_bar = self.dw(x.transpose(1, 2)).transpose(1, 2)
        residual = x - x_bar
        lambda_eff = torch.clamp(self.lambda_s.float(), min=0.0, max=0.3).to(dtype=input_dtype)
        x_tilde = x - lambda_eff * residual
        x_tilde = torch.nan_to_num(x_tilde, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)

        if self.use_pfim:
            importance = self.pfim(x_tilde, x_bar, hw_shape=hw_shape)
        else:
            importance = torch.full((bsz, n_tokens, 1), 0.5, device=x.device, dtype=input_dtype)
        imp_center = importance - 0.5

        u, z = self.in_proj(x_tilde).chunk(2, dim=-1)

        delta_logits = self.delta_proj(u)
        if self.use_delta_gate:
            eta_eff = torch.clamp(self.eta_d.float(), min=0.0, max=0.05).to(dtype=input_dtype)
            delta_logits = delta_logits + eta_eff * importance
        delta_logits = torch.clamp(delta_logits, min=-8.0, max=8.0)
        delta = torch.clamp(F.softplus(delta_logits), min=1e-4, max=8.0)

        b_t = self.B_proj(u)
        c_t = self.C_proj(u)
        A = -torch.exp(self.A_log.float())

        energy = torch.norm(x_bar.float(), p=2, dim=-1, keepdim=True) / (channels ** 0.5)

        alpha_eff = torch.clamp(self.alpha.float(), min=0.0, max=1.0)
        beta_eff = torch.clamp(self.beta_e.float(), min=0.0, max=0.2)
        rho_eff = torch.clamp(self.rho_w.float(), min=0.0, max=0.2).to(dtype=input_dtype)

        erase_logits = -alpha_eff * self.sigma_n * energy
        if self.use_erase_gate:
            erase_logits = erase_logits + beta_eff * imp_center.float()
        retain_gate = self.min_retain + (1.0 - self.min_retain) * torch.sigmoid(erase_logits)
        retain_gate = torch.clamp(retain_gate, min=self.min_retain, max=1.0)
        retain_gate = retain_gate.to(dtype=input_dtype)

        if self.use_write_gate:
            write_logits = rho_eff * imp_center
            write_gate = 1.0 + self.write_scale * torch.tanh(write_logits)
            write_gate = torch.clamp(write_gate, min=self.write_min, max=self.write_max)
        else:
            write_gate = 1.0

        u_write = u * write_gate

        if USE_CUDA_SCAN and selective_scan_fn is not None:
            with torch.cuda.amp.autocast(enabled=False):
                u_fp32 = u_write.float()
                delta_fp32 = delta.float()
                b_fp32 = b_t.float()
                c_fp32 = c_t.float()
                retain_fp32 = retain_gate.float()
                x_tilde_fp32 = x_tilde.float()
                z_fp32 = z.float()
                D_fp32 = self.D.float()

                u_damped = torch.nan_to_num(u_fp32 * retain_fp32, nan=0.0, posinf=1e4, neginf=-1e4)
                y_scan = selective_scan_fn(
                    u_damped.transpose(1, 2).contiguous(),
                    delta_fp32.transpose(1, 2).contiguous(),
                    A,
                    b_fp32.transpose(1, 2).contiguous(),
                    c_fp32.transpose(1, 2).contiguous(),
                    D=None,
                    z=None,
                    delta_bias=None,
                    delta_softplus=False,
                    return_last_state=False,
                )
                y = y_scan.transpose(1, 2).contiguous()
                y = torch.nan_to_num(y, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)
                y = torch.clamp(y, min=-self.output_clip, max=self.output_clip)
                y = (y + D_fp32 * x_tilde_fp32) * F.silu(z_fp32)
                y = torch.nan_to_num(y, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)
                y = torch.clamp(y, min=-self.output_clip, max=self.output_clip)
            y = y.to(dtype=input_dtype)
        else:
            A = A.to(dtype=input_dtype)
            h = x.new_zeros(bsz, channels, self.state_dim)
            ys = []
            for t in range(n_tokens):
                dt = delta[:, t, :].unsqueeze(-1)
                At = torch.exp(dt * A.unsqueeze(0))
                Bt = dt * b_t[:, t, :].unsqueeze(1)
                Ct = c_t[:, t, :].unsqueeze(1)
                h = retain_gate[:, t, :].unsqueeze(-1) * At * h + Bt * u_write[:, t, :].unsqueeze(-1)
                h = torch.nan_to_num(h, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)
                h = torch.clamp(h, min=-self.output_clip, max=self.output_clip)
                yt = (Ct * h).sum(dim=-1) + self.D.to(dtype=input_dtype) * x_tilde[:, t, :]
                ys.append(yt)
            y = torch.stack(ys, dim=1) * F.silu(z)
            y = torch.nan_to_num(y, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)
            y = torch.clamp(y, min=-self.output_clip, max=self.output_clip)

        y = self.out_proj(y)
        y = torch.nan_to_num(y, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)
        y = torch.clamp(y, min=-self.output_clip, max=self.output_clip)
        if return_importance:
            return y, importance
        return y


class DG_DS_VSSBlock(nn.Module):
    """DG-DS-SSM block used for ex30-ex34 SSDD experiments."""

    def __init__(
        self,
        in_channels,
        hidden_dim,
        mode="erase_write",
        state_dim=16,
        mlp_ratio=2.0,
        sigma_n=0.1,
        lambda_init=0.1,
        alpha_init=0.5,
        kernel_size=3,
        drop=0.0,
        beta_init=0.1,
        rho_init=0.1,
        eta_init=0.02,
        min_retain=0.9,
        write_scale=0.1,
        pfim_hidden_dim=None,
        pfim_use_2d_std=True,
    ):
        super().__init__()
        if in_channels != hidden_dim:
            raise ValueError(f"DG_DS_VSSBlock expects c1 == c2, got c1={in_channels}, c2={hidden_dim}")
        self.dim = hidden_dim
        self.mode = str(mode)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ds_ssm = DGDeSpeckleSSM(
            dim=hidden_dim,
            state_dim=state_dim,
            sigma_n=sigma_n,
            lambda_init=lambda_init,
            alpha_init=alpha_init,
            kernel_size=kernel_size,
            mode=mode,
            beta_init=beta_init,
            rho_init=rho_init,
            eta_init=eta_init,
            min_retain=min_retain,
            write_scale=write_scale,
            pfim_hidden_dim=pfim_hidden_dim,
            pfim_use_2d_std=pfim_use_2d_std,
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(drop),
        )
        self.gamma1 = nn.Parameter(torch.ones(hidden_dim) * 1e-2)
        self.gamma2 = nn.Parameter(torch.ones(hidden_dim) * 1e-2)

    def forward(self, x):
        bsz, channels, height, width = x.shape
        if channels != self.dim:
            raise AssertionError(f"DG_DS_VSSBlock dim mismatch: input C={channels}, module dim={self.dim}")

        x_tokens = x.flatten(2).transpose(1, 2).contiguous()
        x_tokens = x_tokens + self.gamma1.to(dtype=x_tokens.dtype) * self.ds_ssm(
            self.norm1(x_tokens),
            hw_shape=(height, width),
        )
        x_tokens = x_tokens + self.gamma2.to(dtype=x_tokens.dtype) * self.mlp(self.norm2(x_tokens))
        return x_tokens.transpose(1, 2).reshape(bsz, channels, height, width).contiguous()
