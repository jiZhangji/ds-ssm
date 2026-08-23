from .common_utils_mbyolo import *

__all__ = (
    "VSSBlock",
    "SimpleStem",
    "VisionClueMerge",
    "XSSBlock",
    "PFIM",
    "DeSpeckleSSM",
    "DS_VSSBlock",
    "DS_VSSBlockSweep",
    "DS_VSSBlockSweepNoPFIM",
    "DS_VSSBlockSweepPFIMOnly",
    "DS_VSSBlockSweepPFIMGlobal",
    "DS_VSSBlockNoPFIM",
    "DS_VSSBlockPFIMOnly",
    "DS_VSSBlockPFIMGamma",
    "DS_VSSBlockPFIMWrite",
    "DS_VSSBlockPFIMDelta",
    "DS_VSSBlockPFIMGammaWrite",
    "DS_VSSBlockPFIMGlobal",
    "DS_VSSBlockPFIMNo2DStd",
)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    USE_CUDA_SCAN = True
except Exception:
    # GTX 1080 Ti / sm_61 fallback:
    # mamba_ssm is unavailable, but local selective_scan_cuda_core has been compiled.
    # SelectiveScanCore comes from common_utils_mbyolo via: from .common_utils_mbyolo import *
    def selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
        nrows=1,
        backnrows=1,
        **kwargs,
    ):
        # local selective_scan_cuda_core expects variable B/C as [B, G, S, L].
        # Your DS-SSM creates B/C as [B, S, L], so add group dim.
        if B is not None and B.dim() == 3:
            B = B.unsqueeze(1)
        if C is not None and C.dim() == 3:
            C = C.unsqueeze(1)

        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous() if B is not None else None
        C = C.contiguous() if C is not None else None
        D = D.contiguous() if D is not None else None
        delta_bias = delta_bias.contiguous() if delta_bias is not None else None

        out = SelectiveScanCore.apply(
            u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows
        )

        if z is not None:
            out = out * F.silu(z)

        if return_last_state:
            return out, None
        return out

    USE_CUDA_SCAN = True
    print(
        "Info: mamba_ssm selective_scan_fn is not available. "
        "DS-SSM uses local selective_scan_cuda_core fast path."
    )

## 注释参考上一版本
class PFIM(nn.Module):
    """
    Perceptual Feature Importance Module.

    Estimate token-level importance:
        I_t in [0, 1], shape [B, N, 1]
    """

    def __init__(self, dim, kernel_size=3, hidden_dim=None, eps=1e-5, use_2d_std=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.eps = eps
        self.use_2d_std = use_2d_std

        if hidden_dim is None:
            hidden_dim = max(dim // 4, 16)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize to I_t = 0.5
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_tilde, x_bar, hw_shape=None):
        input_dtype = x_tilde.dtype
        B, N, C = x_tilde.shape

        if self.use_2d_std and hw_shape is not None:
            H, W = hw_shape

            xt = x_tilde.float().transpose(1, 2).reshape(B, C, H, W)

            local_mean = F.avg_pool2d(
                xt,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                count_include_pad=False,
            )

            local_mean2 = F.avg_pool2d(
                xt.pow(2),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                count_include_pad=False,
            )

            local_std = torch.sqrt(
                torch.clamp(local_mean2 - local_mean.pow(2), min=0.0) + self.eps
            )

            local_std = local_std.flatten(2).transpose(1, 2).contiguous()

        else:
            xt = x_tilde.float().transpose(1, 2)

            local_mean = F.avg_pool1d(
                xt,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                count_include_pad=False,
            ).transpose(1, 2)

            local_mean2 = F.avg_pool1d(
                xt.pow(2),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                count_include_pad=False,
            ).transpose(1, 2)

            local_std = torch.sqrt(
                torch.clamp(local_mean2 - local_mean.pow(2), min=0.0) + self.eps
            )

        imp_feat = torch.cat([x_bar.float(), local_std], dim=-1)

        mlp_dtype = next(self.mlp.parameters()).dtype
        imp_feat = imp_feat.to(dtype=mlp_dtype)

        importance = torch.sigmoid(self.mlp(imp_feat))
        return importance.to(dtype=input_dtype)

class DeSpeckleSSM(nn.Module):
    """
    Importance-Guided De-Speckle Selective State Space Model.
    """

    def __init__(
        self,
        dim,
        state_dim=16,
        sigma_n=0.1,
        lambda_init=0.1,
        alpha_init=0.5,
        kernel_size=3,

        use_pfim=True,
        pfim_on_gamma=True,
        pfim_on_write=True,
        pfim_on_delta=True,

        beta_init=1.0,
        rho_init=1.0,
        eta_init=0.1,
        min_retain=0.5,
        pfim_hidden_dim=None,
        pfim_use_2d_std=True,
        pfim_tokenwise=True,
    ):
        super().__init__()

        self.dim = dim
        self.state_dim = state_dim
        self.sigma_n = sigma_n
        self.eps = 1e-5

        self.use_pfim = use_pfim
        self.pfim_on_gamma = pfim_on_gamma
        self.pfim_on_write = pfim_on_write
        self.pfim_on_delta = pfim_on_delta
        self.pfim_tokenwise = pfim_tokenwise
        self.min_retain = float(min_retain)

        self.lambda_s = nn.Parameter(torch.tensor(float(lambda_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.beta_i = nn.Parameter(torch.tensor(float(beta_init)))
        self.rho_i = nn.Parameter(torch.tensor(float(rho_init)))
        self.eta_i = nn.Parameter(torch.tensor(float(eta_init)))

        if not (use_pfim and pfim_on_gamma):
            self.beta_i.requires_grad_(False)
        if not (use_pfim and pfim_on_write):
            self.rho_i.requires_grad_(False)
        if not (use_pfim and pfim_on_delta):
            self.eta_i.requires_grad_(False)

        pad = kernel_size // 2
        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=dim,
            bias=False,
        )
        nn.init.constant_(self.dw.weight, 1.0 / kernel_size)

        if self.use_pfim:
            self.pfim = PFIM(
                dim=dim,
                kernel_size=kernel_size,
                hidden_dim=pfim_hidden_dim,
                eps=self.eps,
                use_2d_std=pfim_use_2d_std,
            )
        else:
            self.pfim = None

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
        Bsz, N, C = x.shape

        x_bar = self.dw(x.transpose(1, 2)).transpose(1, 2)

        residual = x - x_bar
        x_tilde = x - self.lambda_s.to(dtype=input_dtype) * residual

        if self.use_pfim:
            importance = self.pfim(x_tilde, x_bar, hw_shape=hw_shape)
            if not getattr(self, "pfim_tokenwise", True):
                importance = importance.mean(dim=1, keepdim=True).expand(-1, N, -1)
        else:
            importance = torch.full(
                (Bsz, N, 1),
                0.5,
                device=x.device,
                dtype=input_dtype,
            )

        imp_center = importance - 0.5

        if getattr(self, "capture_importance", False):
            self.last_importance = importance.detach()
            self.last_hw_shape = hw_shape

        u, z = self.in_proj(x_tilde).chunk(2, dim=-1)

        # delta branch
        delta_logits = self.delta_proj(u)
        if self.use_pfim and self.pfim_on_delta:
            delta_logits = delta_logits + self.eta_i.to(dtype=input_dtype) * importance
        delta = F.softplus(delta_logits)

        B_t = self.B_proj(u)
        C_t = self.C_proj(u)
        A = -torch.exp(self.A_log.float())

        energy = torch.norm(
            x_bar.float(),
            p=2,
            dim=-1,
            keepdim=True,
        ) / (C ** 0.5)

        # gamma branch
        if self.use_pfim and self.pfim_on_gamma:
            gamma_logits = (
                -self.alpha.float() * self.sigma_n * energy
                + self.beta_i.float() * imp_center.float()
            )
        else:
            gamma_logits = -self.alpha.float() * self.sigma_n * energy

        gamma_raw = torch.sigmoid(gamma_logits)
        gamma = self.min_retain + (1.0 - self.min_retain) * gamma_raw
        gamma = gamma.to(dtype=input_dtype)

        # write branch
        if self.use_pfim and self.pfim_on_write:
            write_gate = 1.0 + torch.tanh(self.rho_i.to(dtype=input_dtype)) * imp_center
            write_gate = torch.clamp(write_gate, min=0.1)
        else:
            write_gate = 1.0

        u_write = u * write_gate

        if USE_CUDA_SCAN and selective_scan_fn is not None:
            with torch.cuda.amp.autocast(enabled=False):
                u_fp32 = u_write.float()
                delta_fp32 = delta.float()
                B_t_fp32 = B_t.float()
                C_t_fp32 = C_t.float()
                gamma_fp32 = gamma.float()
                x_tilde_fp32 = x_tilde.float()
                z_fp32 = z.float()
                D_fp32 = self.D.float()

                u_damped = u_fp32 * gamma_fp32

                u_scan = u_damped.transpose(1, 2).contiguous()
                delta_scan = delta_fp32.transpose(1, 2).contiguous()
                B_scan = B_t_fp32.transpose(1, 2).contiguous()
                C_scan = C_t_fp32.transpose(1, 2).contiguous()

                y_scan = selective_scan_fn(
                    u_scan,
                    delta_scan,
                    A,
                    B_scan,
                    C_scan,
                    D=None,
                    z=None,
                    delta_bias=None,
                    delta_softplus=False,
                    return_last_state=False,
                )

                y_ssm = y_scan.transpose(1, 2).contiguous()
                y = y_ssm + D_fp32 * x_tilde_fp32
                y = y * F.silu(z_fp32)

            y = y.to(dtype=input_dtype)

        else:
            A = A.to(dtype=input_dtype)
            h = x.new_zeros(Bsz, C, self.state_dim)
            ys = []

            for t in range(N):
                dt = delta[:, t, :].unsqueeze(-1)
                At = torch.exp(dt * A.unsqueeze(0))
                Bt = dt * B_t[:, t, :].unsqueeze(1)
                Ct = C_t[:, t, :].unsqueeze(1)

                u_scan_t = gamma[:, t, :] * u_write[:, t, :]
                h = At * h + Bt * u_scan_t.unsqueeze(-1)

                yt = (
                    (Ct * h).sum(dim=-1)
                    + self.D.to(dtype=input_dtype) * x_tilde[:, t, :]
                )
                ys.append(yt)

            y = torch.stack(ys, dim=1)
            y = y * F.silu(z)

        y = self.out_proj(y)

        if return_importance:
            return y, importance

        return y

class DS_VSSBlock(nn.Module):
    """
    Importance-Guided DS-SSM block used as a drop-in replacement for VSSBlock.
    """

    def __init__(
        self,
        dim,
        state_dim=16,
        mlp_ratio=2.0,
        sigma_n=0.1,
        lambda_init=0.5,
        alpha_init=0.5,
        kernel_size=3,
        drop=0.0,

        use_pfim=True,
        pfim_on_gamma=True,
        pfim_on_write=True,
        pfim_on_delta=True,

        beta_init=1.0,
        rho_init=1.0,
        eta_init=0.1,
        min_retain=0.5,
        pfim_hidden_dim=None,
        pfim_use_2d_std=True,
        pfim_tokenwise=True,
    ):
        super().__init__()

        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)

        self.ds_ssm = DeSpeckleSSM(
            dim=dim,
            state_dim=state_dim,
            sigma_n=sigma_n,
            lambda_init=lambda_init,
            alpha_init=alpha_init,
            kernel_size=kernel_size,

            use_pfim=use_pfim,
            pfim_on_gamma=pfim_on_gamma,
            pfim_on_write=pfim_on_write,
            pfim_on_delta=pfim_on_delta,

            beta_init=beta_init,
            rho_init=rho_init,
            eta_init=eta_init,
            min_retain=min_retain,
            pfim_hidden_dim=pfim_hidden_dim,
            pfim_use_2d_std=pfim_use_2d_std,
            pfim_tokenwise=pfim_tokenwise,
        )

        hidden_dim = int(dim * mlp_ratio)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2)

    def forward(self, x):
        B, C, H, W = x.shape

        assert C == self.dim, (
            f"DS_VSSBlock dim mismatch: input C={C}, module dim={self.dim}"
        )

        x_tokens = x.flatten(2).transpose(1, 2).contiguous()

        x_tokens = x_tokens + self.gamma1.to(dtype=x_tokens.dtype) * self.ds_ssm(
            self.norm1(x_tokens),
            hw_shape=(H, W),
        )

        x_tokens = x_tokens + self.gamma2.to(dtype=x_tokens.dtype) * self.mlp(
            self.norm2(x_tokens)
        )

        x = x_tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return x


class DS_VSSBlockSweep(DS_VSSBlock):
    """DS block adapter with an explicit state dimension for placement sweeps.

    The Ultralytics parser supplies input and scaled output channels as the
    first two positional arguments. The historical ``DS_VSSBlock`` interprets
    its second positional argument as ``state_dim``. This adapter preserves old
    checkpoints while allowing new P2/P3/P4/P5 comparisons to hold state size
    constant.
    """

    def __init__(self, in_channels, hidden_dim, state_dim=16):
        if in_channels != hidden_dim:
            raise ValueError(
                "DS_VSSBlockSweep requires equal input/output channels, "
                f"got in_channels={in_channels}, hidden_dim={hidden_dim}"
            )
        super().__init__(dim=in_channels, state_dim=state_dim)


def _check_sweep_channels(in_channels, hidden_dim):
    if in_channels != hidden_dim:
        raise ValueError(
            "DS sweep blocks require equal input/output channels, "
            f"got in_channels={in_channels}, hidden_dim={hidden_dim}"
        )


class DS_VSSBlockNoPFIM(DS_VSSBlock):
    def __init__(self, dim, state_dim=16):
        super().__init__(dim=dim, state_dim=state_dim, use_pfim=False)


class DS_VSSBlockPFIMOnly(DS_VSSBlock):
    """PFIM control without residual correction or energy-driven damping."""

    def __init__(self, dim, state_dim=16):
        super().__init__(dim=dim, state_dim=state_dim)
        with torch.no_grad():
            self.ds_ssm.lambda_s.zero_()
            self.ds_ssm.alpha.zero_()
        self.ds_ssm.lambda_s.requires_grad_(False)
        self.ds_ssm.alpha.requires_grad_(False)


class DS_VSSBlockPFIMGamma(DS_VSSBlock):
    def __init__(self, dim, state_dim=16):
        super().__init__(
            dim=dim,
            state_dim=state_dim,
            pfim_on_gamma=True,
            pfim_on_write=False,
            pfim_on_delta=False,
        )


class DS_VSSBlockPFIMWrite(DS_VSSBlock):
    def __init__(self, dim, state_dim=16):
        super().__init__(
            dim=dim,
            state_dim=state_dim,
            pfim_on_gamma=False,
            pfim_on_write=True,
            pfim_on_delta=False,
        )


class DS_VSSBlockPFIMDelta(DS_VSSBlock):
    def __init__(self, dim, state_dim=16):
        super().__init__(
            dim=dim,
            state_dim=state_dim,
            pfim_on_gamma=False,
            pfim_on_write=False,
            pfim_on_delta=True,
        )


class DS_VSSBlockPFIMGammaWrite(DS_VSSBlock):
    def __init__(self, dim, state_dim=16):
        super().__init__(
            dim=dim,
            state_dim=state_dim,
            pfim_on_gamma=True,
            pfim_on_write=True,
            pfim_on_delta=False,
        )


class DS_VSSBlockPFIMGlobal(DS_VSSBlock):
    """Parameter-matched control using one image-level importance value per sample."""

    def __init__(self, dim, state_dim=16):
        super().__init__(dim=dim, state_dim=state_dim, pfim_tokenwise=False)


class DS_VSSBlockPFIMNo2DStd(DS_VSSBlock):
    """Parameter-matched PFIM control using sequence-only local variation."""

    def __init__(self, dim, state_dim=16):
        super().__init__(dim=dim, state_dim=state_dim, pfim_use_2d_std=False)


class DS_VSSBlockSweepNoPFIM(DS_VSSBlockNoPFIM):
    """Placement-sweep adapter for the DS-only control."""

    def __init__(self, in_channels, hidden_dim, state_dim=16):
        _check_sweep_channels(in_channels, hidden_dim)
        super().__init__(dim=in_channels, state_dim=state_dim)


class DS_VSSBlockSweepPFIMOnly(DS_VSSBlockPFIMOnly):
    """Placement-sweep adapter for the PFIM-only control."""

    def __init__(self, in_channels, hidden_dim, state_dim=16):
        _check_sweep_channels(in_channels, hidden_dim)
        super().__init__(dim=in_channels, state_dim=state_dim)


class DS_VSSBlockSweepPFIMGlobal(DS_VSSBlockPFIMGlobal):
    """Placement-sweep adapter for the parameter-matched global control."""

    def __init__(self, in_channels, hidden_dim, state_dim=16):
        _check_sweep_channels(in_channels, hidden_dim)
        super().__init__(dim=in_channels, state_dim=state_dim)


############################################### PFIM


class SemanticSSM(nn.Module):
    """
    Semantic Selective State Space Model (适用于 P4/P5 高级语义层)
    丢弃局部去偏置，采用基于激活范数的语义软阻尼，增强前景，抑制背景杂波。
    """
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # 语义门控的投影层：将 token 的能量特征映射为阻尼系数
        self.energy_proj = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1)
        )

        # Mamba 标准投影参数
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.delta_proj = nn.Linear(dim, dim, bias=True)
        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        self.C_proj = nn.Linear(dim, state_dim, bias=False)

        # 稳定 A 初始化
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        input_dtype = x.dtype
        Bsz, N, C = x.shape

        # 1. 计算 Token 级别的语义能量 (L2 Norm)
        # 高能量通常对应显著目标，低能量对应背景
        energy = torch.norm(x.float(), p=2, dim=-1, keepdim=True).to(input_dtype) # [B, N, 1]
        
        # 2. 生成语义阻尼系数 (Semantic Damping)
        # 映射并使用 Sigmoid，使得高能区域接近 1，低能区域接近 0
        gamma = torch.sigmoid(self.energy_proj(energy))
           
        
        u, z = self.in_proj(x).chunk(2, dim=-1)
        delta = F.softplus(self.delta_proj(u))
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)
        A = -torch.exp(self.A_log.float())

        if USE_CUDA_SCAN and selective_scan_fn is not None:
            with torch.cuda.amp.autocast(enabled=False):
                u_fp32 = u.float()
                delta_fp32 = delta.float()
                B_t_fp32 = B_t.float()
                C_t_fp32 = C_t.float()
                gamma_fp32 = gamma.float()
                z_fp32 = z.float()
                D_fp32 = self.D.float()

                # --- 软语义门控 ---
                min_retain = 0.5 
                # 背景杂波最少保留 50%，显著目标保留近乎 100%
                gamma_soft = min_retain + (1.0 - min_retain) * gamma_fp32
                u_gated = u_fp32 * gamma_soft

                u_scan = u_gated.transpose(1, 2).contiguous()
                delta_scan = delta_fp32.transpose(1, 2).contiguous()
                B_scan = B_t_fp32.transpose(1, 2).contiguous()
                C_scan = C_t_fp32.transpose(1, 2).contiguous()

                y_scan = selective_scan_fn(
                    u_scan, delta_scan, A, B_scan, C_scan,
                    D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
                )

                y_ssm = y_scan.transpose(1, 2).contiguous()
                y = y_ssm + D_fp32 * x.float()
                y = y * F.silu(z_fp32)
            y = y.to(dtype=input_dtype)
        else:
            # Fallback for Python loop (省略以保持简洁，与 DS-SSM 类似，将 gamma 作用于输入或隐状态)
            pass

        return self.out_proj(y)

class Sem_VSSBlock(nn.Module):
    """
    用于高层语义的 VSSBlock。替换 YAML 中的深层 VSSBlock。
    """
    def __init__(self, dim, state_dim=16, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.sem_ssm = SemanticSSM(dim=dim, state_dim=state_dim)

        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_tokens = x.flatten(2).transpose(1, 2).contiguous()
        
        x_tokens = x_tokens + self.gamma1.to(dtype=x_tokens.dtype) * self.sem_ssm(self.norm1(x_tokens))
        x_tokens = x_tokens + self.gamma2.to(dtype=x_tokens.dtype) * self.mlp(self.norm2(x_tokens))
        
        return x_tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()


###############################################

class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)
