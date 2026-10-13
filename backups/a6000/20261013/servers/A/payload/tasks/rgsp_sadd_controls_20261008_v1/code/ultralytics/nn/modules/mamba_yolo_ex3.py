from .common_utils_mbyolo import *

__all__ = ("VSSBlock", "SimpleStem", "VisionClueMerge", "XSSBlock")



# =========================
# DS-SSM for SAR Despeckling
# =========================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# try:
#     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
#     USE_CUDA_SCAN = True
# except Exception:
#     selective_scan_fn = None
#     USE_CUDA_SCAN = False
#     print("Warning: mamba_ssm selective_scan_fn is not available. "
#           "DS-SSM will fall back to a slow Python loop.")


# class DeSpeckleSSM(nn.Module):
#     """
#     De-Speckle Selective State Space Model.

#     Input:
#         x: [B, N, C]
#     Output:
#         y: [B, N, C]

#     Design:
#         1. local clean signal estimation by depthwise Conv1d
#         2. multiplicative-speckle-aware input de-biasing
#         3. selective SSM from corrected tokens
#         4. energy-aware damping
#     """

#     def __init__(
#         self,
#         dim,
#         state_dim=16,
#         sigma_n=0.1,
#         lambda_init=0.5,
#         alpha_init=0.5,
#         kernel_size=3,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.state_dim = state_dim
#         self.sigma_n = sigma_n
#         self.eps = 1e-5

#         # Learnable de-speckle parameters
#         self.lambda_s = nn.Parameter(torch.tensor(float(lambda_init)))
#         self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

#         # Local signal estimator
#         pad = kernel_size // 2
#         self.dw = nn.Conv1d(
#             dim,
#             dim,
#             kernel_size=kernel_size,
#             padding=pad,
#             groups=dim,
#             bias=False,
#         )
#         nn.init.constant_(self.dw.weight, 1.0 / kernel_size)

#         # Selective SSM parameter generators
#         self.in_proj = nn.Linear(dim, dim * 2, bias=False)
#         self.delta_proj = nn.Linear(dim, dim, bias=True)

#         # IMPORTANT:
#         # B and C are shared across feature channels for compatibility
#         # with official selective_scan_fn.
#         self.B_proj = nn.Linear(dim, state_dim, bias=False)
#         self.C_proj = nn.Linear(dim, state_dim, bias=False)

#         # Stable negative A initialization
#         A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
#         self.A_log = nn.Parameter(torch.log(A))

#         self.D = nn.Parameter(torch.ones(dim))
#         self.out_proj = nn.Linear(dim, dim, bias=False)

#     def forward(self, x):
#         """
#         Args:
#             x: [B, N, C]
#         """
#         B, N, C = x.shape
#         assert C == self.dim, f"DeSpeckleSSM dim mismatch: input C={C}, module dim={self.dim}"

#         # 1. Local signal estimation
#         # [B, N, C] -> [B, C, N] -> [B, N, C]
#         x_bar = self.dw(x.transpose(1, 2)).transpose(1, 2)

#         # 2. Input-side multiplicative de-biasing
#         # speckle residual: (x - x_bar) / (|x_bar| + eps)
#         # corrected token: x - lambda * x_bar * residual
#         # equivalent to x - lambda * (x - x_bar), but this form is more interpretable
#         residual = (x - x_bar) / (torch.abs(x_bar) + self.eps)
#         x_tilde = x - self.lambda_s * x_bar * residual

#         # 3. Selective parameters from corrected tokens
#         u, z = self.in_proj(x_tilde).chunk(2, dim=-1)  # [B, N, C]
#         delta = F.softplus(self.delta_proj(u))         # [B, N, C]

#         B_t = self.B_proj(u)  # [B, N, S]
#         C_t = self.C_proj(u)  # [B, N, S]

#         A = -torch.exp(self.A_log.float())  # [C, S]

#         # 4. Energy-aware damping
#         # use squared energy to match speckle variance intuition
#         energy = torch.sum(x_bar ** 2, dim=-1, keepdim=True)  # [B, N, 1]
#         gamma = torch.sigmoid(-self.alpha * self.sigma_n * energy)  # [B, N, 1]

#         # 5. Selective scan
#         if USE_CUDA_SCAN and selective_scan_fn is not None:
#             # official selective_scan_fn does not accept an independent gamma,
#             # so we inject damping by modulating u.
#             u_damped = u * gamma

#             # selective_scan_fn expects [B, C, N]
#             u_scan = u_damped.transpose(1, 2).contiguous()
#             delta_scan = delta.transpose(1, 2).contiguous()

#             # variable B/C shape: [B, S, N]
#             B_scan = B_t.transpose(1, 2).contiguous()
#             C_scan = C_t.transpose(1, 2).contiguous()

#             y_scan = selective_scan_fn(
#                 u_scan,
#                 delta_scan,
#                 A,
#                 B_scan,
#                 C_scan,
#                 D=None,
#                 z=None,
#                 delta_bias=None,
#                 delta_softplus=False,
#                 return_last_state=False,
#             )

#             y_ssm = y_scan.transpose(1, 2).contiguous()  # [B, N, C]

#             # output equation: y = C h + D x_tilde
#             y = y_ssm + self.D * x_tilde
#             y = y * F.silu(z)

#         else:
#             # Slow fallback for debugging only
#             h = x.new_zeros(B, C, self.state_dim)
#             ys = []

#             for t in range(N):
#                 dt = delta[:, t, :].unsqueeze(-1)      # [B, C, 1]
#                 At = torch.exp(dt * A.unsqueeze(0))    # [B, C, S]

#                 Bt = dt * B_t[:, t, :].unsqueeze(1)    # [B, C, S]
#                 Ct = C_t[:, t, :].unsqueeze(1)         # [B, 1, S]

#                 h = gamma[:, t, :].unsqueeze(-1) * At * h + Bt * u[:, t, :].unsqueeze(-1)

#                 yt = (Ct * h).sum(dim=-1) + self.D * x_tilde[:, t, :]
#                 ys.append(yt)

#             y = torch.stack(ys, dim=1)  # [B, N, C]
#             y = y * F.silu(z)

#         return self.out_proj(y)


# class DS_VSSBlock(nn.Module):
#     """
#     DS-SSM block used as a drop-in replacement for VSSBlock in Mamba-YOLO.

#     Input:
#         x: [B, C, H, W]
#     Output:
#         x: [B, C, H, W]

#     This block contains:
#         Norm -> DeSpeckleSSM -> Residual
#         Norm -> MLP/FFN       -> Residual

#     YAML usage:
#         - [-1, 3, DS_VSSBlock, [128]]
#         - [-1, 3, DS_VSSBlock, [256]]
#     """

#     def __init__(
#         self,
#         dim,
#         state_dim=16,
#         mlp_ratio=2.0,
#         sigma_n=0.1,
#         lambda_init=0.5,
#         alpha_init=0.5,
#         kernel_size=3,
#         drop=0.0,
#     ):
#         super().__init__()

#         self.dim = dim

#         self.norm1 = nn.LayerNorm(dim)
#         self.ds_ssm = DeSpeckleSSM(
#             dim=dim,
#             state_dim=state_dim,
#             sigma_n=sigma_n,
#             lambda_init=lambda_init,
#             alpha_init=alpha_init,
#             kernel_size=kernel_size,
#         )

#         hidden_dim = int(dim * mlp_ratio)

#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(drop),
#         )

#         # Layer scale: safer for replacing original backbone blocks
#         self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2)
#         self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2)

#     def forward(self, x):
#         """
#         Args:
#             x: [B, C, H, W]
#         """
#         B, C, H, W = x.shape
#         assert C == self.dim, f"DS_VSSBlock dim mismatch: input C={C}, module dim={self.dim}"

#         # [B, C, H, W] -> [B, N, C]
#         x_tokens = x.flatten(2).transpose(1, 2).contiguous()

#         # DS-SSM branch
#         x_tokens = x_tokens + self.gamma1 * self.ds_ssm(self.norm1(x_tokens))

#         # FFN branch
#         x_tokens = x_tokens + self.gamma2 * self.mlp(self.norm2(x_tokens))

#         # [B, N, C] -> [B, C, H, W]
#         x = x_tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
#         return x
    
    
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    USE_CUDA_SCAN = True
except Exception:
    selective_scan_fn = None
    USE_CUDA_SCAN = False
    print(
        "Warning: mamba_ssm selective_scan_fn is not available. "
        "DS-SSM will fall back to a slow Python loop."
    )


class DeSpeckleSSM(nn.Module):
    """
    De-Speckle Selective State Space Model.

    Input:
        x: [B, N, C]
    Output:
        y: [B, N, C]
    """

    def __init__(
        self,
        dim,
        state_dim=16,
        sigma_n=0.1,
        lambda_init=0.5,
        alpha_init=0.5,
        kernel_size=3,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.sigma_n = sigma_n
        self.eps = 1e-5

        # Learnable de-speckle parameters
        self.lambda_s = nn.Parameter(torch.tensor(float(lambda_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Local signal estimator
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

        # Selective SSM parameter generators
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.delta_proj = nn.Linear(dim, dim, bias=True)

        # B and C are shared across feature channels
        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        self.C_proj = nn.Linear(dim, state_dim, bias=False)

        # Stable negative A initialization
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    # gemini 20260520-0：18
    def forward(self, x):
        input_dtype = x.dtype
        Bsz, N, C = x.shape
        
        # 1. 局部信号估计 (Local signal estimation)
        x_bar = self.dw(x.transpose(1, 2)).transpose(1, 2)

        # 2. 输入端去偏置 (Input-side multiplicative de-biasing)
        # 修复：移除导致 nan 的除法，严格遵循数学化简形式
        residual = x - x_bar
        x_tilde = x - self.lambda_s.to(dtype=input_dtype) * residual

        # 3. 生成选择性参数
        u, z = self.in_proj(x_tilde).chunk(2, dim=-1)
        delta = F.softplus(self.delta_proj(u))
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)
        A = -torch.exp(self.A_log.float())

        # 4. 能量感知阻尼 (Energy-aware damping)
        # 修复：使用 L2 范数，并在 FP32 下计算防止溢出
        energy = torch.norm(x_bar.float(), p=2, dim=-1, keepdim=True).to(input_dtype)
        gamma = torch.sigmoid(-self.alpha.to(dtype=input_dtype) * self.sigma_n * energy)


        # 5. 选择性扫描 (Selective scan)
        if USE_CUDA_SCAN and selective_scan_fn is not None:
            with torch.cuda.amp.autocast(enabled=False):
                u_fp32 = u.float()
                delta_fp32 = delta.float()
                B_t_fp32 = B_t.float()
                C_t_fp32 = C_t.float()
                gamma_fp32 = gamma.float()
                x_tilde_fp32 = x_tilde.float()
                z_fp32 = z.float()
                D_fp32 = self.D.float()

                ## 20260520 11:07
                # --- 实验2：引入软阻尼 (Soft Damping) ---
                min_retain = 0.5
                # 将 gamma 的范围从 (0, 1) 限制到 (0.5, 1.0)
                gamma_soft = min_retain + (1.0 - min_retain) * gamma_fp32
                
                # 将软阻尼作用于输入 u 作为状态阻尼的近似
                u_damped = u_fp32 * gamma_soft
                # ----------------------------------------
                ## 20260520 11:07
                
                
                # # 妥协方案：由于 CUDA kernel 无法原生支持对隐状态 h 施加 gamma
                # # 此处采用“输入阻尼”作为近似替代（或者像你之前的测试一样，直接不用 gamma）
                # # u_damped = u_fp32 * gamma_fp32 
                # u_damped = u_fp32 # 如果你想彻底跑通“去掉 gamma 的版本”，保持这样

                u_scan = u_damped.transpose(1, 2).contiguous()
                delta_scan = delta_fp32.transpose(1, 2).contiguous()
                B_scan = B_t_fp32.transpose(1, 2).contiguous()
                C_scan = C_t_fp32.transpose(1, 2).contiguous()

                y_scan = selective_scan_fn(
                    u_scan, delta_scan, A, B_scan, C_scan,
                    D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
                )

                y_ssm = y_scan.transpose(1, 2).contiguous()
                y = y_ssm + D_fp32 * x_tilde_fp32
                y = y * F.silu(z_fp32)

            y = y.to(dtype=input_dtype)

        else:
            # 这里的 Python 循环包含了严格的【状态阻尼】
            A = A.to(dtype=input_dtype)
            h = x.new_zeros(Bsz, C, self.state_dim)
            ys = []
            for t in range(N):
                dt = delta[:, t, :].unsqueeze(-1)
                At = torch.exp(dt * A.unsqueeze(0))
                Bt = dt * B_t[:, t, :].unsqueeze(1)
                Ct = C_t[:, t, :].unsqueeze(1)

                # 严格的状态阻尼
                h = gamma[:, t, :].unsqueeze(-1) * At * h + Bt * u[:, t, :].unsqueeze(-1)
                yt = (Ct * h).sum(dim=-1) + self.D.to(dtype=input_dtype) * x_tilde[:, t, :]
                ys.append(yt)

            y = torch.stack(ys, dim=1)
            y = y * F.silu(z)

        return self.out_proj(y)

    # def forward(self, x):
    #     """
    #     Args:
    #         x: [B, N, C]
    #     """
    #     input_dtype = x.dtype

    #     Bsz, N, C = x.shape
    #     assert C == self.dim, (
    #         f"DeSpeckleSSM dim mismatch: input C={C}, module dim={self.dim}"
    #     )

    #     # 1. Local signal estimation
    #     # [B, N, C] -> [B, C, N] -> [B, N, C]
    #     x_bar = self.dw(x.transpose(1, 2)).transpose(1, 2)

    #     # 2. Input-side multiplicative de-biasing
    #     residual = (x - x_bar) / (torch.abs(x_bar) + self.eps)
    #     x_tilde = x - self.lambda_s.to(dtype=x.dtype) * x_bar * residual

    #     # 3. Selective parameters from corrected tokens
    #     u, z = self.in_proj(x_tilde).chunk(2, dim=-1)  # [B, N, C]
    #     delta = F.softplus(self.delta_proj(u))         # [B, N, C]

    #     B_t = self.B_proj(u)  # [B, N, S]
    #     C_t = self.C_proj(u)  # [B, N, S]

    #     # A should stay FP32 for numerical stability
    #     A = -torch.exp(self.A_log.float())  # [C, S]

    #     # 4. Energy-aware damping
    #     energy = torch.sum(x_bar ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    #     gamma = torch.sigmoid(
    #         -self.alpha.to(dtype=energy.dtype) * self.sigma_n * energy
    #     )  # [B, N, 1]

    #     # 5. Selective scan
    #     if USE_CUDA_SCAN and selective_scan_fn is not None:
    #         # 关键修复：
    #         # selective_scan_cuda 对 dtype 很严格，AMP 下容易出现
    #         # u 是 fp16，但 B/C/A 是 fp32 的情况，所以这里强制 FP32。
    #         with torch.cuda.amp.autocast(enabled=False):
    #             u_fp32 = u.float()
    #             delta_fp32 = delta.float()
    #             B_t_fp32 = B_t.float()
    #             C_t_fp32 = C_t.float()
    #             gamma_fp32 = gamma.float()
    #             x_tilde_fp32 = x_tilde.float()
    #             z_fp32 = z.float()
    #             D_fp32 = self.D.float()

    #             # Inject damping by modulating u
    #             # 20260519 
    #             # u_damped = u_fp32 * gamma_fp32
    #             u_damped = u_fp32

    #             # selective_scan_fn expects:
    #             # u, delta: [B, C, N]
    #             # variable B/C: [B, S, N]
    #             u_scan = u_damped.transpose(1, 2).contiguous()
    #             delta_scan = delta_fp32.transpose(1, 2).contiguous()

    #             B_scan = B_t_fp32.transpose(1, 2).contiguous()
    #             C_scan = C_t_fp32.transpose(1, 2).contiguous()

    #             y_scan = selective_scan_fn(
    #                 u_scan,
    #                 delta_scan,
    #                 A,
    #                 B_scan,
    #                 C_scan,
    #                 D=None,
    #                 z=None,
    #                 delta_bias=None,
    #                 delta_softplus=False,
    #                 return_last_state=False,
    #             )

    #             y_ssm = y_scan.transpose(1, 2).contiguous()  # [B, N, C]

    #             # output equation: y = C h + D x_tilde
    #             y = y_ssm + D_fp32 * x_tilde_fp32
    #             y = y * F.silu(z_fp32)

    #         # 转回输入 dtype，避免后续 residual / MLP 混合类型继续扩大
    #         y = y.to(dtype=input_dtype)

    #     else:
    #         # Slow fallback for debugging only
    #         A = A.to(dtype=x.dtype)
    #         h = x.new_zeros(Bsz, C, self.state_dim)
    #         ys = []

    #         for t in range(N):
    #             dt = delta[:, t, :].unsqueeze(-1)      # [B, C, 1]
    #             At = torch.exp(dt * A.unsqueeze(0))    # [B, C, S]

    #             Bt = dt * B_t[:, t, :].unsqueeze(1)    # [B, C, S]
    #             Ct = C_t[:, t, :].unsqueeze(1)         # [B, 1, S]

    #             h = (
    #                 gamma[:, t, :].unsqueeze(-1) * At * h
    #                 + Bt * u[:, t, :].unsqueeze(-1)
    #             )

    #             yt = (Ct * h).sum(dim=-1) + self.D.to(dtype=x.dtype) * x_tilde[:, t, :]
    #             ys.append(yt)

    #         y = torch.stack(ys, dim=1)  # [B, N, C]
    #         y = y * F.silu(z)

    #     return self.out_proj(y)


class DS_VSSBlock(nn.Module):
    """
    DS-SSM block used as a drop-in replacement for VSSBlock in Mamba-YOLO.

    Input:
        x: [B, C, H, W]
    Output:
        x: [B, C, H, W]

    YAML usage:
        - [-1, 3, DS_VSSBlock, [128]]
        - [-1, 3, DS_VSSBlock, [256]]
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

        # Layer scale
        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.dim, (
            f"DS_VSSBlock dim mismatch: input C={C}, module dim={self.dim}"
        )

        # [B, C, H, W] -> [B, N, C]
        x_tokens = x.flatten(2).transpose(1, 2).contiguous()

        # DS-SSM branch
        x_tokens = x_tokens + self.gamma1.to(dtype=x_tokens.dtype) * self.ds_ssm(
            self.norm1(x_tokens)
        )

        # FFN branch
        x_tokens = x_tokens + self.gamma2.to(dtype=x_tokens.dtype) * self.mlp(
            self.norm2(x_tokens)
        )

        # [B, N, C] -> [B, C, H, W]
        x = x_tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return x    
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
