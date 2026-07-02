# MaDiNet 官方代码其它模块复杂度统计

生成时间：2026-06-14

服务器：`172.26.151.143`

说明：这些结果来自 MaDiNet 官方仓库中的 `MambaSAR.py` 和 `pixel difference convPDC.py`，属于组件级统计，不是完整检测器 end-to-end 统计。

| 组件 | 输入 | Params(M) | GFLOPs(THOP) | latency(ms) | 说明 |
|---|---|---:|---:|---:|---|
| MambaVisionMixer-d1024-L1600 | 1x1600x1024 | 2.1791 | 6.9468 | 0.6079 | official Mamba mixer at res4 token scale |
| SelfAttention-d1024-L1600 | 1x1600x1024 | 4.1984 | 13.4218 | 2.0705 | attention branch used in official Block |
| AgentAttention-d1024-L1600 | 1x1600x1024 | 4.3098 | 13.4621 | 1.7834 | agent attention candidate in official source comments |
| Block-Mamba-d1024-L1600 | 1x1600x1024 | 10.5769 | 33.8166 | 2.2414 | official Block using MambaVisionMixer |
| Block-Attention-d1024-L1600 | 1x1600x1024 | 12.5962 | 40.2915 | 3.6363 | official Block using self-attention |
| PDC-CPDC-256 | 1x256x80x80 | 0.5898 | THOP unsupported | 0.4395 | official pixel difference convolution candidate |
| PDC-APDC-256 | 1x256x80x80 | 0.5898 | THOP unsupported | 0.3945 | official pixel difference convolution candidate |
| PDC-RPDC-256 | 1x256x80x80 | 0.5898 | THOP unsupported | 0.9972 | official pixel difference convolution candidate |

## 观察

1. MaDiNet 官方 MambaVisionMixer 单块在 `d=1024, L=1600` 下约 2.18M 参数、6.95 GFLOPs。
2. 官方 Block-Mamba 单块约 10.58M 参数、33.82 GFLOPs；Block-Attention 单块约 12.60M 参数、40.29 GFLOPs。
3. 官方源码注释中的 res4+res5 组合包含多层 Block，因此整体核心模块达到 223.26M 参数、345.28 GFLOPs。
4. PDC 方法参数量不大，但 THOP 无法识别函数式差分卷积 FLOPs；当前保留服务器实测 latency。

## 对本文的启发

MaDiNet 的 Mamba/Attention 模块具有较强表达能力，但单模块和多层堆叠的复杂度明显较高。本文 DS-SSM 的优势是将 SAR 去斑选择性建模压缩到完整约 6M 参数、约 13G FLOPs 的检测器中，适合写作成轻量高效路线。