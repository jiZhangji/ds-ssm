"""Six editable, code-grounded RGSP mechanism diagrams.

Authoring only: no model import, inference, benchmark, training, or synthetic
activation evidence.  ``build(save, data_dir)`` delegates every output operation
to the gallery's ``save`` callback.  All marks are Matplotlib vector artists.

Source of truth: the frozen 2026-10-09 final-patch implementation, the campaign
manifest, and the current full-experiments manuscript.  In particular, detector
head scales are P3/P4/P5; P2/P3/P4 are the separate placement-control candidates.
The mathematical symbol C denotes feature channels, while C_t is the readout
projection.  I is a trained structural control, never a calibrated reliability
map.  Every emitted artifact is explicitly tagged ``evidence='schematic'``.

No file-system access is needed; data_dir is reserved for the shared interface.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


MODULE = (
    "experiments/rgsp_seedwise_20261009_v1/transfer/final_patch_v1/code/"
    "ultralytics/nn/modules/mamba_yolo.py"
)
MANIFEST = "experiments/rgsp_seedwise_20261009_v1/manifest.py"
PAPER = "paper_draft/rgsp_cn_full_experiments_20261009/main.tex"
ABLATION = "paper_draft/rgsp_cn_full_experiments_20261009/sections/ablation.tex"

INK = "#263845"
MUTED = "#607280"
RULE = "#C6D1D8"
ORANGE = "#B87645"
BLUE_EDGE = "#88A9BF"
GREEN_EDGE = "#82A490"
PALETTE = {
    "blue": ("#EAF2F8", BLUE_EDGE),
    "sage": ("#EAF3EE", GREEN_EDGE),
    "peach": ("#FBEDE0", "#C7976E"),
    "lavender": ("#F0EDF7", "#ACA1C8"),
    "neutral": ("#F6F8FA", RULE),
    "white": ("#FFFFFF", RULE),
}


def _canvas(title, subtitle, *, height=7.6, ymax=6.6):
    fig = plt.figure(figsize=(10, height), facecolor="white")
    ax = fig.add_axes([0.035, 0.075, 0.93, 0.785])
    ax.set_xlim(0, 10)
    ax.set_ylim(ymax, 0)
    ax.axis("off")
    fig.text(0.04, 0.958, title, color=INK, fontsize=17, weight="bold", va="top")
    fig.text(0.04, 0.912, subtitle, color=MUTED, fontsize=10.3, va="top")
    fig.text(
        0.04, 0.026,
        "Code-aligned schematic | not a prediction, activation map, or measured mechanism result",
        color=MUTED, fontsize=9.0, va="bottom",
    )
    return fig, ax


def _label(ax, x, y, text, *, size=10.4, color=INK, bold=False, ha="center", va="center"):
    return ax.text(
        x, y, text, fontsize=size, color=color, ha=ha, va=va,
        weight="bold" if bold else "normal", linespacing=1.35,
        zorder=6, clip_on=False,
    )


def _rect(ax, x, y, w, h, *, fill="white", edge=RULE, dashed=False, zorder=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.1, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _node(ax, x, y, w, h, text, *, kind="blue", size=10.4, bold=False):
    fill, edge = PALETTE[kind]
    patch = _rect(ax, x, y, w, h, fill=fill, edge=edge, zorder=4)
    _label(ax, x + w / 2, y + h / 2, text, size=size, bold=bold)
    return patch


def _line(ax, points, *, color=INK, dashed=False, arrow=True, lw=1.3, zorder=3):
    """Orthogonal or straight connector; only the terminal segment has a head."""
    style = (0, (4, 3)) if dashed else "solid"
    xs, ys = zip(*points)
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=style,
                solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    if arrow:
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=11,
            color=color, linewidth=lw, linestyle=style,
            shrinkA=0, shrinkB=0, zorder=zorder,
        ))
    else:
        ax.plot(xs[-2:], ys[-2:], color=color, lw=lw, linestyle=style,
                solid_capstyle="round", zorder=zorder)


def _dot(ax, x, y, *, color=INK):
    ax.add_patch(Circle((x, y), 0.035, facecolor=color, edgecolor=color, zorder=7))


def _op(ax, x, y, label="+", *, r=0.16):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=1.1, zorder=5))
    _label(ax, x, y, label, size=12)


def _section(ax, x, y, title):
    _label(ax, x, y, title, size=11.4, bold=True, ha="left")


def _m01(save):
    fig, ax = _canvas(
        "RGSP in the detection network",
        "One P3/8 backbone replacement; the detector retains three output scales.",
        height=7.5, ymax=6.35,
    )
    _rect(ax, 0.05, 0.15, 9.85, 1.95, fill="#FCFDFE")
    _section(ax, 0.22, 0.40, "Hierarchical backbone")
    specs = [
        (0.20, 0.92, 0.90, 0.77, "SAR\ninput", "neutral"),
        (1.43, 0.92, 1.15, 0.77, "SimpleStem\nstride 4", "lavender"),
        (2.94, 0.92, 1.18, 0.77, "Stage 1\nP2 / 4", "blue"),
        (4.49, 0.84, 1.48, 0.93, "Stage 2: RGSP\nP3 / 8", "peach"),
        (6.40, 0.92, 1.25, 0.77, "Stage 3\nP4 / 16", "blue"),
        (8.10, 0.84, 1.53, 0.93, "Stage 4 + SPPF\nP5 / 32", "blue"),
    ]
    for x, y, w, h, text, kind in specs:
        _node(ax, x, y, w, h, text, kind=kind, size=10.0, bold=kind == "peach")
    for a, b in zip(specs[:-1], specs[1:]):
        _line(ax, [(a[0] + a[2], 1.305), (b[0], 1.305)])
    for x in (4.29, 6.18, 7.87):
        _label(ax, x, 0.69, "2x", size=9.0, color=MUTED)
    _label(ax, 5.23, 1.94, "one actual block after depth scaling", size=9.0, color=ORANGE)

    # The detector's three lateral inputs are P3/P4/P5, never P2/P3/P4.
    _rect(ax, 4.45, 2.77, 5.20, 1.44, fill="#F7FBF8", edge=GREEN_EDGE)
    _label(ax, 7.05, 3.01, "Bidirectional multi-scale neck", size=11.1, bold=True)
    neck_centres = (5.23, 7.02, 8.85)
    for cx, text in zip(neck_centres, ("P3", "P4", "P5")):
        _node(ax, cx - 0.48, 3.26, 0.96, 0.47, text, kind="sage", size=10.1)
    for left, right in zip(neck_centres[:-1], neck_centres[1:]):
        _line(ax, [(right - 0.48, 3.40), (left + 0.48, 3.40)], color=GREEN_EDGE)
        _line(ax, [(left + 0.48, 3.60), (right - 0.48, 3.60)], color=GREEN_EDGE)
    _label(ax, 7.05, 3.96, "Upsample / downsample + concat + XSSBlocks", size=9.1, color=MUTED)
    for cx, y0 in ((5.23, 1.77), (7.02, 1.69), (8.85, 1.77)):
        _line(ax, [(cx, y0), (cx, 2.77)])
    for cx, text in zip(neck_centres, ("Detect: P3 / 8", "Detect: P4 / 16", "Detect: P5 / 32")):
        _line(ax, [(cx, 4.21), (cx, 4.66)])
        _node(ax, cx - 0.74, 4.66, 1.48, 0.64, text, kind="lavender", size=9.6)
    _label(ax, 7.05, 5.68, "Three scales feed one detection task", size=10.4, bold=True)
    _label(ax, 7.05, 6.02, "Neck internals are abstracted; connections retain their scale identity.",
           size=9.0, color=MUTED)

    _rect(ax, 0.18, 2.78, 3.75, 2.93, fill="#FAFBFD", dashed=True)
    _section(ax, 0.40, 3.08, "Separate placement controls")
    for x, text, kind in ((0.42, "P2 / 4", "blue"), (1.58, "P3 / 8", "peach"), (2.74, "P4 / 16", "blue")):
        _node(ax, x, 3.50, 0.96, 0.51, text, kind=kind, size=9.7)
    _label(ax, 2.05, 4.43, "Each alternative replaces exactly one block.\nThese are not three simultaneous replacements.", size=9.6)
    _label(ax, 2.05, 5.14, "Main instantiation: P3/8, state size 16.\nSpatial size and channels are preserved.", size=9.6, color=MUTED)
    save(
        fig, "M01", "检测网络与单块插入位置", "mechanism",
        "检测网络结构示意：主配置在骨干P3/8替换一个实际RGSP块，P3/8、P4/16、P5/32特征经多尺度颈部进入检测头。"
        "P2/P3/P4仅为相互独立的单位置对照，不是检测头尺度，也不是同时替换。颈部内部作抽象展示；本图不证明P3为最优位置或产生性能提升。",
        [MODULE, MANIFEST, PAPER], evidence="schematic",
    )


def _m02(save):
    fig, ax = _canvas(
        "The complete RGSP residual block",
        "The state-modeling branch and the FFN each have their own residual update.",
        height=7.9, ymax=6.60,
    )
    _section(ax, 0.12, 0.24, "1  State-modeling update")
    _node(ax, 0.10, 1.22, 0.88, 0.76, "$F$\nB,C,H,W", kind="neutral", size=9.3)
    _node(ax, 1.28, 1.22, 1.13, 0.76, "Flatten\nrow-major", kind="lavender", size=10.0)
    _node(ax, 2.96, 1.22, 0.86, 0.76, "$\\mathrm{LN}_1$", kind="lavender", size=11.3)
    _node(ax, 4.23, 1.15, 1.92, 0.90, "State modeling $\\mathcal{R}$\nLRC + LSGM + scan", kind="peach", size=10.0)
    _node(ax, 6.55, 1.22, 0.93, 0.76, "$a_1\\odot$\nLayerScale", kind="lavender", size=9.4)
    _op(ax, 8.06, 1.60)
    _node(ax, 8.65, 1.22, 0.90, 0.76, "$U$", kind="neutral", size=12)
    for a, b in ((0.98, 1.28), (2.41, 2.96), (3.82, 4.23), (6.15, 6.55), (7.48, 7.90), (8.22, 8.65)):
        _line(ax, [(a, 1.60), (b, 1.60)])
    _dot(ax, 2.65, 1.60)
    _line(ax, [(2.65, 1.60), (2.65, 0.60), (8.06, 0.60), (8.06, 1.44)])
    _label(ax, 5.27, 0.41, "Identity path: $X$", size=9.5, color=MUTED)
    _label(ax, 1.85, 2.25, "$X\\in\\mathbb{R}^{B\\times N\\times C}$, $N=HW$", size=10.0, color=MUTED)

    # U is carried unchanged to the start of the second row.
    _line(ax, [(9.55, 1.60), (9.81, 1.60), (9.81, 2.77), (0.42, 2.77), (0.42, 4.10), (0.89, 4.10)])
    _label(ax, 5.15, 2.99, "The same sequence U continues to the FFN update", size=9.5, color=MUTED)
    _section(ax, 0.12, 3.42, "2  Feed-forward update")
    _node(ax, 0.89, 3.69, 0.95, 0.82, "$\\mathrm{LN}_2$", kind="lavender", size=11.3)
    _node(ax, 2.36, 3.62, 2.61, 0.96, "FFN: Linear - GELU - Linear\n$C\\rightarrow 2C\\rightarrow C$", kind="lavender", size=10.0)
    _node(ax, 5.48, 3.69, 1.04, 0.82, "$a_2\\odot$\nLayerScale", kind="lavender", size=9.4)
    _op(ax, 7.24, 4.10)
    _node(ax, 7.94, 3.69, 1.23, 0.82, "Reshape\n$N\\rightarrow H,W$", kind="lavender", size=10.0)
    _label(ax, 9.60, 4.10, "$F_{\\mathrm{out}}$", size=11.8)
    for a, b in ((1.84, 2.36), (4.97, 5.48), (6.52, 7.08), (7.40, 7.94), (9.17, 9.36)):
        _line(ax, [(a, 4.10), (b, 4.10)])
    _dot(ax, 0.42, 4.10)
    _line(ax, [(0.42, 4.10), (0.42, 5.20), (7.24, 5.20), (7.24, 4.26)])
    _label(ax, 3.85, 5.00, "Identity path: $U$", size=9.5, color=MUTED)

    _rect(ax, 0.14, 5.67, 9.60, 0.69, fill="#F8FAFC")
    _label(ax, 4.94, 5.91,
           "$U=X+a_1\\odot\\mathcal{R}(\\mathrm{LN}_1(X))$     "
           "$X_{\\mathrm{out}}=U+a_2\\odot\\mathrm{FFN}(\\mathrm{LN}_2(U))$", size=11.0)
    _label(ax, 4.94, 6.20,
           "$a_1,a_2$: learned per-channel scales, initialized at $10^{-2}$; distinct from scan-drive $\\gamma$.",
           size=9.1, color=MUTED)
    save(
        fig, "M02", "完整RGSP块的双残差与FFN", "mechanism",
        "完整替换块示意：输入由B×C×H×W按行优先展平为B×N×C，先执行LN1、状态建模分支和逐通道LayerScale残差融合，"
        "再执行LN2、C→2C→C的GELU前馈分支和第二次残差融合，最后恢复空间布局。a1/a2初始化为0.01，不能与扫描输入调制gamma混淆。"
        "图中省略默认值为0的Dropout；这是冻结实现的结构说明，不是效果证据。",
        [MODULE, PAPER], evidence="schematic",
    )


def _m03(save):
    fig, ax = _canvas(
        "Local response correction (LRC)",
        "A learned sequence-neighbour correction before content projection and selective scan.",
        height=7.7, ymax=6.45,
    )
    _rect(ax, 0.15, 0.18, 9.65, 0.68, fill="#F8FAFC")
    _label(ax, 4.98, 0.52,
           "$\\bar{x}=\\mathrm{DWConv1D}_{k=3}(x)$    "
           "$r=x-\\bar{x}$    $\\widetilde{x}=x-\\lambda_s r$", size=13.0)

    _node(ax, 0.22, 2.21, 1.08, 0.83, "$x$\nnormalized tokens", kind="neutral", size=9.6)
    _op(ax, 4.94, 2.62, "-")
    _node(ax, 5.62, 2.21, 1.08, 0.83, "$\\lambda_s\\,r$\nlearned scalar", kind="peach", size=9.6)
    _op(ax, 7.57, 2.62, "-")
    _node(ax, 8.19, 2.21, 1.52, 0.83, "$\\widetilde{x}$\ncorrected features", kind="peach", size=10.0)
    _line(ax, [(1.30, 2.62), (4.78, 2.62)])
    _line(ax, [(5.10, 2.62), (5.62, 2.62)])
    _line(ax, [(6.70, 2.62), (7.41, 2.62)])
    _line(ax, [(7.73, 2.62), (8.19, 2.62)])
    _label(ax, 5.34, 2.38, "$r$", size=11.3)
    _label(ax, 3.25, 2.38, "+", size=10.0, color=MUTED)
    _label(ax, 7.09, 2.38, "subtract", size=9.0, color=MUTED)

    # Two branches use the original normalized sequence x.
    _dot(ax, 1.57, 2.62)
    _line(ax, [(1.57, 2.62), (1.57, 1.39), (7.57, 1.39), (7.57, 2.46)])
    _label(ax, 4.51, 1.16, "Original x", size=10.0, color=MUTED)
    _label(ax, 7.80, 2.15, "+", size=10.0, color=MUTED)
    _node(ax, 2.02, 3.90, 1.96, 0.92, "Depthwise Conv1D\n$k=3$, per channel", kind="blue", size=10.0)
    _node(ax, 4.28, 3.90, 1.33, 0.92, "$\\bar{x}$\nlearned trend", kind="blue", size=10.0)
    _line(ax, [(1.57, 2.62), (1.57, 4.36), (2.02, 4.36)])
    _line(ax, [(3.98, 4.36), (4.28, 4.36)])
    _line(ax, [(4.94, 3.90), (4.94, 2.78)])
    _label(ax, 5.17, 3.41, "subtract", size=9.0, color=MUTED, ha="left")
    _node(ax, 6.23, 3.90, 3.46, 0.92,
          "Trend descriptor + feature RMS energy\n$E_t=\\Vert\\bar{x}_t\\Vert_2/\\sqrt{C}$", kind="sage", size=10.0)
    _line(ax, [(5.61, 4.36), (6.23, 4.36)])
    _label(ax, 8.96, 3.46, "To content projection\nand 2D local statistics", size=9.4, color=MUTED)

    _rect(ax, 0.23, 5.32, 4.47, 0.88, fill="#FAFBFD")
    _label(ax, 2.465, 5.76,
           "Row-major sequence: a row end is adjacent\nto the next row start. This is a 1D trend.", size=9.7)
    _rect(ax, 5.06, 5.32, 4.63, 0.88, fill="#FBF8F4", edge="#D8C4B1")
    _label(ax, 7.375, 5.76,
           "$\\lambda_s$ is unconstrained and learned; the operation\nis not guaranteed to smooth or denoise an image.", size=9.7)
    save(
        fig, "M03", "LRC局部趋势与残差修正路径", "mechanism",
        "LRC以逐通道一维卷积得到局部趋势xbar，再形成r=x−xbar以及xtilde=x−lambda_s·r。趋势送入LSGM及特征RMS能量，"
        "修正特征送入内容投影与二维统计。k=3卷积以1/3初始化但后续可学习；lambda_s未被约束为正或小于1，因此不把该算子画成必然平滑、去噪或图像复原。"
        "行优先序列存在跨行首尾邻接，二维统计在LSGM中另行补充。",
        [MODULE, PAPER], evidence="schematic",
    )


def _m04(save):
    fig, ax = _canvas(
        "Local structure guidance (LSGM)",
        "Trend and local variation form a task-trained scalar control at every spatial position.",
        height=8.0, ymax=6.72,
    )
    _rect(ax, 0.12, 0.14, 5.27, 3.53, fill="#FBFDFE")
    _rect(ax, 5.69, 0.14, 4.12, 3.53, fill="#FAFCFB", edge=GREEN_EDGE)
    _section(ax, 0.34, 0.42, "Descriptor construction")
    _section(ax, 5.91, 0.42, "Shared guidance predictor")
    _node(ax, 0.32, 0.92, 1.23, 0.65, "$\\bar{x}$\n1D learned trend", kind="blue", size=9.6)
    _node(ax, 0.32, 2.47, 1.23, 0.65, "$\\widetilde{x}$\ncorrected tokens", kind="peach", size=9.6)
    _node(ax, 2.08, 2.35, 1.85, 0.90,
          "Restore H x W\n3 x 3 local std\nFlatten in same order", kind="sage", size=9.7)
    _node(ax, 4.38, 1.53, 0.71, 0.91, "Concat\n$2C$", kind="lavender", size=9.5)
    _line(ax, [(1.55, 1.245), (4.03, 1.245), (4.03, 1.78), (4.38, 1.78)])
    _line(ax, [(1.55, 2.795), (2.08, 2.795)])
    _line(ax, [(3.93, 2.795), (4.18, 2.795), (4.18, 2.20), (4.38, 2.20)])
    _label(ax, 2.89, 1.02, "C channels", size=9.0, color=MUTED)
    _label(ax, 4.10, 3.39, "$s$: C channels", size=9.0, color=MUTED)

    _node(ax, 6.05, 0.88, 3.40, 0.55, "LayerNorm over 2C channels", kind="lavender", size=10.1)
    _node(ax, 6.05, 1.82, 3.40, 0.79,
          "Linear - GELU - Linear\n$2C\\rightarrow\\max(\\lfloor C/4\\rfloor,16)\\rightarrow1$", kind="sage", size=10.0)
    _node(ax, 6.05, 3.00, 3.40, 0.50,
          "Sigmoid: $I_t\\in(0,1)$, one scalar per token", kind="peach", size=9.7)
    _line(ax, [(5.09, 1.98), (5.58, 1.98), (5.58, 1.155), (6.05, 1.155)])
    _line(ax, [(7.75, 1.43), (7.75, 1.82)])
    _line(ax, [(7.75, 2.61), (7.75, 3.00)])

    _section(ax, 0.26, 4.07, "Descriptor controls preserve the same 2C predictor input")
    controls = (
        (0.26, "Full", "$[\\bar{x},\\ s]$", "blue"),
        (2.69, "Trend only", "$[\\bar{x},\\ 0]$", "blue"),
        (5.12, "Std only", "$[0,\\ s]$", "sage"),
        (7.55, "Point only", "$[\\widetilde{x},\\ 0]$", "peach"),
    )
    for x, title, formula, kind in controls:
        _node(ax, x, 4.42, 2.16, 0.91, title + "\n" + formula, kind=kind, size=11.0)
    _label(ax, 4.99, 5.66,
           "$s=\\mathrm{flatten}\\left(\\sqrt{\\max(\\mathcal{P}(\\widetilde{F}^{2})"
           "-\\mathcal{P}(\\widetilde{F})^{2},0)+10^{-5}}\\right)$", size=12.0)
    _label(ax, 4.99, 6.05,
           "P: valid-neighbour mean pooling; boundary padding is excluded from the divisor.", size=9.8, color=MUTED)
    _label(ax, 4.99, 6.40,
           "Final predictor weights start at zero, so I starts at 0.5. I is not a target probability or calibrated confidence.",
           size=9.2, color=MUTED)
    save(
        fig, "M04", "LSGM描述子与trend/std/point对照", "mechanism",
        "LSGM主路径使用[xbar,s]描述子，其中s来自修正特征xtilde恢复二维布局后的3×3局部标准差，边界均值池化不计补零。"
        "LayerNorm及两层MLP输出经Sigmoid得到每位置一个标量I，并广播至通道。冻结扩展实现中的Trend-only、Std-only、Point-only分别使用[xbar,0]、[0,s]、[xtilde,0]，"
        "并保留2C输入宽度。最后一层零初始化使初始I=0.5。I为任务学习的结构控制量，不是目标概率、亮度可靠性或校准置信度；图中不给出对照性能。",
        [MODULE, PAPER], evidence="schematic",
    )


def _m05(save):
    fig, ax = _canvas(
        "Two conditioned locations in selective scan",
        "Gamma and the write factor share one scan drive; delta independently changes the recurrence coefficients.",
        height=9.5, ymax=8.13,
    )
    _section(ax, 0.15, 0.22, "Content-dependent parameters are generated before modulation")
    _node(ax, 0.16, 0.68, 1.00, 0.65, "$\\widetilde{x}_t$", kind="peach", size=12.0)
    _node(ax, 1.54, 0.68, 1.19, 0.65, "$W_{\\mathrm{in}}$\nsplit into u, z", kind="lavender", size=9.8)
    _node(ax, 3.13, 0.68, 0.73, 0.65, "$u_t$", kind="blue", size=12.0)
    _node(ax, 4.47, 0.60, 2.25, 0.82,
          "$B_t=W_Bu_t$\n$C_t=W_Cu_t$", kind="blue", size=11.6)
    _node(ax, 7.25, 0.60, 2.42, 0.82,
          "$A=-\\exp(A_{\\log})$\nlearned, negative entries", kind="neutral", size=10.2)
    for a, b in ((1.16, 1.54), (2.73, 3.13), (3.86, 4.47)):
        _line(ax, [(a, 1.005), (b, 1.005)])
    _label(ax, 5.51, 1.69, "No direct $I\\rightarrow B_t$ or $I\\rightarrow C_t$ connection", size=10.0, color=MUTED)

    # These two cards are intentionally separate. I is the same scalar in both.
    _rect(ax, 0.16, 2.10, 5.40, 1.93, fill="#FEF7F0", edge="#C7976E")
    _rect(ax, 5.86, 2.10, 3.82, 1.93, fill="#FEF7F0", edge="#C7976E")
    _section(ax, 0.39, 2.37, "1  Scan-drive modulation")
    _label(ax, 2.86, 2.86,
           "$\\gamma_t=g_{\\min}+(1-g_{\\min})\\,\\sigma[-\\alpha s_nE_t+\\beta(I_t-0.5)]$", size=10.8)
    _label(ax, 2.86, 3.28,
           "$g_{w,t}=\\max[0.1,\\ 1+\\tanh(\\rho)(I_t-0.5)]$", size=11.0)
    _label(ax, 2.86, 3.72, "$u_{\\mathrm{scan},t}=\\gamma_t\\,g_{w,t}\\,u_t$", size=13.0, bold=True)
    _section(ax, 6.10, 2.37, "2  Step-size conditioning")
    _label(ax, 7.77, 2.96, "$\\Delta_t=\\mathrm{softplus}$\n$\\left(W_{\\Delta}u_t+b_{\\Delta}+\\eta I_t\\right)$", size=12.1)
    _label(ax, 7.77, 3.67, "$\\Delta_t>0$; one value per channel", size=9.7, color=MUTED)
    _line(ax, [(3.49, 1.33), (3.49, 1.90), (2.86, 1.90), (2.86, 2.10)])
    _line(ax, [(3.49, 1.90), (7.78, 1.90), (7.78, 2.10)])
    _dot(ax, 3.49, 1.90)
    _label(ax, 4.99, 4.28,
           "Shared structural control: $I_t$ is broadcast over C; $E_t=\\Vert\\bar{x}_t\\Vert_2/\\sqrt{C}$.",
           size=10.0, color=ORANGE)

    _rect(ax, 0.18, 4.72, 9.47, 1.12, fill="#EEF5FA", edge=BLUE_EDGE)
    _section(ax, 0.42, 4.95, "Selective recurrence")
    _label(ax, 4.92, 5.36,
           "$h_{t,c,j}=\\exp(\\Delta_{t,c}A_{c,j})h_{t-1,c,j}"
           "+\\Delta_{t,c}B_{t,j}u_{\\mathrm{scan},t,c}$", size=12.2)
    _label(ax, 9.40, 4.97, "$h_0=0$", size=10.0, ha="right", color=MUTED)
    _line(ax, [(2.86, 4.03), (2.86, 4.18)], color=ORANGE, dashed=True, arrow=False)
    _line(ax, [(2.86, 4.42), (2.86, 4.72)], color=ORANGE, dashed=True)
    _line(ax, [(7.78, 4.03), (7.78, 4.18)], color=ORANGE, dashed=True, arrow=False)
    _line(ax, [(7.78, 4.42), (7.78, 4.72)], color=ORANGE, dashed=True)
    # B/C and A join a parameter bundle outside the content and I paths.
    _line(ax, [(6.72, 1.01), (6.96, 1.01), (6.96, 0.40),
               (9.90, 0.40), (9.90, 5.30), (9.65, 5.30)])
    _line(ax, [(9.67, 1.01), (9.90, 1.01)], arrow=False)
    _dot(ax, 9.90, 1.01)

    _section(ax, 0.16, 6.18, "Readout, corrected-feature bypass, and output gate")
    _node(ax, 0.19, 6.61, 2.05, 0.72,
          "$y_{\\mathrm{ssm},t}$\n$\\sum_j C_{t,j}h_{t,:,j}$", kind="blue", size=10.7)
    _op(ax, 3.00, 6.97)
    _node(ax, 4.05, 6.61, 2.19, 0.72,
          "$\\odot\\,\\mathrm{SiLU}(z_t)$", kind="lavender", size=11.3)
    _node(ax, 7.04, 6.61, 1.29, 0.72, "$W_{\\mathrm{out}}$", kind="lavender", size=12.0)
    _node(ax, 9.03, 6.61, 0.61, 0.72, "$o_t$", kind="neutral", size=12.0)
    _line(ax, [(1.22, 5.84), (1.22, 6.61)])
    for a, b in ((2.24, 2.84), (3.16, 4.05), (6.24, 7.04), (8.33, 9.03)):
        _line(ax, [(a, 6.97), (b, 6.97)])
    _label(ax, 3.00, 7.76, "$D\\odot\\widetilde{x}_t$", size=11.4)
    _line(ax, [(3.00, 7.52), (3.00, 7.13)])
    _label(ax, 5.15, 7.77, "$z_t$ comes from the original $W_{\\mathrm{in}}$ split", size=9.3, color=MUTED)
    _label(ax, 7.40, 6.18, "$g_{\\min}=0.5$, $s_n=0.1$ in Full", size=9.1, color=MUTED)
    save(
        fig, "M05", "Selective scan两处调制与输出旁路", "mechanism",
        "冻结Full实现的精确机制图：xtilde经in_proj分成u与z，B_t和C_t均由调制前u生成；趋势RMS与I共同生成gamma，"
        "I另生成write因子，两者相乘后作用于同一u_scan，而非两个独立的状态位置。第二处干预是Delta logits中的eta·I；"
        "Delta同时参与历史状态衰减exp(Delta·A)与当前写入Delta·B·u_scan。扫描读出后加入D·xtilde，再乘SiLU(z)并输出投影。"
        "默认g_min=0.5、s_n=0.1。重复变量表示同一来源，图中不向B/C增加I边；这不是状态传播或抑制效果的实测证据。",
        [MODULE, PAPER], evidence="schematic",
    )


def _matrix_cell(ax, x, y, name, formula, kind):
    _node(ax, x, y, 1.86, 1.00, name + "\n" + formula, kind=kind, size=10.2)


def _m06(save):
    fig, ax = _canvas(
        "Matched controls and two factorial comparisons",
        "Configuration definitions and paired estimands only; no performance values are implied.",
        height=9.0, ymax=7.65,
    )
    _node(ax, 0.18, 0.20, 2.59, 0.90,
          "Original\noriginal four-direction SS2D shell", kind="neutral", size=9.6)
    _node(ax, 3.65, 0.20, 2.63, 0.90,
          "MatchedIdentity\nRGSP shell; scan scale = 1", kind="lavender", size=9.8)
    _node(ax, 7.17, 0.20, 2.64, 0.90,
          "Matched075\nRGSP shell; scan scale = 0.75", kind="peach", size=9.8)
    _line(ax, [(2.77, 0.65), (3.65, 0.65)])
    _line(ax, [(6.28, 0.65), (7.17, 0.65)])
    _label(ax, 3.20, 1.27, "shell difference", size=9.0, color=MUTED)
    _label(ax, 6.72, 1.27, "constant-scale difference", size=9.0, color=MUTED)

    _section(ax, 0.18, 1.82, "(a) Residual x energy")
    _label(ax, 2.55, 2.17, "I-dependent input and delta paths are off", size=9.5, color=MUTED)
    _section(ax, 5.30, 1.82, "(b) Input guidance x delta guidance")
    _label(ax, 7.63, 2.17, "R = E = 1 in all four cells", size=9.5, color=MUTED)
    for x, label in ((1.58, "$E=0$"), (3.64, "$E=1$"),
                     (6.66, "$I\\rightarrow\\Delta:0$"), (8.72, "$I\\rightarrow\\Delta:1$")):
        _label(ax, x, 2.57, label, size=10.9, bold=True)
    for y, label in ((3.28, "$R=0$"), (4.51, "$R=1$")):
        _label(ax, 0.25, y, label, size=10.4)
    _label(ax, 5.29, 3.28, "$U:0$", size=10.2)
    _label(ax, 5.29, 4.51, "$U:1$", size=10.2)

    _matrix_cell(ax, 0.65, 2.78, "Matched075", "$M_{00}$", "neutral")
    _matrix_cell(ax, 2.71, 2.78, "EnergyOnly", "$M_{01}$", "sage")
    _matrix_cell(ax, 0.65, 4.01, "Residual075", "$M_{10}$", "blue")
    _matrix_cell(ax, 2.71, 4.01, "LocalEnergy", "$M_{11}$", "peach")
    _matrix_cell(ax, 5.73, 2.78, "LocalEnergy", "$M_{00}$", "neutral")
    _matrix_cell(ax, 7.79, 2.78, "LocalEnergy + Delta", "$M_{01}$", "sage")
    _matrix_cell(ax, 5.73, 4.01, "LocalEnergy + Input", "$M_{10}$", "blue")
    _matrix_cell(ax, 7.79, 4.01, "Full", "$M_{11}$", "peach")

    _label(ax, 2.60, 5.42, "$\\mathcal{J}_{R,E}=M_{11}-M_{10}-M_{01}+M_{00}$", size=12.0)
    _label(ax, 7.66, 5.42, "$\\mathcal{J}_{U,\\Delta}=M_{11}-M_{10}-M_{01}+M_{00}$", size=12.0)
    _label(ax, 5.00, 5.98,
           "Compute each contrast within the same seed and protocol, then summarize across complete paired seeds.",
           size=9.7, color=MUTED)
    definitions = (
        (0.20, "R: residual correction", "$\\lambda_s$ enabled / fixed at 0"),
        (2.66, "E: trend-energy term", "$\\alpha$ enabled / fixed at 0"),
        (5.12, "U: I in scan drive", "I terms in both $\\gamma$ and $g_w$"),
        (7.58, "Delta: I in step logits", "I term $\\eta I$ on / off"),
    )
    for x, title, text in definitions:
        _rect(ax, x, 6.41, 2.20, 0.81, fill="#F8FAFC")
        _label(ax, x + 1.10, 6.64, title, size=9.5, bold=True)
        _label(ax, x + 1.10, 6.99, text, size=9.1, color=MUTED)
    _label(ax, 4.99, 7.45,
           "Matched075: $0.5+0.5\\,\\sigma(0)=0.75$; inactive controls are buffers. Full - LocalEnergy is not a four-cell interaction.",
           size=9.1, color=MUTED)
    save(
        fig, "M06", "结构匹配锚点与两套四格归因设计", "mechanism",
        "上方区分原始四向SS2D、同壳层MatchedIdentity及扫描输入倍率0.75的Matched075。左侧四格为Matched075、Residual075、EnergyOnly、LocalEnergy，"
        "在I路径关闭下控制残差R与趋势能量E；右侧在R=E=1时控制输入引导U与步长引导Delta，四格为LocalEnergy、+Input、+Delta与Full。"
        "两类交互均应同种子同协议计算M11−M10−M01+M00后再汇总，不能以Full−LocalEnergy代替。R关闭只固定lambda_s=0，并不等于Full-no-LRC删除趋势路径；"
        "U包含gamma与write的I项。关闭参数用buffer保持，图中没有填入成绩、预期优劣或宣称矩阵已完成。",
        [MODULE, MANIFEST, ABLATION], evidence="schematic",
    )


def build(save, data_dir):
    """Create M01-M06 and pass figures plus provenance to ``save``.

    Parameters
    ----------
    save : callable
        ``save(fig, ident, title_cn, category, caption_cn, sources,
        evidence='schematic')``. Sources are repository-relative file names.
    data_dir : object
        Unused. Mechanism schematics are definitions, not data visualizations.

    This function does not open files or run any model.  The caller owns image,
    PDF, SVG serialization and may close each figure in its save callback.
    """
    del data_dir
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }):
        for draw in (_m01, _m02, _m03, _m04, _m05, _m06):
            draw(save)

