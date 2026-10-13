"""Visualize the fixed 29-record detection snapshot, never partial epochs."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from common import (C, LABELS, METRICS, NEW_NOTE, NEW_SOURCE, current_records,
                    finish, grid, color, label, metric_value)

CORE = ["original", "matched_scale075", "local_energy", "full", "post_gate"]
KEY_METRICS = ["ap50", "ap75", "ap50_95"]


def build(save, data_dir):
    snapshot, records = current_records(data_dir)
    groups = {ds: {r["key"]: r for r in records if r["dataset"] == ds}
              for ds in ("SSDD", "SADD")}
    src = [NEW_SOURCE]
    policy = ("固定2026-10-10 10:07回收快照；仅seed0，不能作显著性或稳定性结论。"
              "SSDD为730/186/232清理后划分；SADD为795/44/44阳性图像划分。"
              "复用与历史采纳条目的恢复和审计限制保留。")

    for idx, ds in enumerate(("SSDD", "SADD"), 1):
        rows = sorted(groups[ds].values(), key=metric_value, reverse=True)
        fig, ax = plt.subplots(figsize=(8.6, 3.1 + len(rows)*.265))
        y = np.arange(len(rows))
        vals = [metric_value(r) for r in rows]
        cols = [color(r["key"]) if r["key"] in CORE else "#A1B3C4" for r in rows]
        ax.scatter(vals, y, s=68, c=cols, zorder=3)
        base = metric_value(groups[ds]["original"])
        ax.axvline(base, color=C["gray"], linestyle=(0, (3, 3)), linewidth=1,
                   label=f"Original: {base:.2f}")
        for yi, val in zip(y, vals):
            ax.annotate(f"{val:.2f}", (val, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "boxstyle": "square,pad=0.10"})
        ax.set_yticks(y, [label(r["key"]) for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(min(vals)-1.2, max(vals)+2)
        ax.set_xlabel("Test AP50:95 (%)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9)
        grid(ax)
        finish(fig, f"{ds}: all available seed-0 configurations", NEW_NOTE,
               bottom=.13, top=.87)
        fig.subplots_adjust(left=.31)
        save(fig, f"C0{idx}", f"{ds}全部已回收配置的主指标排序", "current",
             f"{ds}完整可用配置的点图；虚线为Original。排序为多配置单seed的描述性排序，不据此事后选定最终方法。"+policy, src,
             evidence="recorded_single_seed")

    for idx, ds in enumerate(("SSDD", "SADD"), 3):
        rows = sorted(groups[ds].values(), key=metric_value, reverse=True)
        vals = np.array([[metric_value(r, m) for m in KEY_METRICS] for r in rows])
        fig, ax = plt.subplots(figsize=(8.5, 2.7 + len(rows)*.28))
        im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(vals):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if v >= 68 else C["ink"])
        ax.set_xticks(range(3), [METRICS[m] for m in KEY_METRICS])
        ax.set_yticks(range(len(rows)), [label(r["key"]) for r in rows])
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.035, label="Recorded AP (%)")
        finish(fig, f"{ds}: localization-sensitive metrics", NEW_NOTE,
               bottom=.12, top=.88)
        fig.subplots_adjust(left=.31)
        save(fig, f"C0{idx}", f"{ds}三种AP指标完整热力图", "current",
             "所有单元保留百分数原口径，统一0-100色标；不做逐列归一化或显著性标记。"+policy, src,
             evidence="recorded_single_seed")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2))
    for ax, ds in zip(axes, ("SSDD", "SADD")):
        vals = [metric_value(groups[ds][k]) for k in CORE]
        bars = ax.bar(range(len(CORE)), vals, color=[color(k) for k in CORE], width=.64)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=4, fontsize=9)
        ax.set_xticks(range(len(CORE)), ["Original", "Matched075", "LocalEnergy", "Full", "Post-gate"],
                      rotation=30, ha="right")
        ax.set_ylim(0, 62)
        ax.set_ylabel("Test AP50:95 (%)")
        ax.set_title(ds)
        grid(ax, "y")
    finish(fig, "Core comparisons under each dataset protocol", NEW_NOTE, bottom=.24, top=.83)
    fig.subplots_adjust(left=.075, right=.98, wspace=.27)
    save(fig, "C05", "核心五配置分组柱状对照", "current",
         "两数据集各自协议内比较，柱轴从0开始；不跨数据集计算综合分数。"+policy, src,
         evidence="recorded_single_seed")

    common = [k for k in ["original", "matched_identity", "matched_scale075", "local_energy", "full",
                         "full_no_lrc", "const_guidance", "post_gate"]
              if all(k in groups[d] for d in groups)]
    for ident, reference, cn in (("C06", "original", "相对Original的正负差值"),
                                  ("C07", "matched_scale075", "相对同壳层Matched075的差值")):
        keys = [k for k in common if k != reference]
        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        y = np.arange(len(keys))
        for offset, ds, col, marker in [(-.13, "SSDD", C["blue"], "o"), (.13, "SADD", C["orange"], "s")]:
            baseline = metric_value(groups[ds][reference])
            v = np.array([metric_value(groups[ds][k])-baseline for k in keys])
            ax.hlines(y+offset, 0, v, color=col, alpha=.55, linewidth=2)
            ax.scatter(v, y+offset, color=col, s=47, marker=marker, label=ds, zorder=3)
            for yy, vv in zip(y+offset, v):
                ax.text(vv + (.1 if vv >= 0 else -.1), yy, f"{vv:+.2f}",
                        va="center", ha="left" if vv >= 0 else "right", fontsize=8.5)
        ax.axvline(0, color=C["ink"], linewidth=.8)
        ax.set_yticks(y, [label(k) for k in keys]); ax.invert_yaxis()
        ax.set_xlim(-6.5, 6.6); ax.set_xlabel(f"AP50:95 difference from {label(reference)} (percentage points)")
        ax.legend(loc="lower right", bbox_to_anchor=(1,1.005), ncol=2); grid(ax)
        finish(fig, f"Positive and negative differences from {label(reference)}", NEW_NOTE,
               bottom=.16, top=.82)
        fig.subplots_adjust(left=.28)
        save(fig, ident, cn, "current", "保留正负差值；点和短线只是已记录测试分数的差，不是置信区间。"+policy,
             src, evidence="recorded_single_seed")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2))
    for ax, ds in zip(axes, ("SSDD", "SADD")):
        for k in CORE:
            r = groups[ds][k]
            x, y = metric_value(r, "recall"), metric_value(r, "precision")
            ax.scatter(x, y, s=65, color=color(k), label=label(k))
            if k in {"original", "full", "post_gate"}:
                ax.annotate(label(k), (x, y), xytext=(6, 7 if k != "full" else -14),
                            textcoords="offset points", fontsize=8.5)
        ax.margins(x=.45, y=.4); ax.set_title(ds)
        ax.set_xlabel("Recorded recall (%)"); ax.set_ylabel("Recorded precision (%)"); grid(ax, "both")
    finish(fig, "Precision and recall: recorded operating points", NEW_NOTE+"\nNot precision-recall curves; no confidence-threshold sweep is available.",
           bottom=.22, top=.84)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.18, -.24), ncol=5, fontsize=8.5)
    fig.subplots_adjust(left=.08, wspace=.3, bottom=.32)
    save(fig, "C08", "Precision–Recall工作点对比（不是PR曲线）", "current",
         "只显示预定义五个核心配置的P/R工作点，无逐阈值数据，不将其连成PR曲线。"+policy, src,
         evidence="recorded_single_seed")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3))
    for ax, ds in zip(axes, ("SSDD", "SADD")):
        for r in groups[ds].values():
            k = r["key"]; x, y = metric_value(r, "ap50"), metric_value(r)
            ax.scatter(x, y, s=65 if k in CORE else 34,
                       color=color(k) if k in CORE else "#B4C0CB", zorder=3)
            if k in {"original", "full", "post_gate"}:
                ax.annotate(label(k), (x, y), xytext=(6, 6 if k != "full" else -13), textcoords="offset points", fontsize=8.5)
        ax.set_xlabel("AP50 (%)"); ax.set_ylabel("AP50:95 (%)"); ax.set_title(ds)
        ax.margins(.28); grid(ax, "both")
    finish(fig, "AP50 does not determine AP50:95", NEW_NOTE, bottom=.17, top=.84)
    fig.subplots_adjust(left=.08, wspace=.3)
    save(fig, "C09", "AP50与AP50:95的二维关系", "current",
         "显示全部已回收配置的两种AP；无回归线、无拟合或因果解释。"+policy, src,
         evidence="recorded_single_seed")

    ss = groups["SSDD"]
    fact = [
        ("(R+E) x (U+delta)", [["matched_scale075", "guidance_only"], ["local_energy", "full"]],
         ["Guidance off", "Guidance on"], ["R+E off", "R+E on"]),
        ("R x E", [["matched_scale075", "energy_only"], ["residual_scale075", "local_energy"]],
         ["E off", "E on"], ["R off", "R on"]),
        ("U x delta, fixed R+E", [["local_energy", "local_energy_delta"], ["local_energy_input", "full"]],
         ["Delta off", "Delta on"], ["U off", "U on"]),
    ]
    deltas = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6))
    for ax, (title, keys, xlab, ylab) in zip(axes, fact):
        vals = np.array([[metric_value(ss[k]) for k in row] for row in keys])
        d = vals[1, 1]-vals[1, 0]-vals[0, 1]+vals[0, 0]; deltas.append(d)
        ax.imshow(vals, cmap="Blues", vmin=35, vmax=50, aspect="equal")
        for (i, j), val in np.ndenumerate(vals):
            ax.text(j, i, f"{val:.2f}%", va="center", ha="center", fontsize=14,
                    color="white" if val > 44 else C["ink"])
        ax.set_xticks([0, 1], xlab); ax.set_yticks([0, 1], ylab)
        ax.set_title(title, fontsize=11, pad=13)
        ax.set_xlabel(f"Interaction = {d:+.3f} pp", labelpad=14)
        ax.tick_params(length=0)
    finish(fig, "Three different factorial comparisons", "SSDD | seed 0 | values: test AP50:95 (%) | descriptive non-additivity only\nLocalEnergy retains historical adoption and recovery limitations.", bottom=.2, top=.78)
    fig.subplots_adjust(left=.07, right=.98, wspace=.62)
    save(fig, "C10", "三组严格定义的四格消融热力图", "current",
         "分别使用正确锚点计算组合级、R×E及固定R+E下U×Δ。不能用Full−DS替代因素交互，也不声称统计显著或内部物理机制。"+policy,
         src, evidence="recorded_single_seed")

    expected = snapshot["contrasts_ap50_95_percentage_points"]
    for val, key in zip(deltas, ["combined_RE_guidance", "LRC_energy", "input_delta"]):
        if not np.isclose(val, expected[key], atol=1e-9): raise ValueError("Factorial recomputation mismatch")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(3)
    ax.barh(y, deltas, color=[C["teal"] if d >= 0 else C["red"] for d in deltas], height=.52)
    for yy, d in zip(y, deltas):
        ax.text(d+(.15 if d >= 0 else -.15), yy, f"{d:+.3f}", ha="left" if d >= 0 else "right", va="center")
    ax.axvline(0, color=C["ink"], linewidth=.8)
    ax.set_yticks(y, [f[0] for f in fact]); ax.invert_yaxis()
    ax.set_xlim(-8, 2); ax.set_xlabel("Factorial difference in AP50:95 (percentage points)"); grid(ax)
    finish(fig, "Non-additivity depends on the factorial question", "SSDD | seed 0 | no cross-seed uncertainty or significance estimate", bottom=.2, top=.82)
    fig.subplots_adjust(left=.3)
    save(fig, "C11", "因素交互的正负效应条形图", "current",
         "数值由四格原值重新计算并与源JSON核对；无误差条是因为只有一个seed，不是因为误差为零。"+policy,
         src, evidence="recorded_single_seed")

    def horizontal_subset(ident, keys, title, cn, caption):
        vals = [metric_value(ss[k]) for k in keys]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(range(len(keys)), vals, height=.56, color=[color(k) for k in keys])
        for yy, v in enumerate(vals): ax.text(v+.5, yy, f"{v:.2f}", va="center")
        ax.set_yticks(range(len(keys)), [label(k) for k in keys]); ax.invert_yaxis()
        ax.set_xlim(0, 52); ax.set_xlabel("Test AP50:95 (%)"); grid(ax)
        finish(fig, title, NEW_NOTE, bottom=.18, top=.84); fig.subplots_adjust(left=.32)
        save(fig, ident, cn, "current", caption+policy, src, evidence="recorded_single_seed")

    horizontal_subset("C12", ["local_energy", "local_energy_input", "local_energy_delta", "full", "post_gate"],
                      "Where the guidance is applied", "扫描输入、步长与输出门控消融",
                      "Post-gate关闭I→Δ且将相对倍率移至scan输出；因此Full vs Post-gate不是纯位置单因素对照。")
    horizontal_subset("C13", ["full", "global", "lsgm_trend_only", "lsgm_std_only", "lsgm_point_only"],
                      "Local-statistics guidance alternatives", "LSGM局部统计组成对比",
                      "完整局部统计、全局及三个单分量均展示；不把当前std-only表现推广为普遍规律。")
    horizontal_subset("C14", ["full_p2", "full", "full_p4"],
                      "Backbone placement: P2, P3 and P4", "模块插入层级P2/P3/P4敏感性",
                      "P3是当前设计位置，不是本轮单seed最高分；跨层通道和初始化可能不同，P2观测到105轮、其余150轮。")

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 5.2))
    for ax, ds in zip(axes, ("SSDD", "SADD")):
        metrics = list(METRICS)
        y = np.arange(len(metrics))
        for k, off, col in [("full", -.13, C["teal"]), ("full_no_lrc", .13, C["orange"])]:
            v = [metric_value(groups[ds][k], m) for m in metrics]
            ax.scatter(v, y+off, s=45, c=col, label=label(k))
        for yy, m in enumerate(metrics):
            a,b = (metric_value(groups[ds][k], m) for k in ["full", "full_no_lrc"])
            ax.plot([a,b], [yy-.13,yy+.13], color=C["gray"], linewidth=.8, alpha=.6)
        ax.set_yticks(y, [METRICS[m].replace(" (%)", "") for m in metrics]); ax.invert_yaxis()
        ax.set_xlim(20, 102); ax.set_title(ds); ax.set_xlabel("Recorded metric (%)"); grid(ax)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.54,.085), ncol=2, fontsize=9)
    finish(fig, "Full and no-LRC across five metrics", NEW_NOTE+"\nno-LRC also changes the energy/guidance inputs; it is not only lambda = 0.", bottom=.26, top=.84)
    fig.subplots_adjust(left=.105, wspace=.38)
    save(fig, "C15", "Full与no-LRC的五指标哑铃对照", "current",
         "no-LRC跳过trend卷积并改变后续输入，不作单个残差标量的机制归因。连接线只帮助辨识两点，不代表置信区间。"+policy,
         src, evidence="recorded_single_seed")

    ordered = sorted(records, key=lambda r:(r["dataset"], -metric_value(r)))
    fig, ax = plt.subplots(figsize=(10.8, 12.3)); ax.axis("off")
    columns = ["Dataset / configuration", "AP50", "AP75", "AP50:95", "P", "R"]
    cells = [[r["dataset"]+" / "+label(r["key"])] + [f"{metric_value(r,m):.2f}" for m in ["ap50","ap75","ap50_95","precision","recall"]]
             for r in ordered]
    table = ax.table(cellText=cells, colLabels=columns, colWidths=[.46,.108,.108,.108,.108,.108],
                     cellLoc="right", colLoc="center", bbox=[0,0,1,1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row,col), cell in table.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(.7)
        if row == 0: cell.set_facecolor(C["lightblue"]); cell.set_text_props(weight="medium")
        else: cell.set_facecolor("#F4F7F9" if row%2==0 else "white")
        if col == 0: cell.set_text_props(ha="left")
    finish(fig, "All 29 recorded detection results", NEW_NOTE+"\nAll metrics are percentages. Full provenance is provided in the source JSON and gallery manifest.", bottom=.07, top=.94)
    fig.subplots_adjust(left=.03,right=.97)
    save(fig, "C16", "29项检测结果的完整数据表格图", "current",
         "用于逐行挑选或附录核数；保留所有已回收结果，缺失配置未用0补齐。"+policy, src,
         evidence="recorded_single_seed")
