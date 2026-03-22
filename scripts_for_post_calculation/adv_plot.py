#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===========================
# GLOBAL FONT SETTINGS (Elsevier ENERGY)
# ===========================
# =========================================================
# GLOBAL FONT SETTINGS (Elsevier ENERGY style)
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 14,            # global base font
    "axes.labelsize": 15,       # x/y labels
    "axes.titlesize": 16,       # subplot titles
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "mathtext.fontset": "dejavuserif",   # matches serif font
    "figure.titlesize": 18,
    "pdf.fonttype": 42,         # clean text in exported PDF
    "ps.fonttype": 42
})
# =========================================================
# GLOBAL DAMPING LIST (required for all plots)
# =========================================================
damping_list  = [0.10, 0.15, 0.20, 0.25, 0.30]

# ===========================
# LOAD SUMMARIES
# ===========================
df_IC  = pd.read_csv("summary_IC.csv")
df_CC  = pd.read_csv("summary_CC.csv")
df_CPC = pd.read_csv("summary_CPC.csv")

SPD_list = [1.0, 2.0, 3.0]


# ===========================
# COLOR + MARKER DEFINITIONS
# ===========================
high_colors = ["#e73a36", "#2e974e", "#7262ac",
               "#547ac0", "#e25508", "#8c564b"]  # CPC

low_colors  = ["#ED817E", "#7FCB82", "#C4A9DC",
               "#d2ddf1", "#f5b0af", "#c49c94"]  # CC

colors_CPC = {1.0: high_colors[0],
              2.0: high_colors[1],
              3.0: high_colors[2]}

colors_CC = {1.0: low_colors[0],
             2.0: low_colors[1],
             3.0: low_colors[2]}

markers_filled = {1.0: "s", 2.0: "D", 3.0: "^"}
markers_hollow = {1.0: "s", 2.0: "D", 3.0: "^"}

ic_color = "#5f5f5f"
tic_color  = "#999A9E"

SUBLABEL_FONTSIZE = 15

def add_sublabel(ax, label):
    ax.text(
        0.02, 0.90, label,
        transform=ax.transAxes,
        fontsize=SUBLABEL_FONTSIZE,
        fontweight="bold"
    )

# =========================================================
# AUTO Y-LIMITER
# =========================================================
def auto_ylim(ax, y_all, margin=0.15):
    """
    Set y-limits with margin AND return (ymin, ymax)
    so that callers can unify y-limits later.
    """

    if y_all is None or len(y_all) == 0:
        return None, None

    ymin = min(y_all)
    ymax = max(y_all)

    # handle constant dataset
    if ymin == ymax:
        ymin -= 0.1
        ymax += 0.1

    span = ymax - ymin
    ymin -= margin * span
    ymax += margin * span

    ax.set_ylim(ymin, ymax)

    return ymin, ymax


# =========================================================
# SMART TICKS
# =========================================================
def set_ticks(ax, is_phase=False):
    if is_phase:
        ax.set_yticks([0,45,90,135,180])
        return

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        return

    nice_steps = np.array([
        0.01, 0.0125, 0.025, 0.03, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5,
        1, 2, 2.5, 3, 4, 5,
        10, 15, 20, 25, 30, 40, 50, 100, 125, 150, 200
    ])

    target = span / 5
    step = nice_steps[np.argmin(np.abs(nice_steps - target))]

    tmin = step * np.ceil(ymin / step)
    tmax = step * np.floor(ymax / step)
    ticks = np.arange(tmin, tmax + 0.1*step, step)

    if len(ticks) < 4.5:
        idx = np.where(nice_steps < step)[0]
        for s in nice_steps[idx][::-1]:
            tmin = s*np.ceil(ymin/s)
            tmax = s*np.floor(ymax/s)
            new = np.arange(tmin, tmax+0.1*s, s)
            if 5 <= len(new) <= 7:
                ax.set_yticks(new)
                return

    if len(ticks) > 7:
        idx = np.where(nice_steps > step)[0]
        for s in nice_steps[idx]:
            tmin = s*np.ceil(ymin/s)
            tmax = s*np.floor(ymax/s)
            new = np.arange(tmin, tmax+0.1*s, s)
            if 5 <= len(new) <= 7:
                ax.set_yticks(new)
                return

    ax.set_yticks(ticks)

def set_x_ticks(ax):
    xmin, xmax = ax.get_xlim()

    # Apply ±0.02 margin
    xmin -= 0.01
    xmax += 0.01
    ax.set_xlim(xmin, xmax)

    # Fixed ticks every 0.05
    ticks = np.arange(0, 1.0, 0.05)
    ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
    ax.set_xticks(ticks)

# ===========================
# Polynomial fit helper
# ===========================
def polyfit_curve(x, y):
    if len(x) < 3:
        return None, None
    coeff = np.polyfit(x, y, 2)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = np.polyval(coeff, xx)
    return xx, yy


# ===========================
# MAIN PLOT FUNCTION
# ===========================

# =========================================================
# GRAPHIC ABSTRACT — PERFORMANCE + OAOP IMPROVEMENT
# (STYLE MATCHES plot_OAOP_3x2, WITH 2×IC)
# =========================================================
def plot_graphic_abstract_performance_only(savepath):

    fig, (ax, ax_bar) = plt.subplots(
        2, 1, figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------
    x_ic = df_IC["damping"]
    y_ic = df_IC["OAOP_total"]
    y_2ic = 2.0 * y_ic

    df_cpc = df_CPC[(df_CPC["SP_U"] == 0.7) & (df_CPC["SP_D"] == 2.0)]
    df_cc  = df_CC [(df_CC ["SP_U"] == 0.5) & (df_CC ["SP_D"] == 1.0)]

    # =====================================================
    # TOP PANEL — OAOP_total
    # =====================================================
    # IC
    (h_ic,) = ax.plot(
        x_ic, y_ic,
        marker="o", markersize=7,
        linestyle="None",
        color=ic_color,
        markerfacecolor=ic_color,
        markeredgecolor=ic_color,
        label="IC"
    )
    xx, yy = polyfit_curve(x_ic, y_ic)
    if xx is not None:
        ax.plot(xx, yy, color=ic_color, linewidth=2)

    # 2×IC  (same as plot_OAOP_3x2)
    (h_2ic,) = ax.plot(
        x_ic, y_2ic,
        marker="o", markersize=8,
        linestyle="None",
        color=ic_color,
        markerfacecolor="none",
        markeredgecolor=ic_color,
        label="2×IC"
    )
    xx, yy = polyfit_curve(x_ic, y_2ic)
    if xx is not None:
        ax.plot(xx, yy, linestyle="--",
                color=ic_color, linewidth=2.2)

    # C-P-C (optimal)
    (h_cpc,) = ax.plot(
        df_cpc["damping"], df_cpc["OAOP_total"],
        marker=markers_filled[2.0], markersize=7,
        linestyle="None",
        color=colors_CPC[2.0],
        markerfacecolor=colors_CPC[2.0],
        markeredgecolor=colors_CPC[2.0],
        label="C-P-C (SPU, SPD) = (0.7, 2)"
    )
    xx, yy = polyfit_curve(df_cpc["damping"], df_cpc["OAOP_total"])
    if xx is not None:
        ax.plot(xx, yy, color=colors_CPC[2.0], linewidth=2)

    # C-C (optimal)
    (h_cc,) = ax.plot(
        df_cc["damping"], df_cc["OAOP_total"],
        marker=markers_hollow[1.0], markersize=8,
        linestyle="None",
        color=colors_CC[1.0],
        markerfacecolor="none",
        markeredgecolor=colors_CC[1.0],
        label="C-C (SPU, SPD) = (0.5, 1)"
    )
    xx, yy = polyfit_curve(df_cc["damping"], df_cc["OAOP_total"])
    if xx is not None:
        ax.plot(xx, yy, linestyle="--",
                color=colors_CC[1.0], linewidth=2)

    ax.set_ylabel(r"$\mathit{OAOP}_{\mathrm{Total}}$")
    ax.grid(True, linestyle="--", alpha=0.3)
    set_ticks(ax)

    ax.legend(
        handles=[h_ic, h_2ic, h_cpc, h_cc],
        frameon=False, fontsize=11
    )

    # =====================================================
    # BOTTOM PANEL — ΔOAOP (%)  (CSV BASED)
    # =====================================================
    imp_cpc = df_cpc["imp_power_total"]
    imp_cc  = df_cc ["imp_power_total"]

    bar_width = 0.018

    ax_bar.bar(
        df_cpc["damping"] - bar_width/2,
        imp_cpc,
        width=bar_width,
        color=colors_CPC[2.0],
        edgecolor=colors_CPC[2.0],
        label="C-P-C"
    )

    ax_bar.bar(
        df_cc["damping"] + bar_width/2,
        imp_cc,
        width=bar_width,
        facecolor="none",
        edgecolor=colors_CC[1.0],
        linewidth=1.8,
        hatch="///",
        label="C-C"
    )

    ax_bar.axhline(0, color="k", linewidth=1.0)
    ax_bar.set_ylabel(r"$\Delta \mathit{OAOP}_{\mathrm{Total}}$ (%)")
    ax_bar.set_xlabel(r"Damping ratio $\zeta$")
    ax_bar.grid(True, linestyle="--", alpha=0.3)

    set_x_ticks(ax_bar)
    set_ticks(ax_bar)
   
    ax_bar.legend(frameon=False, fontsize=10)

    # =====================================================
    # FINALIZE
    # =====================================================
    fig.tight_layout()
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved graphic abstract (with 2×IC): {savepath}")



plot_graphic_abstract_performance_only("Graphic_Abstract_Performance.png")




# =========================================================
# MAIN PLOT FUNCTION (Unified Top Legend)
# =========================================================
def plot_for_SPU(spu_value, filename):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.83)
    
    sublabs = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    sid = 0

    rows = ["UC", "DC", "Total"]

    legend_handles = []
    legend_labels  = []
    added = set()    # to prevent duplicates

    for i, row in enumerate(rows):
        axP = axes[i][0]   # Peak Power
        axE = axes[i][1]   # Peak Efficiency

        # ---- SUBLABELS ----
        add_sublabel(axP, sublabs[sid]); sid += 1
        add_sublabel(axE, sublabs[sid]); sid += 1


        yP_all = []
        yE_all = []
        # -----------------------------
        # IC (ALL ROWS)
        # -----------------------------
        x_ic = df_IC["damping"]

        if row == "UC":
            yP_ic = df_IC["peak_power_uc"]
            yE_ic = df_IC["peak_eff_uc"]
        elif row == "DC":
            yP_ic = df_IC["peak_power_uc"]
            yE_ic = df_IC["peak_eff_uc"]
        else:
            yP_ic = df_IC["peak_power_total"]
            yE_ic = df_IC["peak_eff_total"]

        # IC marker (solid black)
        (h_ic,) = axP.plot(
            x_ic, yP_ic,
            marker="o", markersize=7,
            linestyle="None",
            color=ic_color,
            markerfacecolor=ic_color,
            markeredgecolor=ic_color
        )
        xx, yy = polyfit_curve(x_ic, yP_ic)
        if xx is not None:
            axP.plot(xx, yy, color=ic_color, linewidth=2)

        # Efficiency IC
        axE.plot(
            x_ic, yE_ic,
            marker="o", markersize=7,
            linestyle="None",
            color=ic_color,
            markerfacecolor=ic_color,
            markeredgecolor=ic_color
        )
        xx, yy = polyfit_curve(x_ic, yE_ic)
        if xx is not None:
            axE.plot(xx, yy, color=ic_color, linewidth=2)

        if "IC" not in added:
            legend_handles.append(h_ic)
            legend_labels.append("IC")
            added.add("IC")

        yP_all.extend(list(yP_ic))
        yE_all.extend(list(yE_ic))

        # ================================
        # CPC + CC
        # ================================
        for spd in SPD_list:

            # ---------------- CPC ----------------
            df_sub = df_CPC[(df_CPC["SP_U"] == spu_value) &
                            (df_CPC["SP_D"] == spd)]
            if len(df_sub) > 0:

                x = df_sub["damping"]
                if row == "UC":
                    yP = df_sub["peak_power_uc"]
                    yE = df_sub["peak_eff_uc"]
                elif row == "DC":
                    yP = df_sub["peak_power_dc"]
                    yE = df_sub["peak_eff_dc"]
                else:
                    yP = df_sub["peak_power_total"]
                    yE = df_sub["peak_eff_total"]

                # CPC marker
                (h,) = axP.plot(
                    x, yP,
                    marker=markers_filled[spd], markersize=7,
                    linestyle="None",
                    color=colors_CPC[spd],
                    markerfacecolor=colors_CPC[spd],
                    markeredgecolor=colors_CPC[spd],
                )
                xx, yy = polyfit_curve(x, yP)
                if xx is not None:
                    axP.plot(xx, yy, color=colors_CPC[spd], linewidth=2)

                axE.plot(
                    x, yE,
                    marker=markers_filled[spd], markersize=7,
                    linestyle="None",
                    color=colors_CPC[spd],
                    markerfacecolor=colors_CPC[spd],
                    markeredgecolor=colors_CPC[spd],
                )
                xx, yy = polyfit_curve(x, yE)
                if xx is not None:
                    axE.plot(xx, yy, color=colors_CPC[spd], linewidth=2)

                if f"C-P-C{spd}" not in added:
                    legend_handles.append(h)
                    legend_labels.append(f"C-P-C SPD={spd}")
                    added.add(f"C-P-C{spd}")

                yP_all.extend(list(yP))
                yE_all.extend(list(yE))

            # ---------------- CC ----------------
            df_sub = df_CC[(df_CC["SP_U"] == spu_value) &
                           (df_CC["SP_D"] == spd)]
            if len(df_sub) > 0:

                x = df_sub["damping"]
                if row == "UC":
                    yP = df_sub["peak_power_uc"]
                    yE = df_sub["peak_eff_uc"]
                elif row == "DC":
                    yP = df_sub["peak_power_dc"]
                    yE = df_sub["peak_eff_dc"]
                else:
                    yP = df_sub["peak_power_total"]
                    yE = df_sub["peak_eff_total"]

                # CC marker (hollow)
                (h,) = axP.plot(
                    x, yP,
                    marker=markers_hollow[spd], markersize=8,
                    linestyle="None",
                    color=colors_CC[spd],
                    markerfacecolor="none",
                    markeredgecolor=colors_CC[spd],
                )
                xx, yy = polyfit_curve(x, yP)
                if xx is not None:
                    axP.plot(xx, yy,
                             color=colors_CC[spd],
                             linestyle="--", linewidth=2)

                axE.plot(
                    x, yE,
                    marker=markers_hollow[spd], markersize=8,
                    linestyle="None",
                    color=colors_CC[spd],
                    markerfacecolor="none",
                    markeredgecolor=colors_CC[spd],
                )
                xx, yy = polyfit_curve(x, yE)
                if xx is not None:
                    axE.plot(xx, yy,
                             color=colors_CC[spd],
                             linestyle="--", linewidth=2)

                if f"C-C{spd}" not in added:
                    legend_handles.append(h)
                    legend_labels.append(f"C-C SPD={spd}")
                    added.add(f"C-C{spd}")

                yP_all.extend(list(yP))
                yE_all.extend(list(yE))

        # 2×IC only for TOTAL
        if row == "Total":
            # -------- 2×IC baseline (ONLY local legend) --------
            yP_2x = 2 * df_IC["peak_power_total"]

            (h2,) = axP.plot(
                x_ic, yP_2x,
                marker="o", markersize=8,
                linestyle="None",
                color=tic_color,
                markerfacecolor="none",
                markeredgecolor=tic_color,
            )
            xx, yy = polyfit_curve(x_ic, yP_2x)
            if xx is not None:
                axP.plot(
                   xx, yy,
                    linestyle="-.", color=tic_color, linewidth=2.2
                )

            # ⬅ Local legend inside subplot
            axP.legend(
                [h2], ["2×IC"],
                loc="upper right",
                frameon=False,
                fontsize=10
            )

            yP_all.extend(list(yP_2x))


        # formatting
        auto_ylim(axP, yP_all)
        auto_ylim(axE, yE_all)
        set_x_ticks(axP)
        set_ticks(axP)   # y axis (keep smart)
        set_x_ticks(axE)
        set_ticks(axE)   # y axis (keep smart)


        axP.grid(True, linestyle="--", alpha=0.3)
        axE.grid(True, linestyle="--", alpha=0.3)

        # ====== AXIS LABELS (Updated: no "Peak", proper subscripts, italic zeta) ======

        if row == "UC":
            P_label = r"$P^{\ast}_{\mathrm{UC}}$"
            E_label = r"$\eta_{\mathrm{UC}}$ (%)"
        elif row == "DC":
            P_label = r"$P^{\ast}_{\mathrm{DC}}$"
            E_label = r"$\eta_{\mathrm{DC}}$ (%)"
        else:
            P_label = r"$P^{\ast}_{\mathrm{Total}}$"
            E_label = r"$\eta_{\mathrm{Total}}$ (%)"

        axP.set_xlabel(r"Damping ratio $\zeta$")
        axE.set_xlabel(r"Damping ratio $\zeta$")

        # Power in mW/m
        axP.set_ylabel(P_label)

        # Efficiency dimensionless
        axE.set_ylabel(E_label)



    # ======================================================
    # Unified Legend at TOP center
    # ======================================================
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        frameon=False,
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(filename, dpi=900, bbox_inches="tight")
    print(f"✔ Saved {filename}")


# ===========================
# Generate SPU=0.5 and SPU=0.7
# ===========================
plot_for_SPU(0.5, "Peak_Power_Eff_SP_U_0.5.png")
plot_for_SPU(0.7, "Peak_Power_Eff_SP_U_0.7.png")


# ============================================================
# OPTION 6 – Jitter Scatter (clean, no overlap, IC shown everywhere)
# ============================================================
import numpy as np

def plot_UR_jitter(spu_value, filename):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.88)

    rows = ["UC", "DC", "Total"]
    sublabs = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    sid = 0

    damping_levels = sorted(df_CPC["damping"].unique())

    # jitter magnitudes
    jitter_IC  = 0.004
    jitter_CPC = 0.007
    jitter_CC  = 0.007

    for i, row in enumerate(rows):

        axP = axes[i][0]   # UR at peak power
        axE = axes[i][1]   # UR at peak efficiency

        add_sublabel(axP, sublabs[sid]); sid += 1
        add_sublabel(axE, sublabs[sid]); sid += 1

        # === choose UR columns for this row ===
        if row == "UC":
            colP = "UR_peak_power_uc"
            colE = "UR_peak_eff_uc"
        elif row == "DC":
            colP = "UR_peak_power_dc"
            colE = "UR_peak_eff_dc"
        else:
            colP = "UR_peak_power_total"
            colE = "UR_peak_eff_total"

        # ==========================================
        # 1. IC — shown for UC, DC, TOTAL
        # ==========================================
        df_ic_sorted = df_IC.sort_values("damping")
        y_ic = df_ic_sorted["damping"]

        if row in ["UC", "DC"]:
            # IC has no DC → UC baseline used
            xP_ic = df_ic_sorted["UR_peak_power_total"]
            xE_ic = df_ic_sorted["UR_peak_eff_total"]
        else:
            xP_ic = df_ic_sorted["UR_peak_power_total"]
            xE_ic = df_ic_sorted["UR_peak_eff_total"]

        axP.scatter(
            xP_ic,
            y_ic + np.random.uniform(-jitter_IC, jitter_IC, len(y_ic)),
            s=70,
            color=ic_color,
            marker="o"
        )
        axE.scatter(
            xE_ic,
            y_ic + np.random.uniform(-jitter_IC, jitter_IC, len(y_ic)),
            s=70,
            color=ic_color,
            marker="o"
        )

        # ==========================================
        # 2. CPC + CC — jittered to avoid overlap
        # ==========================================
        for spd in SPD_list:

            # CPC
            df_sub1 = df_CPC[
                (df_CPC["SP_U"] == spu_value) &
                (df_CPC["SP_D"] == spd)
            ]

            for _, r in df_sub1.iterrows():
                yjit = r["damping"] + np.random.uniform(-jitter_CPC, jitter_CPC)
                axP.scatter(
                    r[colP], yjit,
                    s=70,
                    marker=markers_filled[spd],
                    color=colors_CPC[spd]
                )
                axE.scatter(
                    r[colE], yjit,
                    s=70,
                    marker=markers_filled[spd],
                    color=colors_CPC[spd]
                )

            # CC
            df_sub2 = df_CC[
                (df_CC["SP_U"] == spu_value) &
                (df_CC["SP_D"] == spd)
            ]

            for _, r in df_sub2.iterrows():
                yjit = r["damping"] + np.random.uniform(-jitter_CC, jitter_CC)
                axP.scatter(
                    r[colP], yjit,
                    s=80,
                    marker=markers_hollow[spd],
                    facecolors="none",
                    edgecolors=colors_CC[spd],
                    linewidths=1.8
                )
                axE.scatter(
                    r[colE], yjit,
                    s=80,
                    marker=markers_hollow[spd],
                    facecolors="none",
                    edgecolors=colors_CC[spd],
                    linewidths=1.8
                )

        # ==========================================
        # Formatting + unified x-axis limits
        # ==========================================
        for ax in [axP, axE]:
            ax.set_ylabel(r"Damping ratio $\zeta$")
            ax.set_yticks(damping_levels)
            ax.grid(True, ls="--", alpha=0.3)
            ax.set_xlim(2, 16)      # <---- unified x-limit


        # axis labels
        if row == "UC":
            axP.set_xlabel(r"$U_r$ at peak $P^{\ast}_{\mathrm{UC}}$")
            axE.set_xlabel(r"$U_r$ at peak $\eta_{\mathrm{UC}}$")
        elif row == "DC":
            axP.set_xlabel(r"$U_r$ at peak $P^{\ast}_{\mathrm{DC}}$")
            axE.set_xlabel(r"$U_r$ at peak $\eta_{\mathrm{DC}}$")
        else:
            axP.set_xlabel(r"$U_r$ at peak $P^{\ast}_{\mathrm{Total}}$")
            axE.set_xlabel(r"$U_r$ at peak $\eta_{\mathrm{Total}}$")


    # ==========================================================
    # Unified Legend
    # ==========================================================
    fig.legend(
        handles=[
            plt.Line2D([], [], color=ic_color, marker="o", ls="None"),
            *[
                plt.Line2D([], [], color=colors_CPC[s], marker=markers_filled[s], ls="None")
                for s in SPD_list
            ],
            *[
                plt.Line2D([], [], color=colors_CC[s], marker=markers_hollow[s], ls="None")
                for s in SPD_list
            ],
        ],
        labels=["IC"] +
               [f"C-P-C SPD={s}" for s in SPD_list] +
               [f"C-C SPD={s}" for s in SPD_list],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        frameon=False,
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=900, bbox_inches="tight")
    print(f"✔ Saved jitter plot: {filename}")


# ============================================================
# Run the jitter plot for SPU = 0.5 and 0.7
# ============================================================
plot_UR_jitter(0.5, "UR_jitter_SP_U_0.5.png")
plot_UR_jitter(0.7, "UR_jitter_SP_U_0.7.png")


# =========================================================
# 1×2 SYSTEM PEAK POWER DENSITY PLOT (IC + CPC + CC)
# =========================================================
def plot_PD_1x2(savepath):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    SPUs = [0.5, 0.7]

    legend_handles = []
    legend_labels  = []

    # For unified y-limit
    y_all = []

    # --------------------------------------------------------
    # FIRST PASS: Collect all y-values from both subplots
    # --------------------------------------------------------
    for col, spu_value in enumerate(SPUs):


        # ----- CPC & CC for each SPD -----
        for spd in SPD_list:

            df_sub = df_CPC[(df_CPC["SP_U"] == spu_value) &
                            (df_CPC["SP_D"] == spd)]
            if len(df_sub) > 0:
                y_all.extend(df_sub["PD_peak_total"].tolist())

            df_sub = df_CC[(df_CC["SP_U"] == spu_value) &
                           (df_CC["SP_D"] == spd)]
            if len(df_sub) > 0:
                y_all.extend(df_sub["PD_peak_total"].tolist())

    # --------------------------------------------------------
    # UNIFIED Y-LIMIT (0 → ymax + margin)
    # --------------------------------------------------------
    if len(y_all) > 0:
        ymax   = max(y_all)
        margin = 0.12 * ymax
        y_low, y_high = 0, ymax + margin
    else:
        y_low, y_high = 0, 1

    # ========================================================
    # SECOND PASS: ACTUAL PLOTTING
    # ========================================================
    for col, spu_value in enumerate(SPUs):

        ax = axes[col]



        # -----------------------
        # CPC and CC
        # -----------------------
        for spd in SPD_list:

            # ---- CPC ----
            df_sub = df_CPC[(df_CPC["SP_U"] == spu_value) &
                            (df_CPC["SP_D"] == spd)]

            if len(df_sub) > 0:
                x = df_sub["damping"]
                y = df_sub["PD_peak_total"]

                (h,) = ax.plot(
                    x, y,
                    marker=markers_filled[spd], markersize=7,
                    linestyle="None",
                    color=colors_CPC[spd],
                    markerfacecolor=colors_CPC[spd],
                    markeredgecolor=colors_CPC[spd],
                )

                xx, yy = polyfit_curve(x, y)
                if xx is not None:
                    ax.plot(xx, yy,
                            color=colors_CPC[spd],
                            linewidth=2)

                if col == 0:
                    legend_handles.append(h)
                    legend_labels.append(f"C-P-C SPD={spd}")

            # ---- CC ----
            df_sub = df_CC[(df_CC["SP_U"] == spu_value) &
                           (df_CC["SP_D"] == spd)]

            if len(df_sub) > 0:
                x = df_sub["damping"]
                y = df_sub["PD_peak_total"]

                (h,) = ax.plot(
                    x, y,
                    marker=markers_hollow[spd], markersize=8,
                    linestyle="None",
                    color=colors_CC[spd],
                    markerfacecolor="none",
                    markeredgecolor=colors_CC[spd],
                )

                xx, yy = polyfit_curve(x, y)
                if xx is not None:
                    ax.plot(xx, yy,
                            color=colors_CC[spd],
                            linestyle="--",
                            linewidth=2)

                if col == 0:
                    legend_handles.append(h)
                    legend_labels.append(f"C-C SPD={spd}")

        # -----------------------
        # Axes formatting
        # -----------------------
        ax.set_ylim(y_low, y_high)
        set_ticks(ax)
        set_x_ticks(ax)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel(r"Damping ratio $\zeta$")
        ax.set_title(f"SPU = {spu_value}", fontsize=14)
        ax.set_ylabel(r"$PPD$")

    # --------------------------------------------------------
    # Unified Legend
    # --------------------------------------------------------
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        frameon=False,
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved {savepath}")



# =========================================================
# Generate 3×2 Peak Density Figure
# =========================================================
plot_PD_1x2("Peak_PD_1x2.png")


# =========================================================
# 3×2 OAOP Plot for SPU = 0.5 and 0.7
# Matches Peak/Eff/PD style, markers, legends, layout
# =========================================================

# =========================================================
# ============== 3×2 OAOP (UC / DC / TOTAL) ===============
# =========================================================

def plot_OAOP_3x2(savepath):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.83)

    rows = ["UC", "DC", "Total"]
    SPUs = [0.5, 0.7]

    legend_handles = []
    legend_labels  = []
    added = set()

    sublab_matrix = [
    ["(a)", "(b)"],
    ["(c)", "(d)"],
    ["(e)", "(f)"]
    ]

    sid = 0

    # To unify row-wise y-limits
    row_ymins = {0: [], 1: [], 2: []}
    row_ymaxs = {0: [], 1: [], 2: []}

    for col, spu_value in enumerate(SPUs):

        for r, row in enumerate(rows):
            ax = axes[r][col]

            # ---------- Add sublabel ----------
            add_sublabel(ax, sublab_matrix[r][col])
            sid += 1

            y_all = []

            # ===================================================
            # 1) IC LINE — ALWAYS PRESENT FOR ALL ROWS
            # ===================================================
            x_ic = df_IC["damping"]
            y_ic = df_IC["OAOP_total"]   # IC only has TOTAL

            # Scatter for IC
            (h_ic,) = ax.plot(
                x_ic, y_ic,
                marker="o", markersize=7,
                linestyle="None",
                color=ic_color,
                markerfacecolor=ic_color,
                markeredgecolor=ic_color
            )

            # Fit curve for IC
            xx, yy = polyfit_curve(x_ic, y_ic)
            if xx is not None:
                ax.plot(xx, yy, color=ic_color, linewidth=2)

            y_all.extend(list(y_ic))

            if (r == 0) and (col == 0) and ("IC" not in added):
                legend_handles.append(h_ic)
                legend_labels.append("IC")
                added.add("IC")

            # ===================================================
            # 2) CPC + CC
            # ===================================================
            for spd in SPD_list:

                # ---------- CPC ----------
                sub = df_CPC[(df_CPC["SP_U"] == spu_value) &
                             (df_CPC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[f"OAOP_{row.lower()}"] if row != "Total" else sub["OAOP_total"]

                    (h,) = ax.plot(
                        x, y,
                        marker=markers_filled[spd], markersize=7,
                        linestyle="None",
                        color=colors_CPC[spd],
                        markerfacecolor=colors_CPC[spd],
                        markeredgecolor=colors_CPC[spd]
                    )

                    # Fit CPC
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy, color=colors_CPC[spd], linewidth=2)

                    y_all.extend(list(y))

                    if (r == 0) and (col == 0) and (f"CPC{spd}" not in added):
                        legend_handles.append(h)
                        legend_labels.append(f"C-P-C SPD={spd}")
                        added.add(f"CPC{spd}")

                # ---------- CC ----------
                sub = df_CC[(df_CC["SP_U"] == spu_value) &
                             (df_CC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[f"OAOP_{row.lower()}"] if row != "Total" else sub["OAOP_total"]

                    (h,) = ax.plot(
                        x, y,
                        marker=markers_hollow[spd], markersize=8,
                        linestyle="None",
                        color=colors_CC[spd],
                        markerfacecolor="none",
                        markeredgecolor=colors_CC[spd]
                    )

                    # Fit CC
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy, linestyle="--",
                                color=colors_CC[spd], linewidth=2)

                    y_all.extend(list(y))

                    if (r == 0) and (col == 0) and (f"CC{spd}" not in added):
                        legend_handles.append(h)
                        legend_labels.append(f"C-C SPD={spd}")
                        added.add(f"CC{spd}")

            # ============================================================
            # 3)  TWO-TIMES IC (2×IC) — ONLY IN TOTAL ROW
            # ============================================================
            if row == "Total":

                y2 = 2 * df_IC["OAOP_total"]

                # Hollow IC markers
                (h2,) = ax.plot(
                    x_ic, y2,
                    marker="o", markersize=8,
                    linestyle="None",
                    color=ic_color,
                    markerfacecolor="none",
                    markeredgecolor=ic_color,
                )

                # Fit dashed line
                xx, yy = polyfit_curve(x_ic, y2)
                if xx is not None:
                    ax.plot(
                        xx, yy,
                        linestyle="--",
                        color=ic_color,
                        linewidth=2.2
                    )

                y_all.extend(list(y2))

                # a small legend inside subplot
                ax.legend(
                    [h2], ["2×IC"],
                    loc="upper right",
                    frameon=False,
                    fontsize=10
                )

            # =====================================================
            # Record y-limits BEFORE we unify rows
            # =====================================================
            ymin, ymax = auto_ylim(ax, y_all)
            row_ymins[r].append(ymin)
            row_ymaxs[r].append(ymax)

            ax.grid(True, ls="--", alpha=0.3)
            ax.set_xlabel(r"Damping ratio $\zeta$")
            ax.set_ylabel(rf"$\mathit{{OAOP}}_{{\mathrm{{{row}}}}}$")

    # ==========================================================
    # UNIFY YLIMIT PER ROW
    # ==========================================================
    for r in range(3):
        ymin = min(row_ymins[r])
        ymax = max(row_ymaxs[r])

        axes[r][0].set_ylim(ymin, ymax)
        axes[r][1].set_ylim(ymin, ymax)

    # ==========================================================
    # APPLY TICKS AFTER UNIFY
    # ==========================================================
    for r in range(3):
        for c in range(2):
            set_ticks(axes[r][c])
            set_x_ticks(axes[r][c])

    # ==========================================================
    # Unified top legend
    # ==========================================================
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        frameon=False,
        fontsize=12
    )

    fig.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved: {savepath}")


# =========================================================
# ================= 3×2 OAOE (UC / DC / TOTAL) ===========
# =========================================================

def plot_OAOE_3x2(savepath):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.83)

    rows = ["UC", "DC", "Total"]
    SPUs = [0.5, 0.7]

    legend_handles = []
    legend_labels  = []
    added = set()

    sublab_matrix = [
    ["(a)", "(b)"],
    ["(c)", "(d)"],
    ["(e)", "(f)"]
    ]
    sid = 0

    row_ymins = {0: [], 1: [], 2: []}
    row_ymaxs = {0: [], 1: [], 2: []}

    for col, spu_value in enumerate(SPUs):

        for r, row in enumerate(rows):
            ax = axes[r][col]
            add_sublabel(ax, sublab_matrix[r][col])
            sid += 1

            y_all = []

            x_ic = df_IC["damping"]
            y_ic = df_IC["OAOE_total"]

            (h_ic,) = ax.plot(
                x_ic, y_ic,
                marker="o", markersize=7,
                linestyle="None",
                color=ic_color,
                markerfacecolor=ic_color,
                markeredgecolor=ic_color
            )
            xx, yy = polyfit_curve(x_ic, y_ic)
            if xx is not None:
                ax.plot(xx, yy, color=ic_color, linewidth=2)

            y_all.extend(list(y_ic))

            if (r == 0) and (col == 0) and ("IC" not in added):
                legend_handles.append(h_ic)
                legend_labels.append("IC")
                added.add("IC")

            # CPC + CC
            for spd in SPD_list:

                sub = df_CPC[(df_CPC["SP_U"] == spu_value) & (df_CPC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[f"OAOE_{row.lower()}"] if row != "Total" else sub["OAOE_total"]

                    (h,) = ax.plot(
                        x, y,
                        marker=markers_filled[spd], markersize=7,
                        linestyle="None",
                        color=colors_CPC[spd],
                        markerfacecolor=colors_CPC[spd],
                        markeredgecolor=colors_CPC[spd]
                    )
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy, color=colors_CPC[spd], linewidth=2)

                    y_all.extend(list(y))

                    if (r == 0) and (col == 0) and (f"CPC{spd}" not in added):
                        legend_handles.append(h)
                        legend_labels.append(f"C-P-C SPD={spd}")
                        added.add(f"CPC{spd}")

                # CC
                sub = df_CC[(df_CC["SP_U"] == spu_value) & (df_CC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[f"OAOE_{row.lower()}"] if row != "Total" else sub["OAOE_total"]

                    (h,) = ax.plot(
                        x, y,
                        marker=markers_hollow[spd], markersize=8,
                        linestyle="None",
                        color=colors_CC[spd],
                        markerfacecolor="none",
                        markeredgecolor=colors_CC[spd]
                    )
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy, linestyle="--", color=colors_CC[spd], linewidth=2)

                    y_all.extend(list(y))

                    if (r == 0) and (col == 0) and (f"CC{spd}" not in added):
                        legend_handles.append(h)
                        legend_labels.append(f"C-C SPD={spd}")
                        added.add(f"CC{spd}")

            ymin, ymax = auto_ylim(ax, y_all)
            row_ymins[r].append(ymin)
            row_ymaxs[r].append(ymax)

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xlabel(r"Damping ratio $\zeta$")
            ax.set_ylabel(rf"$\mathit{{OAOE}}_{{\mathrm{{{row}}}}}$ (%)")

            

    # ====================================================
    # UNIFY Y LIMITS PER ROW
    # ====================================================
    for r in range(3):
        ymin = min(row_ymins[r])
        ymax = max(row_ymaxs[r])

        axes[r][0].set_ylim(ymin, ymax)
        axes[r][1].set_ylim(ymin, ymax)

    # ====================================================
    # APPLY TICKS AFTER UNIFIED Y-LIMIT
    # ====================================================
    for r in range(3):
        for c in range(2):
            set_ticks(axes[r][c])
            set_x_ticks(axes[r][c])

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        frameon=False,
        fontsize=12
    )

    fig.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved: {savepath}")


# ============================
# RUN BOTH PLOTS
# ============================
plot_OAOP_3x2("OAOP_3x2.png")
plot_OAOE_3x2("OAOE_3x2.png")

# =========================================================
# 3×2 OAOP Improvement Plot (using imp_power_*)
# =========================================================

def plot_OAOP_improve_3x2(savepath):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.83)

    rows = ["UC", "DC", "Total"]
    imp_cols = {
        "UC": "imp_power_uc",
        "DC": "imp_power_dc",
        "Total": "imp_power_total"
    }
    SPUs = [0.5, 0.7]

    sublab_matrix = [
    ["(a)", "(b)"],
    ["(c)", "(d)"],
    ["(e)", "(f)"]
    ]
    sid = 0

    # store y-limits
    row_ymins = {0: [], 1: [], 2: []}
    row_ymaxs = {0: [], 1: [], 2: []}

    for col, spu_value in enumerate(SPUs):
        for r, row in enumerate(rows):

            ax = axes[r][col]
            add_sublabel(ax, sublab_matrix[r][col])


            y_all = []
            colname = imp_cols[row]

            # --- CPC + CC ---
            for spd in SPD_list:

                # CPC
                sub = df_CPC[(df_CPC["SP_U"] == spu_value) &
                             (df_CPC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[colname]

                    ax.plot(
                        x, y,
                        marker=markers_filled[spd],
                        markersize=7,
                        linestyle="None",
                        color=colors_CPC[spd],
                    )

                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy,
                                color=colors_CPC[spd], linewidth=2)

                    y_all.extend(list(y))

                # CC
                sub = df_CC[(df_CC["SP_U"] == spu_value) &
                            (df_CC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[colname]

                    ax.plot(
                        x, y,
                        marker=markers_hollow[spd],
                        markersize=8,
                        linestyle="None",
                        color=colors_CC[spd],
                        markerfacecolor="none",
                        markeredgecolor=colors_CC[spd],
                    )
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy, linestyle="--",
                                color=colors_CC[spd], linewidth=2)

                    y_all.extend(list(y))

            # get y limits before unify
            ymin, ymax = auto_ylim(ax, y_all)
            row_ymins[r].append(ymin)
            row_ymaxs[r].append(ymax)

            ax.grid(True, ls="--", alpha=0.3)
            ax.set_xlabel(r"Damping ratio $\zeta$")

            if row == "UC":
                ax.set_ylabel(r"$\Delta \mathit{OAOP}_{\mathrm{UC}}$ (%)")
            elif row == "DC":
                ax.set_ylabel(r"$\Delta \mathit{OAOP}_{\mathrm{DC}}$ (%)")
            else:
                ax.set_ylabel(r"$\Delta \mathit{OAOP}_{\mathrm{Total}}$ (%)")

    # =======================================================
    # unify y-limit per row
    # =======================================================
    for r in range(3):
        ymin = min(row_ymins[r])
        ymax = max(row_ymaxs[r])

        axes[r][0].set_ylim(ymin, ymax)
        axes[r][1].set_ylim(ymin, ymax)

    # =======================================================
    # apply ticks AFTER unify
    # =======================================================
    for r in range(3):
        for c in range(2):
            set_ticks(axes[r][c])
            set_x_ticks(axes[r][c])

    # legend
    fig.legend(
        [plt.Line2D([], [], color=colors_CPC[s], marker=markers_filled[s], linestyle="") for s in SPD_list] +
        [plt.Line2D([], [], color=colors_CC[s], marker=markers_hollow[s], linestyle="", markerfacecolor="none")
         for s in SPD_list],
        [f"C-P-C SPD={s}" for s in SPD_list] +
        [f"C-C SPD={s}" for s in SPD_list],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved {savepath}")

def plot_OAOE_improve_3x2(savepath):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.83)

    rows = ["UC", "DC", "Total"]
    imp_cols = {
        "UC": "imp_eff_uc",
        "DC": "imp_eff_dc",
        "Total": "imp_eff_total"
    }
    SPUs = [0.5, 0.7]

    
    sublab_matrix = [
    ["(a)", "(b)"],
    ["(c)", "(d)"],
    ["(e)", "(f)"]
    ]
    sid = 0

    row_ymins = {0: [], 1: [], 2: []}
    row_ymaxs = {0: [], 1: [], 2: []}

    for col, spu_value in enumerate(SPUs):
        for r, row in enumerate(rows):

            ax = axes[r][col]
            add_sublabel(ax, sublab_matrix[r][col])

            y_all = []
            colname = imp_cols[row]

            # CPC + CC
            for spd in SPD_list:

                # CPC
                sub = df_CPC[(df_CPC["SP_U"] == spu_value) &
                             (df_CPC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[colname]

                    ax.plot(
                        x, y,
                        marker=markers_filled[spd],
                        markersize=7,
                        linestyle="None",
                        color=colors_CPC[spd]
                    )
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy,
                                color=colors_CPC[spd], linewidth=2)

                    y_all.extend(list(y))

                # CC
                sub = df_CC[(df_CC["SP_U"] == spu_value) &
                            (df_CC["SP_D"] == spd)]
                if len(sub):
                    x = sub["damping"]
                    y = sub[colname]

                    ax.plot(
                        x, y,
                        marker=markers_hollow[spd],
                        markersize=8,
                        linestyle="None",
                        color=colors_CC[spd],
                        markerfacecolor="none",
                        markeredgecolor=colors_CC[spd],
                    )
                    xx, yy = polyfit_curve(x, y)
                    if xx is not None:
                        ax.plot(xx, yy, linestyle="--",
                                color=colors_CC[spd], linewidth=2)

                    y_all.extend(list(y))

            ymin, ymax = auto_ylim(ax, y_all)
            row_ymins[r].append(ymin)
            row_ymaxs[r].append(ymax)

            ax.grid(True, ls="--", alpha=0.3)
            ax.set_xlabel(r"Damping ratio $\zeta$")

            if row == "UC":
                ax.set_ylabel(r"$\Delta \mathit{OAOE}_{UC}$ (%)")
            elif row == "DC":
                ax.set_ylabel(r"$\Delta \mathit{OAOE}_{DC}$ (%)")
            else:
                ax.set_ylabel(r"$\Delta \mathit{OAOE}_{Total}$ (%)")

    # =======================================================
    # unify y-limit per row
    # =======================================================
    for r in range(3):
        ymin = min(row_ymins[r])
        ymax = max(row_ymaxs[r])

        axes[r][0].set_ylim(ymin, ymax)
        axes[r][1].set_ylim(ymin, ymax)

    # =======================================================
    # apply ticks AFTER unify
    # =======================================================
    for r in range(3):
        for c in range(2):
            set_ticks(axes[r][c])
            set_x_ticks(axes[r][c])

    # Legend
    fig.legend(
        [plt.Line2D([], [], color=colors_CPC[s], marker=markers_filled[s], linestyle="") for s in SPD_list] +
        [plt.Line2D([], [], color=colors_CC[s], marker=markers_hollow[s], linestyle="", markerfacecolor="none")
         for s in SPD_list],
        [f"C-P-C SPD={s}" for s in SPD_list] +
        [f"C-C SPD={s}" for s in SPD_list],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=12
    )

    fig.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved {savepath}")


# ============================
# CALL FUNCTIONS
# ============================

plot_OAOP_improve_3x2("OAOP_improve_3x2.png")
plot_OAOE_improve_3x2("OAOE_improve_3x2.png")



# =========================================================
# 3×2 OAOP Contribution Plot — Uniform Color Version
# NO color change with damping — only style differences
# =========================================================

import matplotlib.colors as mc
import colorsys

# ------------------------------------
# Damping list
# ------------------------------------
damping_list  = [0.10, 0.15, 0.20, 0.25, 0.30]

# ------------------------------------
# Base colors (uniform)
# ------------------------------------
base_uc_cpc_color = "#1f77b4"     # strong blue
base_dc_cpc_color = "#4fa3ff"     # lighter blue

base_uc_cc_color  = "#1f77b4"     # same hue but hollow
base_dc_cc_color  = "#4fa3ff"     # same hue but hollow


# ------------------------------------
# Lighten function for hatch outline
# ------------------------------------
def lighten_color(color, factor=0.5):
    """lighten the CC outline color"""
    r, g, b = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - factor * (1 - l)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


# ------------------------------------
# Main plot function
# ------------------------------------
def plot_OAOP_contribution_3x2(savepath):

    SPUs = [0.5, 0.7]
    SPDs = [1, 2, 3]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(top=0.83, hspace=0.32)

    # --- plotting parameters ---
    bar_width = 0.18
    x_positions = np.arange(len(damping_list))

    sublabs = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    sid = 0

    # --- legend items ---
    legend_handles = []
    legend_labels  = []
    added = set()

    # Fixed style (does not depend on damping color)
    color_uc_cpc = "#1f77b4"      # strong blue
    color_dc_cpc = "#1f77b4"      # same hue but lighter
    color_uc_cc  = "#1f77b4"      # same hue
    color_dc_cc  = "#1f77b4"

    alpha_dc_cpc = 0.55
    alpha_dc_cc  = 0.55

    hatch_cc = "///"

    for col, spu in enumerate(SPUs):
        for row, spd in enumerate(SPDs):

            ax = axes[row][col]
            add_sublabel(ax, sublabs[sid])
            sid += 1

            df_cpc = df_CPC[(df_CPC["SP_U"] == spu) & (df_CPC["SP_D"] == spd)].sort_values("damping")
            df_cc  = df_CC [(df_CC ["SP_U"] == spu) & (df_CC ["SP_D"] == spd)].sort_values("damping")

            for i, dmp in enumerate(damping_list):

                cpc_row = df_cpc[df_cpc["damping"] == dmp]
                cc_row  = df_cc [df_cc ["damping"] == dmp]

                if len(cpc_row)==0 or len(cc_row)==0:
                    continue

                cpc = cpc_row.iloc[0]
                cc  = cc_row.iloc[0]

                # OAOP % contributions
                uc_cpc = cpc["pct_OAOP_uc"]
                dc_cpc = cpc["pct_OAOP_dc"]
                uc_cc  = cc ["pct_OAOP_uc"]
                dc_cc  = cc ["pct_OAOP_dc"]

                # ----------------------
                # 4 bars at this ζ
                # ----------------------

                # UC CPC (solid)
                ax.bar(
                    x_positions[i] - 1.5*bar_width,
                    uc_cpc, width=bar_width,
                    color=color_uc_cpc, edgecolor="black"
                )

                # DC CPC (solid lighter)
                ax.bar(
                    x_positions[i] - 0.5*bar_width,
                    dc_cpc, width=bar_width,
                    color=color_dc_cpc, edgecolor="black",
                    alpha=alpha_dc_cpc
                )

                # UC CC (hollow + hatch)
                ax.bar(
                    x_positions[i] + 0.5*bar_width,
                    uc_cc, width=bar_width,
                    facecolor="none",
                    edgecolor=color_uc_cc,
                    hatch=hatch_cc,
                    linewidth=1.2
                )

                # DC CC (hollow + hatch lighter)
                ax.bar(
                    x_positions[i] + 1.5*bar_width,
                    dc_cc, width=bar_width,
                    facecolor="none",
                    edgecolor=color_dc_cc,
                    hatch=hatch_cc,
                    linewidth=1.2,
                    alpha=alpha_dc_cc
                )

                # Legend (only once)
                if "UC_CPC" not in added:
                    legend_handles.append(
                        plt.Rectangle((0,0),1,1,color=color_uc_cpc,edgecolor="black")
                    )
                    legend_labels.append("UC (C-P-C)")
                    added.add("UC_CPC")

                if "DC_CPC" not in added:
                    legend_handles.append(
                        plt.Rectangle((0,0),1,1,color=color_dc_cpc,edgecolor="black",alpha=alpha_dc_cpc)
                    )
                    legend_labels.append("DC (C-P-C)")
                    added.add("DC_CPC")

                if "UC_CC" not in added:
                    legend_handles.append(
                        plt.Rectangle((0,0),1,1,facecolor="none",edgecolor=color_uc_cc,hatch=hatch_cc)
                    )
                    legend_labels.append("UC (C-C)")
                    added.add("UC_CC")

                if "DC_CC" not in added:
                    legend_handles.append(
                        plt.Rectangle((0,0),1,1,facecolor="none",edgecolor=color_dc_cc,hatch=hatch_cc,alpha=alpha_dc_cc)
                    )
                    legend_labels.append("DC (C-C)")
                    added.add("DC_CC")


            # --- formatting ---
            ax.set_ylim(0, 100)
            ax.set_yticks([0,20,40,60,80,100])
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{d:.2f}" for d in damping_list])

            ax.set_xlabel(r"Damping ratio $\zeta$")
            ax.set_ylabel("OAOP contribution (%)")
            ax.set_title(f"SPU={spu}, SPD={spd}", fontsize=13)
            ax.grid(True, linestyle="--", alpha=0.30)

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False, ncol=4, fontsize=12
    )

    fig.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved {savepath}")




# run
plot_OAOP_contribution_3x2("OAOP_contribution_3x2.png")


import matplotlib.colors as mc
import colorsys

# ---------------------------------------------------------
# Helper: create low-saturation color for CC system
# ---------------------------------------------------------
def desaturate(color, factor=0.45):
    """
    Convert a high-saturation color into a lower-saturation color
    Used for C-C curves
    factor <1 → less saturation
    """
    c = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*c)
    s = s * factor
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


# ---------------------------------------------------------
# OAOP trend: CPC high sat, CC low sat
# ---------------------------------------------------------
def plot_OAOP_contribution_trend_3x2(savepath):

    SPUs = [0.5, 0.7]
    SPDs = [1, 2, 3]

    fig, axes = plt.subplots(3, 2, figsize=(14,12))
    plt.subplots_adjust(top=0.83, hspace=0.32)

    sublabs = ["(a)", "(c)", "(e)", "(b)", "(d)", "(f)"]
    sid = 0

    # -----------------------------
    # Base colors
    # -----------------------------
    UC_color = "#D92523"   
    DC_color = "#2E974E"   

    # CC system uses lower saturation
    UC_color_CC = "#ef8476"
    DC_color_CC = "#8ccdbf"

    # marker and line styles
    marker_uc = "^"
    marker_dc = "v"

    linestyle_cpc = "-"
    linestyle_cc  = "--"

    # Fixed x-axis
    xticks = np.arange(0.095, 0.305, 0.05)

    for col, spu in enumerate(SPUs):
        for row, spd in enumerate(SPDs):

            ax = axes[row][col]
            add_sublabel(ax, sublabs[sid])
            sid += 1

            # ========================
            # Plot CPC (high saturation)
            # ========================
            df = df_CPC[(df_CPC["SP_U"] == spu) & (df_CPC["SP_D"] == spd)]
            df = df.sort_values("damping")

            ax.plot(
                df["damping"], df["pct_OAOP_uc"],
                linestyle=linestyle_cpc, marker=marker_uc,
                color=UC_color, markersize=7,
                label="UC (C-P-C)" if row==0 and col==0 else None
            )

            ax.plot(
                df["damping"], df["pct_OAOP_dc"],
                linestyle=linestyle_cpc, marker=marker_dc,
                color=DC_color, markersize=7,
                label="DC (C-P-C)" if row==0 and col==0 else None
            )

            # ========================
            # Plot CC (low saturation)
            # ========================
            df = df_CC[(df_CC["SP_U"] == spu) & (df_CC["SP_D"] == spd)]
            df = df.sort_values("damping")

            ax.plot(
                df["damping"], df["pct_OAOP_uc"],
                linestyle=linestyle_cc, marker=marker_uc, markersize=7,
                color=UC_color_CC,                  # line color
                markerfacecolor="none",             # hollow marker
                markeredgecolor=UC_color_CC,        # edge color
                markeredgewidth=1.3,
                label="UC (C-C)" if row==0 and col==0 else None
            )

            ax.plot(
                df["damping"], df["pct_OAOP_dc"],
                linestyle=linestyle_cc, marker=marker_dc, markersize=7,
                color=DC_color_CC,
                markerfacecolor="none",
                markeredgecolor=DC_color_CC,
                markeredgewidth=1.3,
                label="DC (C-C)" if row==0 and col==0 else None
            )


            # formatting
            ax.set_ylim(0,100)
            ax.set_yticks([0,20,40,60,80,100])

            xticks = np.array([0.10, 0.15, 0.20, 0.25, 0.30])

            ax.set_xlim(0.09, 0.31)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.2f}" for x in xticks])


            ax.set_xlabel(r"Damping ratio $\zeta$")
            ax.set_ylabel("OAOP contribution (%)")
            ax.set_title(f"SPU={spu}, SPD={spd}")
            ax.grid(True, ls="--", alpha=0.3)

    # legend
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False, ncol=4
    )

    fig.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(savepath, dpi=900, bbox_inches="tight")
    print(f"✔ Saved {savepath}")



plot_OAOP_contribution_trend_3x2("OAOP_contribution_trend_3x2.png")



