#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np



# =========================================================
# PATHS
# =========================================================
PATH_IC  = "/ibex/scratch/projects/c2298/energy_harvesting/single_viv/post/final_results.csv"
PATH_CC  = "/ibex/scratch/projects/c2298/energy_harvesting/double/post/final_results.csv"
PATH_CPC = "/ibex/scratch/projects/c2298/energy_harvesting/c_sp_c/post/final_results.csv"


# Global IC baseline (assigned in driver)
df_ic_global = None



# =========================================================
# LOAD CSV
# =========================================================
def load_csv(tag, path):
    print(f"\n📂 Loading {tag}: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("-", "_")
        .str.replace(" ", "_")
        .str.replace("*", "_star")
    )
    print(f"   ✔ Loaded shape {df.shape}")
    return df



# =========================================================
# STANDARDIZE COLUMNS
# =========================================================
def harmonize(df, system):
    rename_map = {
        "Y*_1": "Y1", "Y*_2": "Y2",
        "Y_star_1": "Y1", "Y_star_2": "Y2",
        "P_uc": "P_UC", "P_dc": "P_DC",
        "efficiency_uc": "eff_UC",
        "efficiency_dc": "eff_DC",
        "efficiency_total": "eff_total",
        "P_total": "P_total",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ensure essential fields exist
    if "Y1" not in df.columns: df["Y1"] = np.nan
    if "Y2" not in df.columns: df["Y2"] = np.nan
    if "P_UC" not in df.columns: df["P_UC"] = np.nan
    if "P_DC" not in df.columns: df["P_DC"] = np.nan
    if "P_total" not in df.columns: df["P_total"] = np.nan
    if "eff_UC" not in df.columns: df["eff_UC"] = np.nan
    if "eff_DC" not in df.columns: df["eff_DC"] = np.nan
    if "eff_total" not in df.columns:
        df["eff_total"] = df["eff_UC"] + df["eff_DC"]

    df["system"] = system
    return df



# =========================================================
# DETECT SYSTEM TYPE
# =========================================================
def detect_system(df):

    cols = set(df.columns)

    if "Cd_ave3" in cols and "P_DC" in cols:
        return "CPC"

    if "Cd_ave2" in cols and "P_DC" in cols:
        return "CC"

    # IC: SP_U = SP_D = 0
    if "SP_U" in df.columns and "SP_D" in df.columns:
        if df["SP_U"].nunique()==1 and df["SP_D"].nunique()==1:
            if df["SP_U"].iloc[0]==0 and df["SP_D"].iloc[0]==0:
                return "IC"

    # fallback: only total power
    if "P_total" in cols and "P_DC" not in cols:
        return "IC"

    return "UNKNOWN"



# =========================================================
# HELPERS
# =========================================================
def trapz_avg(y, x):
    y = np.array(y, float)
    x = np.array(x, float)
    if len(x) < 2: return np.nan
    dx = x.max() - x.min()
    if dx == 0: return np.nan
    return np.trapezoid(y, x) / dx


# =========================================================
# PEAK METRICS (with UR location)
# =========================================================
def compute_peak_metrics(df, sys):

    def UR_at_peak(series):
        if series.isna().all():
            return np.nan
        idx = series.idxmax()
        return df.loc[idx, "UR"]

    if sys == "IC":
        peakP = df["P_total"].max()
        peakE = df["eff_total"].max()

        return dict(
            peak_power_uc    = peakP,
            peak_power_dc    = np.nan,
            peak_power_total = peakP,

            peak_eff_uc      = peakE,
            peak_eff_dc      = np.nan,
            peak_eff_total   = peakE,

            peak_Y_uc        = df["Y1"].max(),
            peak_Y_dc        = np.nan,

            UR_peak_power_uc    = np.nan,
            UR_peak_power_dc    = np.nan,
            UR_peak_power_total = UR_at_peak(df["P_total"]),

            UR_peak_eff_uc      = np.nan,
            UR_peak_eff_dc      = np.nan,
            UR_peak_eff_total   = UR_at_peak(df["eff_total"]),
        )

    # CC / CPC
    return dict(
        peak_power_uc    = df["P_UC"].max(),
        peak_power_dc    = df["P_DC"].max(),
        peak_power_total = (df["P_total"]).max(),

        peak_eff_uc      = df["eff_UC"].max(),
        peak_eff_dc      = df["eff_DC"].max(),
        peak_eff_total   = df["eff_total"].max(),

        peak_Y_uc        = df["Y1"].max(),
        peak_Y_dc        = df["Y2"].max(),

        UR_peak_power_uc    = UR_at_peak(df["P_UC"]),
        UR_peak_power_dc    = UR_at_peak(df["P_DC"]),
        UR_peak_power_total = UR_at_peak(df["P_total"]),

        UR_peak_eff_uc      = UR_at_peak(df["eff_UC"]),
        UR_peak_eff_dc      = UR_at_peak(df["eff_DC"]),
        UR_peak_eff_total   = UR_at_peak(df["eff_total"]),
    )



# =========================================================
# OAOP / OAOE
# =========================================================
def compute_OAOP_OAOE(df, sys, ic_baseline=None):

    UR = df["UR"]

    # ====================== IC（isolated cylinder） ======================
    if sys == "IC":
        OAOP_uc = trapz_avg(df["P_total"], UR)         # IC 只有 total → 作为 UC
        OAOE_uc = trapz_avg(df["eff_total"], UR)       # IC 只有 total → 作为 UC

        return dict(
            OAOP_uc=OAOP_uc,
            OAOP_dc=np.nan,
            OAOP_total=OAOP_uc,

            OAOE_uc=OAOE_uc,
            OAOE_dc=np.nan,
            OAOE_total=OAOE_uc,

            OAOP_IC=OAOP_uc,      # baseline = IC
            OAOE_IC=OAOE_uc       # baseline = IC
        )

    # ====================== CPC / CC ======================
    OAOP_uc    = trapz_avg(df["P_UC"], UR)
    OAOP_dc    = trapz_avg(df["P_DC"], UR)
    OAOP_total = trapz_avg(df["P_total"], UR)

    OAOE_uc    = trapz_avg(df["eff_UC"], UR)
    OAOE_dc    = trapz_avg(df["eff_DC"], UR)
    OAOE_total = trapz_avg(df["eff_total"], UR)

    # MUST use IC baseline
    if ic_baseline is None:
        raise ValueError("CPC/CC must provide ic_baseline!")

    return dict(
        OAOP_uc=OAOP_uc,
        OAOP_dc=OAOP_dc,
        OAOP_total=OAOP_total,

        OAOE_uc=OAOE_uc,
        OAOE_dc=OAOE_dc,
        OAOE_total=OAOE_total,

        OAOP_IC=ic_baseline["OAOP_IC"],
        OAOE_IC=ic_baseline["OAOE_IC"],
    )
# =========================================================
# IMPROVEMENTS
# =========================================================
def compute_improvements(avg, ic_avg, sys):

    # IC → no improvement
    if sys == "IC":
        return {
            "imp_power_uc": np.nan,
            "imp_power_dc": np.nan,
            "imp_power_total": np.nan,
            "imp_eff_uc": np.nan,
            "imp_eff_dc": np.nan,
            "imp_eff_total": np.nan
        }

    # IC baseline (UC only)
    ic_pow = ic_avg["OAOP_IC"]
    ic_eff = ic_avg["OAOE_IC"]

    # ===================== OAOP 改善 (你的最终公式) =====================
    imp_power_uc    = (avg["OAOP_uc"]    / ic_pow - 1)*100
    imp_power_dc    = (avg["OAOP_dc"]    / ic_pow - 1)*100
    imp_power_total = (avg["OAOP_total"] / (2 * ic_pow) - 1)*100

    # ===================== OAOE 改善 (你的最终公式) =====================
    imp_eff_uc    = (avg["OAOE_uc"] / ic_eff - 1)*100
    imp_eff_dc    = (avg["OAOE_dc"] / ic_eff - 1)*100
    imp_eff_total = (avg["OAOE_total"] / ic_eff - 1)*100   # 注意：total 不除以 2

    return dict(
        imp_power_uc=imp_power_uc,
        imp_power_dc=imp_power_dc,
        imp_power_total=imp_power_total,

        imp_eff_uc=imp_eff_uc,
        imp_eff_dc=imp_eff_dc,
        imp_eff_total=imp_eff_total
    )


# =========================================================
# CONTRIBUTION RATIOS
# =========================================================
def compute_contribution(avg, sys):

    if sys == "IC":
        return dict(
            pct_OAOP_uc=np.nan,
            pct_OAOP_dc=np.nan,
            pct_OAOE_uc=np.nan,
            pct_OAOE_dc=np.nan,
        )

    total_P = avg["OAOP_total"]
    if total_P > 0:
        pct_OAOP_uc = 100 * avg["OAOP_uc"] / total_P
        pct_OAOP_dc = 100 * avg["OAOP_dc"] / total_P
    else:
        pct_OAOP_uc = pct_OAOP_dc = np.nan

    total_E = avg["OAOE_uc"] + avg["OAOE_dc"]
    if total_E > 0:
        pct_OAOE_uc = 100 * avg["OAOE_uc"] / total_E
        pct_OAOE_dc = 100 * avg["OAOE_dc"] / total_E
    else:
        pct_OAOE_uc = pct_OAOE_dc = np.nan

    return dict(
        pct_OAOP_uc=pct_OAOP_uc,
        pct_OAOP_dc=pct_OAOP_dc,
        pct_OAOE_uc=pct_OAOE_uc,
        pct_OAOE_dc=pct_OAOE_dc,
    )


# =========================================================
# MAIN SUMMARY (SYSTEM-LEVEL POWER DENSITY)
# =========================================================
def summarize(df, outfile):

    global df_ic_global

    # ----------------------------------------------------
    # Detect system: IC, CC, or CPC
    # ----------------------------------------------------
    sys = detect_system(df)
    if sys == "UNKNOWN":
        raise ValueError("Unrecognized CSV format")

    summaries = []

    # ----------------------------------------------------
    # Prepare IC baseline for OAOE/OAOP
    # ----------------------------------------------------
    if sys == "IC":
        ic_avg = None
    else:
        ic_avg = compute_OAOP_OAOE(df_ic_global, "IC", ic_baseline=None)

    # Plate thickness & cylinder diameter
    THK = 0.2
    D   = 1.0

    # ----------------------------------------------------
    # Loop over (SPU, SPD, damping) groups
    # ----------------------------------------------------
    for (spu, spd, damp), sub in df.groupby(["SP_U", "SP_D", "damping"]):

        sub = sub.sort_values("UR")

        # Peak-based metrics (based on power only, but NOT PD)
        peak = compute_peak_metrics(sub, sys)

        # OAOE / OAOP averages
        if sys == "IC":
            avg = compute_OAOP_OAOE(sub, sys, ic_baseline=None)
        else:
            avg = compute_OAOP_OAOE(sub, sys, ic_baseline=ic_avg)

        # Improvement (relative to IC)
        use_ic = avg if sys == "IC" else ic_avg
        imp  = compute_improvements(avg, use_ic, sys)
        pct  = compute_contribution(avg, sys)

        # ------------------------------------------------
        # WIDTH (constant for the case)
        # ------------------------------------------------
        if sys == "IC":
            width = D
        else:
            width = 2*D + spu + spd + THK

        # ------------------------------------------------
        # Compute PD(UR) at EACH UR (CORRECT)
        # ------------------------------------------------
        PD_list = []

        for idx, r in sub.iterrows():

            # -- Ymax at THIS UR --
            if sys == "IC":
                Ymax_ur = r["Y1"]
            else:
                Ymax_ur = max(r["Y1"], r["Y2"])

            # -- volume at THIS UR --
            height = 2*Ymax_ur + D
            volume_ur = width * height if height > 0 else np.nan

            # -- system power at THIS UR --
            if sys == "IC":
                P_sys = r["P_total"]
            else:
                P_sys = r["P_UC"] + r["P_DC"]

            # -- power density --
            if volume_ur > 0:
                PD_ur = P_sys / volume_ur
            else:
                PD_ur = np.nan

            PD_list.append(PD_ur)

        # ------------------------------------------------
        # Real peak system power density
        # ------------------------------------------------
        PD_peak = np.nanmax(PD_list) if len(PD_list) else np.nan

        # ------------------------------------------------
        # Build a summary row
        # ------------------------------------------------
        row = dict(
            system=sys,
            SP_U=spu,
            SP_D=spd,
            damping=damp,

            PD_peak_total = PD_peak,   # *** This is the real system PD ***
            width=width,
        )

        # Merge in other metrics
        row.update(peak)
        row.update(avg)
        row.update(imp)
        row.update(pct)

        summaries.append(row)

    # ----------------------------------------------------
    # Save CSV
    # ----------------------------------------------------
    out = pd.DataFrame(summaries)
    out.to_csv(outfile, index=False)
    print(f"✔ Saved {outfile} ({out.shape[0]} rows)")
    return out


# =========================================================
# DRIVER
# =========================================================
if __name__ == "__main__":

    print("\n=== GENERATING TRAPEZOID-INTEGRATED SUMMARY ===")

    # Load & harmonize all three systems
    df_ic  = harmonize(load_csv("IC",  PATH_IC),  "IC")
    df_cc  = harmonize(load_csv("C-C", PATH_CC),  "CC")
    df_cpc = harmonize(load_csv("C-P-C", PATH_CPC), "CPC")

    # Store IC baseline globally
    df_ic_global = df_ic.copy()

    # Summaries
    summarize(df_ic,  "summary_IC.csv")
    summarize(df_cc,  "summary_CC.csv")
    summarize(df_cpc, "summary_CPC.csv")

    print("\n🎉 DONE — all summaries generated successfully!")
