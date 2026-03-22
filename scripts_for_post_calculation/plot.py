import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

df = pd.read_csv('final_results.csv')
# Assuming 'df' has been loaded and power density calculated:
df['Y_MAX'] = df[['Y*_1', 'Y*_2']].max(axis=1)
df['Power_Density'] = (df['P_uc'] + df['P_dc']) *2 / ((2 + df['SP_D'] + df['SP_U'] + 0.2) * (8 + 1))
df['Total_power'] = (df['P_uc'] + df['P_dc'])
df['UCpower'] = df['P_uc']
df['DCpower'] = df['P_dc']               
df['Total_efficiency'] = df['efficiency_total'] 



def plot_contour_with_ic(
    df,
    spu_value,
    damping_value=0,
    save_prefix="contour",
    ic_dir=None,
    uc_col='Emean_uc11',
    dc_col='Emean_dc22',
    sp_u_col='SP_U',
    sp_d_col='SP_D',
    ur_col='UR',
    damping_col='damping',
    ic_file_pattern='computation_results',
    interp_method='linear',      # e.g. 'linear', 'cubic'
    fill_value=0,
    levels=100,
    cmap='PiYG',
    norm_center_frac=0.2,
    xlim=(3, 30),
    xticks=None,
    figsize=(16, 4)
):
    # ── 0) Validate columns ────────────────────────────────────────────────────
    needed = [sp_u_col, sp_d_col, ur_col, damping_col, uc_col, dc_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # ── 1) Filter by SPU & damping ─────────────────────────────────────────────
    df_spu = df[(df[sp_u_col] == spu_value) & (df[damping_col] == damping_value)].copy()
    if df_spu.empty:
        raise ValueError(f"No data for {sp_u_col}={spu_value}, {damping_col}={damping_value}")

    # ── 2) Read & append IC data at SP_D=0 ─────────────────────────────────────
    if ic_dir:
        ic_frames = []
        for sub in os.listdir(ic_dir):
            subdir = os.path.join(ic_dir, sub)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if fname.startswith(ic_file_pattern) and fname.endswith('.csv'):
                    path = os.path.join(subdir, fname)
                    try:
                        ic_df = pd.read_csv(path)
                    except Exception as e:
                        print(f"⚠️ Could not read {path}: {e}")
                        continue
                    if ur_col not in ic_df.columns or uc_col not in ic_df.columns:
                        continue
                    tmp = pd.DataFrame({
                        sp_u_col: spu_value,
                        sp_d_col: 0,
                        damping_col: damping_value,
                        ur_col: ic_df[ur_col].values,
                    })
                    tmp[uc_col] = ic_df[uc_col].values
                    tmp[dc_col] = ic_df[uc_col].values
                    ic_frames.append(tmp)
        if ic_frames:
            df_spu = pd.concat([df_spu, *ic_frames], ignore_index=True)

    # ── 3) Pivot to grid ────────────────────────────────────────────────────────
    pivot_uc = df_spu.pivot_table(index=sp_d_col, columns=ur_col, values=uc_col)
    pivot_dc = df_spu.pivot_table(index=sp_d_col, columns=ur_col, values=dc_col)

    # ── 4) Reindex to full SP_D × UR grid ──────────────────────────────────────
    spd_vals = np.union1d(pivot_uc.index, pivot_dc.index)
    ur_vals  = np.union1d(pivot_uc.columns, pivot_dc.columns)
    pivot_uc = pivot_uc.reindex(index=spd_vals, columns=ur_vals)
    pivot_dc = pivot_dc.reindex(index=spd_vals, columns=ur_vals)

    # ── 5) Safe interpolation ───────────────────────────────────────────────────
    def safe_interp(df, method, axis):
        try:
            return df.interpolate(method=method, axis=axis)
        except Exception as e:
            print(f"⚠️ {method} interpolation failed on axis {axis}: {e}\n   → falling back to linear")
            return df.interpolate(method='linear', axis=axis)

    for axis in (0, 1):
        pivot_uc = safe_interp(pivot_uc, interp_method, axis)
        pivot_dc = safe_interp(pivot_dc, interp_method, axis)

    pivot_uc = pivot_uc.fillna(fill_value)
    pivot_dc = pivot_dc.fillna(fill_value)

    # ── 6) Mesh & Z arrays ──────────────────────────────────────────────────────
    X, Y = np.meshgrid(ur_vals, spd_vals)
    Z_uc, Z_dc = pivot_uc.values, pivot_dc.values

    # ── 7) Transpose if needed ────────────────────────────────────────────────
    def align(Z, name):
        if Z.shape != X.shape:
            if Z.T.shape == X.shape:
                print(f"↔️ Transposing Z_{name}")
                return Z.T
            else:
                raise ValueError(f"Shape mismatch Z_{name} {Z.shape} vs mesh {X.shape}")
        return Z

    Z_uc, Z_dc = align(Z_uc, 'UC'), align(Z_dc, 'DC')
    Z_uc, Z_dc = np.ma.masked_invalid(Z_uc), np.ma.masked_invalid(Z_dc)

    # ── 8) Create norms ─────────────────────────────────────────────────────────
    def make_norm(Z):
        vmin, vmax = Z.min(), Z.max()
        center = vmin + norm_center_frac*(vmax - vmin)
        return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    norm_uc, norm_dc = make_norm(Z_uc), make_norm(Z_dc)

    # ── 9) Plot ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    c1 = ax1.contourf(X, Y, Z_uc, levels=levels, cmap=cmap, norm=norm_uc)
    fig.colorbar(c1, ax=ax1, label=uc_col)
    ax1.set(title=f"{uc_col} @ {sp_u_col}={spu_value}, ζ={damping_value}",
            xlabel=ur_col, ylabel=sp_d_col)
    ax1.set_xlim(*xlim)
    if xticks is not None: ax1.set_xticks(xticks)

    c2 = ax2.contourf(X, Y, Z_dc, levels=levels, cmap=cmap, norm=norm_dc)
    fig.colorbar(c2, ax=ax2, label=dc_col)
    ax2.set(title=f"{dc_col} @ {sp_u_col}={spu_value}, ζ={damping_value}",
            xlabel=ur_col)
    ax2.set_xlim(*xlim)
    if xticks is not None: ax2.set_xticks(xticks)

    plt.tight_layout()
    out = f"{save_prefix}_spu_{spu_value}_damp_{damping_value:g}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out

# ── Examples ────────────────────────────────────────────────────────────────
# Single-damping call:
# Loop over multiple dampings:
for ζ in (0.1, 0.15, 0.2, 0.25, 0.3):
    plot_contour_with_ic(
        df,
        spu_value=0.5,
        damping_value=ζ,
        save_prefix="Emean_custom",
        ic_dir="/ibex/scratch/projects/c2298/energy_harvesting/single_viv/",
        uc_col='Emean_uc1',
        dc_col='Emean_dc2',
        interp_method='cubic',
        fill_value=np.nan,
        levels=150,
        cmap='Greens',
        norm_center_frac=0.3,
        xlim=(3, 30),
        xticks=np.arange(3, 31, 3),
        figsize=(18, 5)
    )

# Loop over multiple dampings:
for ζ in (0.1, 0.15, 0.2, 0.25, 0.3):
    plot_contour_with_ic(
        df,
        spu_value=0.7,
        damping_value=ζ,
        save_prefix="Emean_custom",
        ic_dir="/ibex/scratch/projects/c2298/energy_harvesting/single_viv/",
        uc_col='Emean_uc1',
        dc_col='Emean_dc2',
        interp_method='cubic',
        fill_value=np.nan,
        levels=150,
        cmap='Greens',
        norm_center_frac=0.3,
        xlim=(3, 30),
        xticks=np.arange(3, 31, 3),
        figsize=(18, 5)
    )

from scipy.interpolate import griddata


def plot_energy_output(
    df,
    spu=0.5,
    damping=0.1,
    uc_col='UCpower',
    dc_col='DCpower',
    sp_u_col='SP_U',
    sp_d_col='SP_D',
    ur_col='UR',
    cmap='Greens',
    levels=32,
    figsize=(10, 8),
    save_prefix='energy_output'
):
    """
    2D color‐maps of UC/DC power vs UR (x) and SPD (y),
    at fixed SPU and damping, using the original data grid.
    """
    # 1) filter
    df_f = df[(df[sp_u_col] == spu) & (df['damping'] == damping)]
    if df_f.empty:
        raise ValueError(f"No data at {sp_u_col}={spu}, damping={damping}")

    # 2) pivot onto the original grid
    pivot_uc = (df_f
                .pivot(index=sp_d_col, columns=ur_col, values=uc_col)
                .sort_index()
                .sort_index(axis=1))
    pivot_dc = (df_f
                .pivot(index=sp_d_col, columns=ur_col, values=dc_col)
                .sort_index()
                .sort_index(axis=1))

    UR_vals  = pivot_uc.columns.values
    SPD_vals = pivot_uc.index.values

    # 3) build mesh & extract Z
    UR_grid, SPD_grid = np.meshgrid(UR_vals, SPD_vals)
    Z_uc = pivot_uc.values
    Z_dc = pivot_dc.values

    # 4) plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    cf1 = ax1.contourf(UR_grid, SPD_grid, Z_uc, levels=levels, cmap=cmap)
    fig.colorbar(cf1, ax=ax1, label=uc_col)
    ax1.set_title(f'{uc_col} @ SPU={spu}, damping={damping}')
    ax1.set_ylabel(sp_d_col)

    cf2 = ax2.contourf(UR_grid, SPD_grid, Z_dc, levels=levels, cmap=cmap)
    fig.colorbar(cf2, ax=ax2, label=dc_col)
    ax2.set_title(f'{dc_col} @ SPU={spu}, damping={damping}')
    ax2.set_xlabel(ur_col)
    ax2.set_ylabel(sp_d_col)

    plt.tight_layout()
    fname = f"{save_prefix}_SPU_{spu}_damp_{damping}.png"
    fig.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot as {fname}")


def plot_energy_for_all_dampings(
    df,
    spu,
    damping_list,
    save_prefix,        # take save_prefix as its own argument
    **plot_kwargs       # everything else goes here—but NOT save_prefix
):
    for ζ in damping_list:
        # build a unique prefix for each ζ
        prefix = f"{save_prefix}_SPU_{spu}_damp_{ζ:g}"
        plot_energy_output(
            df,
            spu=spu,
            damping=ζ,
            save_prefix=prefix,   # pass only this one
            **plot_kwargs         # note: do NOT include 'save_prefix' in plot_kwargs
        )

# ── Now call it ────────────────────────────────────────────────────────────────
common_kwargs = {
    'uc_col':   'UCpower',
    'dc_col':   'DCpower',
    'sp_u_col': 'SP_U',
    'sp_d_col': 'SP_D',
    'ur_col':   'UR',
    'cmap':     'Greens',
    'levels':   32,
    'figsize':  (10, 8)
}

plot_energy_for_all_dampings(
    df,
    spu=0.5,
    damping_list=[0.1, 0.15, 0.2, 0.25, 0.3],
    save_prefix='CPC_energy',
    **common_kwargs
)

plot_energy_for_all_dampings(
    df,
    spu=0.7,
    damping_list=[0.1, 0.15, 0.2, 0.25, 0.3],
    save_prefix='CPC_energy',
    **common_kwargs
)



def plot_power_density_combined(df, spd_values):

    save_name = "C-P-C-PowerDensity-UR_SPU_0.5_0.7"

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Define colors and markers for each SPD
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for each SPD value
    
    for idx, (spu, ax) in enumerate(zip([0.5, 0.7], axs)):
        for spd_idx, spd in enumerate(spd_values):
            power_density_values = []
            damping_ratios = []
            ur_values = []
            
            for damping in df['damping'].unique():
                df_damping = df[(df['damping'] == damping) & (df['SP_U'] == spu) & (df['SP_D'] == spd)]
                
                if not df_damping.empty:
                    max_power_density = df_damping['Power_Density'].max()
                    max_ur = df_damping[df_damping['Power_Density'] == max_power_density]['UR'].values[0]
                    
                    power_density_values.append(max_power_density)
                    damping_ratios.append(damping)
                    ur_values.append(max_ur)

            # Plot points and annotations with unique color and marker per SPD
            if power_density_values:
                ax.scatter(damping_ratios, power_density_values, color=colors[spd_idx], marker=markers[spd_idx], label=f'SPD={spd}')
                for x, y, ur in zip(damping_ratios, power_density_values, ur_values):
                    ax.text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
                
                # Fit a polynomial (degree 2) to data and plot it as a dashed line
                p = np.polyfit(damping_ratios, power_density_values, 2)
                fit = np.poly1d(p)
                x_fit = np.linspace(min(damping_ratios), max(damping_ratios), 100)
                ax.plot(x_fit, fit(x_fit), linestyle='--', color=colors[spd_idx], label=f'Fit SPD={spd}')
            
        # Set limits, labels, title, and legend
        ax.set_ylim(0, 10)  # Adjust if needed
        ax.set_xlim(0, 0.4)
        ax.set_xlabel('Damping Ratio')
        ax.set_title(f'Power Density vs Damping Ratio for SPU={spu}')
        ax.legend()
        ax.grid(True)

    axs[0].set_ylabel('Power Density')
    plt.tight_layout()
    plt.savefig(f"{save_name}_combined.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to release memory

    print(f"Saved plot as {save_name}_combined.png")

# Example usage
plot_power_density_combined(df, spd_values=[1, 2, 3])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_weighted(df, spu, spd, damping, ur_min, ur_max, power_col):
    sub = (df[(df['SP_U']==spu)
              & (df['SP_D']==spd)
              & (df['damping']==damping)
              & (df['UR']>=ur_min)
              & (df['UR']<=ur_max)]
           .sort_values('UR'))
    UR = sub['UR'].to_numpy()
    P  = sub[power_col].to_numpy()
    if UR.size < 2:
        return np.nan
    oaop = np.trapezoid(P, x=UR) / (ur_max - ur_min)
    if UR.size < 2:
        return np.nan
    weight = (UR * P).sum() / UR.sum()
    return oaop
def load_isolated_cylinder(ic_root):
    frames = []
    for sub in os.listdir(ic_root):
        sp = os.path.join(ic_root, sub)
        if not os.path.isdir(sp) or not sub.startswith('Single_VIV'):
            continue
        for fname in os.listdir(sp):
            if fname.startswith('computation_results') and fname.endswith('.csv'):
                df_tmp = pd.read_csv(os.path.join(sp, fname))
                df_tmp['Total_power'] = df_tmp['P_uc']
                frames.append(df_tmp)
    if not frames:
        raise RuntimeError("No IC data")
    return pd.concat(frames, ignore_index=True)

# --- load data ---
ic_dir = '/ibex/scratch/projects/c2298/energy_harvesting/single_viv'
df_ic = load_isolated_cylinder(ic_dir)
df_ic0 = df_ic[(df_ic['SP_U']==0) & (df_ic['SP_D']==0)]

# Prepare CPC total power
df['Total_power'] = df['P_uc'] + df['P_dc']

# Parameters
spu_values     = [0.5, 0.7]
spd_values     = [1, 2, 3]
damping_values = [0.1, 0.15, 0.2, 0.25, 0.3]
ur_min, ur_max = 3, 30

# Panels: (column name, ylabel)
panels = [
    ('P_uc',        'UC weighted mean [W]'),
    ('P_dc',        'DC weighted mean [W]'),
    ('Total_power', 'Total weighted mean [W]')
]

# Create 3 rows x 3 columns
fig, axes = plt.subplots(
    nrows=len(panels),
    ncols=len(spu_values),
    figsize=(15, 12),
    sharex=True,
    sharey='row'
)

for col, spu in enumerate(spu_values):
    for row, (power_col, ylabel) in enumerate(panels):
        ax = axes[row, col]

        # CPC curves for each SPD
        for idx, spd in enumerate(spd_values):
            y_cpc = [
                compute_weighted(df, spu, spd, d, ur_min, ur_max, power_col)
                for d in damping_values
            ]
            ax.plot(
                damping_values, y_cpc,
                '-o', color=f"C{idx}",
                label=f'CPC SPD={spd}'
            )

        # IC baseline (always using P_uc)
        y_ic = [
            compute_weighted(df_ic0, 0, 0, d, ur_min, ur_max, 'P_uc')
            for d in damping_values
        ]
        ax.plot(
            damping_values, y_ic,
            '--k', linewidth=2, marker='X', markersize=8,
            label='Isolated Cylinder (UC)'
        )

        # Titles, labels, legend
        if row == 0:
            ax.set_title(f'SPU = {spu}', fontsize=14)
        if row == len(panels) - 1:
            ax.set_xlabel('Damping ratio ζ')
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=8)

# Super-title and layout
fig.suptitle(
    "UR-weighted mean vs damping ratio for SPU = 0.5, 0.7",
    fontsize=16,
    y=0.95
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('weighted_means_uc_dc_total_3SPU.png', dpi=300)


def plot_weighted_improvement(
    df, df_ic0,
    spu_values, spd_values, damping_values,
    ur_min, ur_max
):
    # 1) compute the IC baseline (UC) for each damping
    ic_uc_baseline = [
        compute_weighted(df_ic0, 0, 0, d, ur_min, ur_max, 'P_uc')
        for d in damping_values
    ]

    labels = ['UC Improvement', 'DC Improvement', 'Total Improvement']
    fig, axes = plt.subplots(3, len(spu_values),
                             figsize=(15, 12),
                             sharex=True, sharey='row')

    for col, spu in enumerate(spu_values):
        for row, metric in enumerate(['P_uc', 'P_dc', 'Total_power']):
            ax = axes[row, col]

            for idx, spd in enumerate(spd_values):
                improvs = []
                for i, d in enumerate(damping_values):
                    w_cpc = compute_weighted(df, spu, spd, d, ur_min, ur_max, metric)
                    baseline = ic_uc_baseline[i]
                    if row < 2:
                        imp = w_cpc / baseline
                    else:
                        imp = w_cpc / (2 * baseline)
                    improvs.append(imp)

                ax.plot(damping_values,
                        improvs,
                        '-o',
                        color=f"C{idx}",
                        label=f"SPD={spd}")
            
            if row == 0:
                ax.set_ylim(0, 5)
            elif row == 1:
                ax.set_ylim(0, 10)
            else:  # row == 2
                ax.set_ylim(0, 6.5)

            if col == 0:
                ax.set_ylabel(labels[row])
            if row == 0:
                ax.set_title(f"SPU = {spu}")
            if row == 2:
                ax.set_xlabel("Damping ratio ζ")

            ax.grid(True)
            ax.legend(fontsize=8)

    fig.suptitle("Improvement factor of UR‐weighted OAOP vs. IC baseline", fontsize=16, y=0.94)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig("weighted_improvement_png", dpi=300)
    plt.show()
    plt.close(fig)
    print("Saved plot as weighted_improvement_png.png")


# ── Example usage ─────────────────────────────────────────────────────────
spu_vals     = [0.5, 0.7]
spd_vals     = [1, 2, 3]
damping_vals = [0.1, 0.15, 0.2, 0.25, 0.3]
ur_min, ur_max = 3, 30

plot_weighted_improvement(
    df, df_ic0,
    spu_values=spu_vals,
    spd_values=spd_vals,
    damping_values=damping_vals,
    ur_min=ur_min,
    ur_max=ur_max
)


def compute_avg_efficiency(
    df, spu, spd, damping, ur_min, ur_max, eff_col
):
    """
    Returns the UR-weighted mean of eff_col over UR_min→UR_max:
      weighted_eff = ∑(UR * eff) / ∑UR
    """
    sub = (df[(df['SP_U']==spu)
              & (df['SP_D']==spd)
              & (df['damping']==damping)
              & (df['UR']>=ur_min)
              & (df['UR']<=ur_max)]
           .sort_values('UR'))

    UR  = sub['UR'].to_numpy()
    E   = sub[eff_col].to_numpy()
    if UR.size < 1 or UR.sum() == 0:
        return np.nan

    return (UR * E).sum() / UR.sum()


def plot_efficiency_improvement(
    df, df_ic0,
    spu_values, spd_values, damping_values,
    ur_min, ur_max
):
    # 1) compute IC baseline (UC efficiency) for each damping
    ic_baseline = [
        compute_avg_efficiency(df_ic0, 0, 0, d, ur_min, ur_max, 'efficiency_uc')
        for d in damping_values
    ]

    metrics = ['efficiency_uc', 'efficiency_dc', 'Total_efficiency']
    labels  = ['UC Eff. Improvement', 'DC Eff. Improvement', 'Total Eff. Improvement']

    fig, axes = plt.subplots(
        nrows=3, ncols=len(spu_values),
        figsize=(15, 12),
        sharex=True, sharey=False
    )

    for col, spu in enumerate(spu_values):
        for row, eff in enumerate(metrics):
            ax = axes[row, col]
            for idx, spd in enumerate(spd_values):
                imps = []
                for i, d in enumerate(damping_values):
                    cpc_eff = compute_avg_efficiency(df, spu, spd, d, ur_min, ur_max, eff)
                    base    = ic_baseline[i]
                    # for total eff, divide by 2*baseline
                    if row < 2:
                        imp = cpc_eff / base
                    else:
                        imp = cpc_eff / (2 * base)
                    imps.append(imp)

                ax.plot(
                    damping_values, imps,
                    '-o', color=f"C{idx}",
                    label=f"SPD={spd}"
                )

            if col == 0:
                ax.set_ylabel(labels[row])
            if row == 0:
                ax.set_title(f"SPU = {spu}")
            if row == 2:
                ax.set_xlabel("Damping ratio ζ")

            ax.grid(True)
            ax.legend(fontsize=8)

    fig.suptitle(
        "Improvement in UR-weighted efficiency vs. IC baseline",
        fontsize=16, y=0.94
    )
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig("efficiency_improvement.png", dpi=300)
    plt.show()
    plt.close(fig)
    print("Saved plot as efficiency_improvement.png")


# ── Example usage ─────────────────────────────────────────────────────────────
spu_vals     = [0.5, 0.7]
spd_vals     = [1, 2, 3]
damping_vals = [0.1, 0.15, 0.2, 0.25, 0.3]
ur_min, ur_max = 3, 30

plot_efficiency_improvement(
    df, df_ic0,
    spu_values=spu_vals,
    spd_values=spd_vals,
    damping_values=damping_vals,
    ur_min=ur_min,
    ur_max=ur_max
    )


def compute_oaop(df, spu, spd, damping, ur_min, ur_max, power_col='Total_power'):
    sub = (df[(df['SP_U']==spu)
              & (df['SP_D']==spd)
              & (df['damping']==damping)
              & (df['UR']>=ur_min)
              & (df['UR']<=ur_max)]
           .sort_values('UR'))
    UR = sub['UR'].to_numpy()
    P  = sub[power_col].to_numpy()
    if UR.size < 2:
        return np.nan
    return np.trapezoid(P, x=UR) / (ur_max - ur_min)

def compute_avg_efficiency(df, spu, spd, damping, ur_min, ur_max, eff_col):
    sub = (df[(df['SP_U']==spu)
              & (df['SP_D']==spd)
              & (df['damping']==damping)
              & (df['UR']>=ur_min)
              & (df['UR']<=ur_max)]
           .sort_values('UR'))
    UR = sub['UR'].to_numpy()
    E  = sub[eff_col].to_numpy()
    if UR.size < 2:
        return np.nan
    return np.trapezoid(E, x=UR) / (ur_max - ur_min)

def plot_power_eff_improvement(
    df, df_ic0,
    spu_values, spd_values, damping_values,
    ur_min, ur_max
):
    # pre‐compute IC baselines for UC power & UC efficiency
    ic_pow_uc = np.array([
        compute_oaop(df_ic0, 0, 0, d, ur_min, ur_max, 'P_uc')
        for d in damping_values
    ])
    ic_eff_uc = np.array([
        compute_avg_efficiency(df_ic0, 0, 0, d, ur_min, ur_max, 'efficiency_uc')
        for d in damping_values
    ])

    # rows: (power_col, eff_col, power_factor, eff_factor, ylabel)
    rows = [
        ('P_uc',        'efficiency_uc',    1,  1, 'UC Improvement'),
        ('P_dc',        'efficiency_dc',    1,  1, 'DC Improvement'),
        ('Total_power', 'Total_efficiency', 2,  2, 'Total Improvement'),
    ]

    fig, axes = plt.subplots(
        nrows=3, ncols=len(spu_values),
        figsize=(6*len(spu_values), 12),
        sharex=True
    )

    for col, spu in enumerate(spu_values):
        for row_idx, (pcol, ecol, pfac, efac, ylabel) in enumerate(rows):
            ax = axes[row_idx, col]

            for idx, spd in enumerate(spd_values):
                # CPC OAOP and avg-eff
                oaop_vals = np.array([
                    compute_oaop(df, spu, spd, d, ur_min, ur_max, pcol)
                    for d in damping_values
                ])
                eff_vals = np.array([
                    compute_avg_efficiency(df, spu, spd, d, ur_min, ur_max, ecol)
                    for d in damping_values
                ])

                # baselines
                pow_base = ic_pow_uc * pfac
                eff_base = ic_eff_uc * efac

                # improvement factors
                pow_imp = oaop_vals / pow_base
                eff_imp = eff_vals / eff_base

                color = f"C{idx}"
                ax.plot(damping_values, pow_imp,
                        '-o', color=color, label=f"SPD={spd} power")
                ax.plot(damping_values, eff_imp,
                        '--s', color=color, label=f"SPD={spd} eff")

            # set the y-limits by row_idx
            if row_idx == 0:
                ax.set_ylim(0, 4)
            elif row_idx == 1:
                ax.set_ylim(0, 8)
            else:  # row_idx == 2
                ax.set_ylim(0, 5)

            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 2:
                ax.set_xlabel("Damping ζ")
            if row_idx == 0:
                ax.set_title(f"SPU = {spu}")

            ax.grid(True)
            ax.legend(fontsize=8)

    plt.suptitle(
        "Improvement in OAOP (solid) & Efficiency (dashed)",
        fontsize=16, y=0.92
    )
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig("power_eff_improvement_rows.png", dpi=300)
    plt.show()
    plt.close(fig)
    print("Saved plot as power_eff_improvement_rows.png")

plot_power_eff_improvement(
    df, df_ic0,
    spu_values=spu_vals,
    spd_values=spd_vals,
    damping_values=damping_vals,
    ur_min=ur_min,
    ur_max=ur_max
)

rows = []
for spu in spu_vals:
    for spd in spd_vals:
        for d in damping_vals:
            oaop_cpc = compute_oaop(df, spu, spd, d, ur_min, ur_max, 'Total_power')
            oaop_ic  = compute_oaop(df_ic0, 0, 0, d, ur_min, ur_max, 'P_uc')
            imp_pow  = oaop_cpc/oaop_ic if oaop_ic else np.nan

            eff_cpc  = compute_avg_efficiency(df, spu, spd, d, ur_min, ur_max, 'Total_efficiency')
            eff_ic   = compute_avg_efficiency(df_ic0, 0, 0, d, ur_min, ur_max, 'efficiency_uc')
            imp_eff  = eff_cpc/(2*eff_ic) if eff_ic else np.nan

            rows.append({
                'SPU': spu, 'SPD': spd, 'Damping': d,
                'Power_Improvement': imp_pow,
                'Efficiency_Improvement': imp_eff
            })

df_imp = pd.DataFrame(rows)

# --- 6) Find and print the peaks ---
max_pow_row = df_imp.loc[df_imp['Power_Improvement'].idxmax()]
max_eff_row = df_imp.loc[df_imp['Efficiency_Improvement'].idxmax()]

print("=== Peak Power Improvement ===")
print(f"  SPU = {max_pow_row.SPU}")
print(f"  SPD = {max_pow_row.SPD}")
print(f"  Damping ratio = {max_pow_row.Damping}")
print(f"  Power improvement = {max_pow_row.Power_Improvement:.3f}")

print("\n=== Peak Efficiency Improvement ===")
print(f"  SPU = {max_eff_row.SPU}")
print(f"  SPD = {max_eff_row.SPD}")
print(f"  Damping ratio = {max_eff_row.Damping}")
print(f"  Efficiency improvement = {max_eff_row.Efficiency_Improvement:.3f}")


df_sub = (df[['SP_U','SP_D','damping','UR','UCpower']]
            .query("SP_U==0.7 and SP_D==2 and damping==0.2")
            .sort_values('UR')
            .set_index('UR'))

# 2) Define your original and refined grids
UR_orig    = np.array([3, 5, 7, 10, 15, 20, 25, 30])
UR_refined = np.array([3, 4, 5, 6, 7, 8.5, 10, 15, 20, 25, 30])

# 3) Extract the true UCpower at *all* these UR’s
#    (this will be NaN if you haven’t run the sim at that UR)
P_all = df_sub.reindex(UR_refined)['UCpower'].to_numpy()

# 4) Split back into orig vs. extras
P_orig    = P_all[np.isin(UR_refined, UR_orig)]
P_refined = P_all  # includes NaNs or real data at 4,6,8.5

# 5) Compute OAOP only if you have no NaNs (or drop them)
mask_good = ~np.isnan(P_refined)
oaop_8pt  = np.trapezoid(P_orig,    x=UR_orig)    / (UR_orig[-1]    - UR_orig[0])
oaop_11pt = np.trapezoid(P_refined[mask_good],
                     x=UR_refined[mask_good]) / (UR_refined[mask_good][-1] - UR_refined[mask_good][0])

delta = (oaop_11pt/oaop_8pt - 1)*100

# 6) Plot the two curves (only non-nan points)
plt.figure(figsize=(8,6))
plt.plot(UR_orig,    P_orig,    '-o', label='Original 8-point', linewidth=2)
plt.plot(UR_refined[mask_good], P_refined[mask_good],
         '--s', label='Refined 11-point', linewidth=2)
plt.xlabel('Reduced Velocity (UR)')
plt.ylabel('UC Power [W]')
plt.title(f'UC Power vs UR (SPU=0.7, SPD=2, ζ=0.2)\nΔOAOP = {delta:.2f}%')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('UCpower_vs_UR_refinement_realdata.png', dpi=300)

def plot_power_total_combined(df, spd_values):

    save_name = "C-P-C-Powertotal-UR_SPU_0.5_0.7"

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Define colors and markers for each SPD
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for each SPD value
    
    for idx, (spu, ax) in enumerate(zip([0.5, 0.7], axs)):
        for spd_idx, spd in enumerate(spd_values):
            power_total_values = []
            damping_ratios = []
            ur_values = []
            
            for damping in df['damping'].unique():
                df_damping = df[(df['damping'] == damping) & (df['SP_U'] == spu) & (df['SP_D'] == spd)]
                
                if not df_damping.empty:
                    max_power_total = df_damping['Total_power'].max()
                    max_ur = df_damping.loc[df_damping['Total_power'] == max_power_total, 'UR'].values[0]
                    
                    power_total_values.append(max_power_total)
                    damping_ratios.append(damping)
                    ur_values.append(max_ur)

            # Plot points and annotations with unique color and marker per SPD
            if power_total_values:
                ax.scatter(damping_ratios, power_total_values, color=colors[spd_idx], marker=markers[spd_idx], label=f'SPD={spd}')
                for x, y, ur in zip(damping_ratios, power_total_values, ur_values):
                    ax.text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
                
                # Fit a polynomial (degree 2) to data and plot it as a dashed line
                p = np.polyfit(damping_ratios, power_total_values, 2)
                fit = np.poly1d(p)
                x_fit = np.linspace(min(damping_ratios), max(damping_ratios), 100)
                ax.plot(x_fit, fit(x_fit), linestyle='--', color=colors[spd_idx], label=f'Fit SPD={spd}')
            
        # Set limits, labels, title, and legend
        ax.set_ylim(0, 200)  # Adjust if needed
        ax.set_xlim(0, 0.4)
        ax.set_xlabel('Damping Ratio')
        ax.set_title(f'Total Power Output vs Damping Ratio for SPU={spu}')
        ax.legend()
        ax.grid(True)

    axs[0].set_ylabel('Power (mW)')
    plt.tight_layout()
    plt.savefig(f"{save_name}_combined.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to release memory
    print(f"Saved plot as {save_name}_combined.png")

# Example usage
plot_power_total_combined(df, spd_values=[1, 2, 3])

def plot_efficiency_total_combined(df, spd_values):

    save_name = "C-P-C-Efficiencytotal-UR_SPU_0.5_0.7"

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Define colors and markers for each SPD
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for each SPD value
    
    for idx, (spu, ax) in enumerate(zip([0.5, 0.7], axs)):
        for spd_idx, spd in enumerate(spd_values):
            efficiency_values = []
            damping_ratios = []
            ur_values = []
            
            for damping in df['damping'].unique():
                df_damping = df[(df['damping'] == damping) & (df['SP_U'] == spu) & (df['SP_D'] == spd)]
                
                if not df_damping.empty:
                    max_efficiency = df_damping['Total_efficiency'].max()
                    max_ur = df_damping.loc[df_damping['Total_efficiency'] == max_efficiency, 'UR'].values[0]
                    
                    efficiency_values.append(max_efficiency)
                    damping_ratios.append(damping)
                    ur_values.append(max_ur)

            # Plot points and annotations with unique color and marker per SPD
            if efficiency_values:
                ax.scatter(damping_ratios, efficiency_values, color=colors[spd_idx], marker=markers[spd_idx], label=f'SPD={spd}')
                for x, y, ur in zip(damping_ratios, efficiency_values, ur_values):
                    ax.text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
                
                # Fit a polynomial (degree 2) to data and plot it as a dashed line
                p = np.polyfit(damping_ratios, efficiency_values, 2)
                fit = np.poly1d(p)
                x_fit = np.linspace(min(damping_ratios), max(damping_ratios), 100)
                ax.plot(x_fit, fit(x_fit), linestyle='--', color=colors[spd_idx], label=f'Fit SPD={spd}')
            
        # Set limits, labels, title, and legend
        ax.set_ylim(0, 30)  # Adjust if needed
        ax.set_xlim(0, 0.4)
        ax.set_xlabel('Damping Ratio')
        ax.set_title(f'Total Efficiency vs Damping Ratio for SPU={spu}')
        ax.legend()
        ax.grid(True)

    axs[0].set_ylabel('Efficiency (%)')
    plt.tight_layout()
    plt.savefig(f"{save_name}_combined.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to release memory

    print(f"Saved plot as {save_name}_combined.png")

# Example usage
plot_efficiency_total_combined(df, spd_values=[1, 2, 3])

def plot_cd_cl_combined(df, spu, spd_values, save_name="cd_cl_vs_ur"):

    # Filter data for damping = 0
    df = df[df['damping'] == 0.1]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Variables to track y-axis limits with margins
    cd_max, cd_min = float('-inf'), float('inf')
    cl_rms_max, cl_rms_min = float('-inf'), float('inf')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values*2)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for different SPDs
    
    for idx, spd in enumerate(spd_values):
        # Filter data for each specific configuration
        config_df = df[(df['SP_U'] == spu) & (df['SP_D'] == spd)]
        
        # Sort by UR to ensure the plot is in ascending order of UR
        config_df = config_df.sort_values(by='UR')
        
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # Upper panel: Plotting Cd-ave against UR with same color for UC and DC, DC points are hollow
        axs[0].plot(config_df['UR'], config_df['Cd-ave1'], marker=marker, linestyle='-', color=color, markerfacecolor='none', label=f'SP_D={spd} (UC)')
        axs[0].plot(config_df['UR'], config_df['Cd-ave2'], marker=marker, linestyle='--', color=color, markerfacecolor='none', label=f'SP_D={spd} (DC)')
        
        # Update y-axis limits for Cd-ave with a margin
        cd_max = max(cd_max, config_df['Cd-ave1'].max(), config_df['Cd-ave2'].max())
        cd_min = min(cd_min, config_df['Cd-ave1'].min(), config_df['Cd-ave2'].min())
        
        # Lower panel: Plotting Cl-rms against UR with same color for UC and DC, DC points are hollow
        axs[1].plot(config_df['UR'], config_df['Cl-rms1'], marker=marker, linestyle='-', color=color, markerfacecolor='none', label=f'SP_D={spd} (UC)')
        axs[1].plot(config_df['UR'], config_df['Cl-rms2'], marker=marker, linestyle='--', color=color, markerfacecolor='none', label=f'SP_D={spd} (DC)')
        
        # Update y-axis limits for Cl-rms with a margin
        cl_rms_max = max(cl_rms_max, config_df['Cl-rms1'].max(), config_df['Cl-rms2'].max())
        cl_rms_min = min(cl_rms_min, config_df['Cl-rms1'].min(), config_df['Cl-rms2'].min())
    
    # Add margins to y-axis limits
    cd_margin = (cd_max - cd_min) * 0.05  # 5% margin
    cl_rms_margin = (cl_rms_max - cl_rms_min) * 0.05  # 5% margin
    
    # Set unified y-axis limits for each panel with margins
    axs[0].set_ylim(cd_min - cd_margin, cd_max + cd_margin)
    axs[1].set_ylim(cl_rms_min - cl_rms_margin, cl_rms_max + cl_rms_margin)
    
    # Customize each subplot
    axs[0].set_ylabel('Cd-ave')
    axs[0].set_title(f'Drag Coefficient (Cd-ave) vs UR (SP_U={spu})')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_xlabel('UR')
    axs[1].set_ylabel('Cl-rms')
    axs[1].set_title(f'Lift Coefficient RMS (Cl-rms) vs UR (SP_U={spu})')
    axs[1].legend()
    axs[1].grid(True)
    
    # Adjust layout, save, and show plot
    plt.tight_layout()
    plt.savefig(f"{save_name}_spu_{spu}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot as {save_name}_spu_{spu}.png")
    
def plot_vibration_frequency_phase_combined(df, spu, spd_values, save_name="vibrationamplitude_frequency_phase1"):

    # Filter data for damping = 0
    df = df[df['damping'] == 0.1]
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
    
    # Variables to track y-axis limits with margins
    y_star_max, y_star_min = float('-inf'), float('inf')
    f_ratio_max, f_ratio_min = float('-inf'), float('inf')
    phase_max, phase_min = float('-inf'), float('inf')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values*2)))  # Use the same colormap to assign consistent colors
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for different SPDs
    
    for idx, spd in enumerate(spd_values):
        # Filter data for each specific configuration
        config_df = df[(df['SP_U'] == spu) & (df['SP_D'] == spd)]
        
        # Sort by UR to ensure the plot is in ascending order of UR
        config_df = config_df.sort_values(by='UR')
        
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # Top row: Plotting Y* (vibration amplitude) against UR
        axs[0, 0].plot(config_df['UR'], config_df['Y*_1'], marker=marker, linestyle='-', color=color, markerfacecolor='none', label=f'SP_D={spd}')
        axs[0, 1].plot(config_df['UR'], config_df['Y*_2'], marker=marker, linestyle='--', color=color, markerfacecolor='none', label=f'SP_D={spd}')
        
        # Update y-axis limits for Y* with a margin
        y_star_max = max(y_star_max, config_df['Y*_1'].max(), config_df['Y*_2'].max())
        y_star_min = min(y_star_min, config_df['Y*_1'].min(), config_df['Y*_2'].min())
        
        # Second row: Plotting frequency ratio against UR
        axs[1, 0].plot(config_df['UR'], config_df['f_ratio_1'], marker=marker, linestyle='-', color=color, markerfacecolor='none', label=f'SP_D={spd}')
        axs[1, 1].plot(config_df['UR'], config_df['f_ratio_2'], marker=marker, linestyle='--', color=color, markerfacecolor='none', label=f'SP_D={spd}')
        
        # Update y-axis limits for frequency ratio with a margin
        f_ratio_max = max(f_ratio_max, config_df['f_ratio_1'].max(), config_df['f_ratio_2'].max())
        f_ratio_min = min(f_ratio_min, config_df['f_ratio_1'].min(), config_df['f_ratio_2'].min())
        
        # Third row: Plotting phase difference against UR
        axs[2, 0].plot(config_df['UR'], config_df['phase_difference_CLY1'], marker=marker, linestyle='-', color=color, markerfacecolor='none', label=f'SP_D={spd}')
        axs[2, 1].plot(config_df['UR'], config_df['phase_difference_CLY2'], marker=marker, linestyle='--', color=color, markerfacecolor='none', label=f'SP_D={spd}')
        
        # Update y-axis limits for phase difference with a margin
        phase_max = max(phase_max, config_df['phase_difference_CLY1'].max(), config_df['phase_difference_CLY2'].max())
        phase_min = min(phase_min, config_df['phase_difference_CLY1'].min(), config_df['phase_difference_CLY2'].min())
    
    # Add margins to y-axis limits
    y_star_margin = (y_star_max - y_star_min) * 0.05  # 5% margin
    f_ratio_margin = (f_ratio_max - f_ratio_min) * 0.05  # 5% margin
    phase_margin = (phase_max - phase_min) * 0.05  # 5% margin
    
    # Set unified y-axis limits for each row with margins
    axs[0, 0].set_ylim(y_star_min - y_star_margin, y_star_max + y_star_margin)
    axs[0, 1].set_ylim(y_star_min - y_star_margin, y_star_max + y_star_margin)
    axs[1, 0].set_ylim(f_ratio_min - f_ratio_margin, f_ratio_max + f_ratio_margin)
    axs[1, 1].set_ylim(f_ratio_min - f_ratio_margin, f_ratio_max + f_ratio_margin)
    axs[2, 0].set_ylim(phase_min - phase_margin, phase_max + phase_margin)
    axs[2, 1].set_ylim(phase_min - phase_margin, phase_max + phase_margin)
    
    # Customize each subplot
    axs[0, 0].set_ylabel('Y*')
    axs[0, 0].set_title(f'Vibration Amplitude Y* (UC) vs UR (SP_U={spu})')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].set_title(f'Vibration Amplitude Y* (DC) vs UR (SP_U={spu})')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].set_ylabel('Frequency Ratio (f/f_n)')
    axs[1, 0].set_title(f'Frequency Ratio (UC) vs UR (SP_U={spu})')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].set_title(f'Frequency Ratio (DC) vs UR (SP_U={spu})')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    axs[2, 0].set_xlabel('UR')
    axs[2, 0].set_ylabel('Phase Difference (degrees)')
    axs[2, 0].set_title(f'Phase Difference (UC) vs UR (SP_U={spu})')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    axs[2, 1].set_xlabel('UR')
    axs[2, 1].set_title(f'Phase Difference (DC) vs UR (SP_U={spu})')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    # Adjust layout, save, and show plot
    plt.tight_layout()
    plt.savefig(f"{save_name}_spu_{spu}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot as {save_name}_spu_{spu}.png")


def plot_ca(df, spu, spd_values, save_name="Ca_vs_ur"):
    # Filter data for damping = 0
    df = df[df['damping'] == 0.1]

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for different SPDs

    # Initialize limits for y-axis
    cd_max, cd_min = -np.inf, np.inf

    for idx, spd in enumerate(spd_values):
        # Filter data for each specific configuration
        config_df = df[(df['SP_U'] == spu) & (df['SP_D'] == spd)]
        
        # Sort by UR to ensure the plot is in ascending order of UR
        config_df = config_df.sort_values(by='UR')
        
        color = colors[idx]
        marker = markers[idx % len(markers)]
        linestyles = ['-', '--', ':']
        # Upper panel: Plotting Ca_uc against UR with filled markers and Ca_dc with hollow markers
        ax[0].plot(config_df['UR'], config_df['Ca_uc1'], marker=marker, linestyle=linestyles[0], color=color, label=f'1SP_D={spd} (UC)')
        ax[0].plot(config_df['UR'], config_df['Ca_uc2'], marker=marker, linestyle=linestyles[2], color=color, label=f'111SP_D={spd} (UC)')
        ax[1].plot(config_df['UR'], config_df['Ca_dc1'], marker=marker, linestyle=linestyles[0], color=color, markerfacecolor='none', label=f'1SP_D={spd} (DC)')
        ax[1].plot(config_df['UR'], config_df['Ca_dc2'], marker=marker, linestyle=linestyles[2], color=color, markerfacecolor='none', label=f'111SP_D={spd} (DC)')

    # Add reference horizontal line at Ca = 0
    ax[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax[1].axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Customize the plot
    ax[0].set_ylabel('Added Mass Coefficient (Ca)')
    ax[0].set_xlabel('UR')
    ax[0].set_title(f'Ca of UC vs UR (SP_U={spu})')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_ylabel('Added Mass Coefficient (Ca)')
    ax[1].set_xlabel('UR')
    ax[1].set_title(f'Ca of DC vs UR (SP_U={spu})')
    ax[1].legend()
    ax[1].grid(True)
    
    # Adjust layout, save, and show plot
    plt.tight_layout()
    plt.savefig(f"{save_name}_spu_{spu}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot as {save_name}_spu_{spu}.png")



def plot_Emean(df, spu, spd_values, save_name="Emean_vs_ur"):
    # Filter data for damping = 0
    df = df[df['damping'] == 0.1]

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for different SPDs

    # Initialize limits for y-axis
    Emean_max, Emean_min = -np.inf, np.inf

    for idx, spd in enumerate(spd_values):
        # Filter data for each specific configuration
        config_df = df[(df['SP_U'] == spu) & (df['SP_D'] == spd)]
        
        # Sort by UR to ensure the plot is in ascending order of UR
        config_df = config_df.sort_values(by='UR')
        
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # Plot Emean_uc and Emean_dc against UR
        ax[0].plot(config_df['UR'], config_df['Emean_uc1'], marker=marker, linestyle='-', color=color, label=f'SP_D={spd} (UC)')

        ax[1].plot(config_df['UR'], config_df['Emean_dc1'], marker=marker, linestyle='--', color=color, markerfacecolor='none', label=f'SP_D={spd} (DC)')
        
        # Update y-axis limits for Emean with a margin
        Emean_max = max(Emean_max, config_df['Emean_uc1'].max(), config_df['Emean_dc2'].max())
        Emean_min = min(Emean_min, config_df['Emean_uc1'].min(), config_df['Emean_dc2'].min())
    
    # Add margins to y-axis limits
    Emean_margin = (Emean_max - Emean_min) * 0.05  # 5% margin
    ax[0].set_ylim(Emean_min - Emean_margin, Emean_max + Emean_margin)
    ax[1].set_ylim(Emean_min - Emean_margin, Emean_max + Emean_margin)

    # Customize the plot
    ax[0].set_ylabel('Mean Energy Transfer (Emean)')
    ax[0].set_title(f'Emean of UC vs UR (SP_U={spu})')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].set_ylabel('Mean Energy Transfer (Emean)')
    ax[1].set_xlabel('UR')
    ax[1].set_title(f'Emean of DC vs UR (SP_U={spu})')
    ax[1].legend()
    # Adjust layout, save, and show plot
    plt.tight_layout()
    fig.savefig(f"{save_name}_spu_{spu}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot as {save_name}_spu_{spu}.png")





def plot_power_output(df, spu, spd_values, damping_values, save_name="PowerOutput_vs_ur"):

    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))  # Use consistent colors for different SPDs
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for different SPDs
    
    for damping in damping_values:
        # Filter data for the specified damping value
        df_damping = df[df['damping'] == damping]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        # Variables to track y-axis limits with margins
        power_max, power_min = float('-inf'), float('inf')
        efficiency_max, efficiency_min = float('-inf'), float('inf')
        
        for idx, spd in enumerate(spd_values):
            # Filter data for each specific configuration
            config_df = df_damping[(df_damping['SP_U'] == spu) & (df_damping['SP_D'] == spd)]
            
            # Sort by UR to ensure the plot is in ascending order of UR
            config_df = config_df.sort_values(by='UR')
            
            color = colors[idx]
            marker = markers[idx % len(markers)]
            
            # Upper-left panel: Plotting P_uc against UR
            axs[0, 0].plot(config_df['UR'], config_df['P_uc'], marker=marker, linestyle='-', color=color, label=f'SP_D={spd}')
            
            # Upper-right panel: Plotting efficiency_uc against UR
            axs[0, 1].plot(config_df['UR'], config_df['efficiency_uc'], marker=marker, linestyle='-', color=color, label=f'SP_D={spd}')
            
            # Lower-left panel: Plotting P_dc against UR
            axs[1, 0].plot(config_df['UR'], config_df['P_dc'], marker=marker, linestyle='--', color=color, label=f'SP_D={spd}')
            
            # Lower-right panel: Plotting efficiency_dc against UR
            axs[1, 1].plot(config_df['UR'], config_df['efficiency_dc'], marker=marker, linestyle='--', color=color, label=f'SP_D={spd}')
            
            # Update y-axis limits for power and efficiency with a margin
            power_max = max(power_max, config_df['P_uc'].max(), config_df['P_dc'].max())
            power_min = min(power_min, config_df['P_uc'].min(), config_df['P_dc'].min())
            efficiency_max = max(efficiency_max, config_df['efficiency_uc'].max(), config_df['efficiency_dc'].max())
            efficiency_min = min(efficiency_min, config_df['efficiency_uc'].min(), config_df['efficiency_dc'].min())
        
        # Add margins to y-axis limits
        power_margin = (power_max - power_min) * 0.05  # 5% margin
        efficiency_margin = (efficiency_max - efficiency_min) * 0.05  # 5% margin
        
        # Set unified y-axis limits for each row with margins
        axs[0, 0].set_ylim(power_min - power_margin, power_max + power_margin)
        axs[0, 1].set_ylim(efficiency_min - efficiency_margin, efficiency_max + efficiency_margin)
        axs[1, 0].set_ylim(power_min - power_margin, power_max + power_margin)
        axs[1, 1].set_ylim(efficiency_min - efficiency_margin, efficiency_max + efficiency_margin)
        
        # Customize each subplot
        axs[0, 0].set_ylabel('Power Output (P_uc)')
        axs[0, 0].set_title(f'Power Output UC vs UR (SP_U={spu}, Damping={damping})')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        axs[0, 1].set_ylabel('Efficiency (UC)')
        axs[0, 1].set_title(f'Efficiency UC vs UR (SP_U={spu}, Damping={damping})')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        axs[1, 0].set_ylabel('Power Output (P_dc)')
        axs[1, 0].set_xlabel('UR')
        axs[1, 0].set_title(f'Power Output DC vs UR (SP_U={spu}, Damping={damping})')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        axs[1, 1].set_ylabel('Efficiency (DC)')
        axs[1, 1].set_xlabel('UR')
        axs[1, 1].set_title(f'Efficiency DC vs UR (SP_U={spu}, Damping={damping})')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        # Adjust layout, save, and show plot
        plt.tight_layout()
        plt.savefig(f"{save_name}_spu_{spu}_damping_{damping}.png", dpi=300)
        plt.show()
        plt.close(fig)
        print(f"Saved plot as {save_name}_spu_{spu}_damping_{damping}.png")




def plot_peak_efficiency_vs_damping_multiple_spd(df, spu, spd_values):
    """
    Finds the maximum efficiency for UC and DC at specific SPD values and the corresponding reduced velocity (UR) for each damping ratio,
    then plots the peak efficiency against the damping ratio in multiple subplots with UR annotations and a dashed fit line.

    Parameters:
    df : DataFrame
        The merged data containing all configurations.
    spu : float
        The upstream spacing ratio to filter the data.
    spd_values : list of float
        The downstream spacing ratios to filter and plot in separate subplots.
    """
    save_name = f"C-P-C-Peakefficiency-UR_SPU_{spu}"

    fig, axs = plt.subplots(len(spd_values), 2, figsize=(16, 12), sharex=True)
    
    for i, spd in enumerate(spd_values):
        peak_efficiency_uc = []
        peak_efficiency_dc = []
        damping_ratios = []
        ur_uc_values = []
        ur_dc_values = []
        
        for damping in df['damping'].unique():
            df_damping = df[(df['damping'] == damping) & (df['SP_U'] == spu) & (df['SP_D'] == spd)]
            
            if not df_damping.empty:
                max_efficiency_uc = df_damping['efficiency_uc'].max()
                max_ur_uc = df_damping[df_damping['efficiency_uc'] == max_efficiency_uc]['UR'].values[0]
                
                max_efficiency_dc = df_damping['efficiency_dc'].max()
                max_ur_dc = df_damping[df_damping['efficiency_dc'] == max_efficiency_dc]['UR'].values[0]
                
                peak_efficiency_uc.append(max_efficiency_uc)
                peak_efficiency_dc.append(max_efficiency_dc)
                damping_ratios.append(damping)
                ur_uc_values.append(max_ur_uc)
                ur_dc_values.append(max_ur_dc)

        if peak_efficiency_uc:
            # Plot points and annotations for UC
            axs[i, 0].scatter(damping_ratios, peak_efficiency_uc, color='blue', label=f'SPD={spd}')
            for x, y, ur in zip(damping_ratios, peak_efficiency_uc, ur_uc_values):
                axs[i, 0].text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
            
            # Fit a polynomial (degree 2) to UC data and plot it as a dashed line
            p_uc = np.polyfit(damping_ratios, peak_efficiency_uc, 2)
            fit_uc = np.poly1d(p_uc)
            x_fit = np.linspace(min(damping_ratios), max(damping_ratios), 100)
            axs[i, 0].plot(x_fit, fit_uc(x_fit), 'b--', label='Fit (UC)')
            
            axs[i, 0].set_ylim(0, 20)
            axs[i, 0].set_xlim(0, 0.4)
            axs[i, 0].set_ylabel('Peak Efficiency (UC)')
            axs[i, 0].set_title(f'Peak Efficiency vs Damping Ratio for UC at SPD={spd}')
            axs[i, 0].legend()
            axs[i, 0].grid(True)
        
        if peak_efficiency_dc:
            # Plot points and annotations for DC
            axs[i, 1].scatter(damping_ratios, peak_efficiency_dc, color='red', label=f'SPD={spd}')
            for x, y, ur in zip(damping_ratios, peak_efficiency_dc, ur_dc_values):
                axs[i, 1].text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
            
            # Fit a polynomial (degree 2) to DC data and plot it as a dashed line
            p_dc = np.polyfit(damping_ratios, peak_efficiency_dc, 2)
            fit_dc = np.poly1d(p_dc)
            x_fit = np.linspace(min(damping_ratios), max(damping_ratios), 100)
            axs[i, 1].plot(x_fit, fit_dc(x_fit), 'r--', label='Fit (DC)')
            
            axs[i, 1].set_ylim(0, 20)
            axs[i, 1].set_xlim(0, 0.4)
            axs[i, 1].set_ylabel('Peak Efficiency (DC)')
            axs[i, 1].set_title(f'Peak Efficiency vs Damping Ratio for DC at SPD={spd}')
            axs[i, 1].legend()
            axs[i, 1].grid(True)

    axs[-1, 0].set_xlabel('Damping Ratio')
    axs[-1, 1].set_xlabel('Damping Ratio')
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved plot as {save_name}.png")

# Example usage
plot_peak_efficiency_vs_damping_multiple_spd(df, spu=0.5, spd_values=[1, 2, 3])
plot_peak_efficiency_vs_damping_multiple_spd(df, spu=0.7, spd_values=[1, 2, 3])



def plot_peak_efficiency_vs_damping_single_plot1(df, spu, spd_values):
    """
    Finds the maximum efficiency for UC and DC at specific SPD values and the corresponding reduced velocity (UR) for each damping ratio,
    then plots the peak efficiency against the damping ratio in a single plot with different colors and markers for each SPD.

    Parameters:
    df : DataFrame
        The merged data containing all configurations.
    spu : float
        The upstream spacing ratio to filter the data.
    spd_values : list of float
        The downstream spacing ratios to filter and plot in a single combined plot.
    """
    save_name = f"C-P-C-Peakefficiency1-UR_SPU_{spu}"

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    # Define colors and markers for each SPD
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for each SPD value
    
    for idx, spd in enumerate(spd_values):
        peak_efficiency_uc = []
        peak_efficiency_dc = []
        damping_ratios = []
        ur_uc_values = []
        ur_dc_values = []
        
        for damping in df['damping'].unique():
            df_damping = df[(df['damping'] == damping) & (df['SP_U'] == spu) & (df['SP_D'] == spd)]
            
            if not df_damping.empty:
                max_efficiency_uc = df_damping['efficiency_uc'].max()
                max_ur_uc = df_damping[df_damping['efficiency_uc'] == max_efficiency_uc]['UR'].values[0]
                
                max_efficiency_dc = df_damping['efficiency_dc'].max()
                max_ur_dc = df_damping[df_damping['efficiency_dc'] == max_efficiency_dc]['UR'].values[0]
                
                peak_efficiency_uc.append(max_efficiency_uc)
                peak_efficiency_dc.append(max_efficiency_dc)
                damping_ratios.append(damping)
                ur_uc_values.append(max_ur_uc)
                ur_dc_values.append(max_ur_dc)

        # Plot points and annotations for UC with unique color and marker per SPD
        if peak_efficiency_uc:
            axs[0].scatter(damping_ratios, peak_efficiency_uc, color=colors[idx], marker=markers[idx], label=f'SPD={spd}')
            for x, y, ur in zip(damping_ratios, peak_efficiency_uc, ur_uc_values):
                axs[0].text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
            
            # Fit a polynomial (degree 2) to UC data and plot it as a dashed line
            p_uc = np.polyfit(damping_ratios, peak_efficiency_uc, 2)
            fit_uc = np.poly1d(p_uc)
            x_fit = np.linspace(min(damping_ratios), max(damping_ratios), 100)
            axs[0].plot(x_fit, fit_uc(x_fit), linestyle='--', color=colors[idx], label=f'Fit (UC) SPD={spd}')
        
        # Plot points and annotations for DC with unique color and marker per SPD
        if peak_efficiency_dc:
            axs[1].scatter(damping_ratios, peak_efficiency_dc, color=colors[idx], marker=markers[idx], label=f'SPD={spd}')
            for x, y, ur in zip(damping_ratios, peak_efficiency_dc, ur_dc_values):
                axs[1].text(x, y, f'Ur = {ur}', ha='right', va='bottom', fontsize=8)
            
            # Fit a polynomial (degree 2) to DC data and plot it as a dashed line
            p_dc = np.polyfit(damping_ratios, peak_efficiency_dc, 2)
            fit_dc = np.poly1d(p_dc)
            axs[1].plot(x_fit, fit_dc(x_fit), linestyle='--', color=colors[idx], label=f'Fit (DC) SPD={spd}')

    # Set limits, labels, titles, and legends for both UC and DC plots
    for ax, title in zip(axs, ["UC", "DC"]):
        ax.set_ylim(0, 20)
        ax.set_xlim(0, 0.4)
        ax.set_xlabel('Damping Ratio')
        ax.set_ylabel(f'Peak Efficiency ({title})')
        ax.set_title(f'Peak Efficiency vs Damping Ratio for {title}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_name}_combined.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to release memory

    print(f"Saved plot as {save_name}_combined.png")

# Example usage
plot_peak_efficiency_vs_damping_single_plot1(df, spu=0.5, spd_values=[1, 2, 3])
plot_peak_efficiency_vs_damping_single_plot1(df, spu=0.7, spd_values=[1, 2, 3])


def plot_vibration_frequency_phase_fixed_spd(df, spu, spd, damping_values, save_name="vibrationamplitude_frequency_phase"):
    """
    Plots the vibration amplitude Y*, frequency ratio f_ratio, and phase difference against UR for a fixed downstream spacing ratio (SPD)
    with specified damping ratios in a single 3x2 grid.

    Parameters:
    df : DataFrame
        The merged data containing all configurations.
    spu : float
        The upstream spacing ratio to filter the data.
    spd : float
        The downstream spacing ratio to filter and plot.
    damping_values : list of float
        The specific damping ratios to plot.
    save_name : str
        The base name to use when saving the figure.
    """
    # Filter data for the fixed SPD value and specified damping ratios
    df = df[(df['SP_U'] == spu) & (df['SP_D'] == spd) & (df['damping'].isin(damping_values))]
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
    
    # Define colors and markers for each damping ratio
    colors = plt.cm.tab10(np.linspace(0, 1, len(damping_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for each damping ratio
    
    # Initialize limits for consistent y-axis scaling across plots
    y_star_max, y_star_min = float('-inf'), float('inf')
    f_ratio_max, f_ratio_min = float('-inf'), float('inf')
    phase_max, phase_min = float('-inf'), float('inf')
    
    for idx, damping in enumerate(damping_values):
        # Filter data for each damping ratio
        config_df = df[df['damping'] == damping].sort_values(by='UR')
        
        # Set color and marker for this damping ratio
        color = colors[idx]
        marker = markers[idx % len(markers)]
        
        # Top row: Plotting Y* (vibration amplitude) against UR
        axs[0, 0].plot(config_df['UR'], config_df['Y*_1'], marker=marker, linestyle='-', color=color, label=f'Damping={damping}')
        axs[0, 1].plot(config_df['UR'], config_df['Y*_2'], marker=marker, linestyle='--', color=color, label=f'Damping={damping}')
        
        # Update y-axis limits for Y*
        y_star_max = max(y_star_max, config_df['Y*_1'].max(), config_df['Y*_2'].max())
        y_star_min = min(y_star_min, config_df['Y*_1'].min(), config_df['Y*_2'].min())
        
        # Middle row: Plotting frequency ratio against UR
        axs[1, 0].plot(config_df['UR'], config_df['f_ratio_1'], marker=marker, linestyle='-', color=color, label=f'Damping={damping}')
        axs[1, 1].plot(config_df['UR'], config_df['f_ratio_2'], marker=marker, linestyle='--', color=color, label=f'Damping={damping}')
        
        # Update y-axis limits for frequency ratio
        f_ratio_max = max(f_ratio_max, config_df['f_ratio_1'].max(), config_df['f_ratio_2'].max())
        f_ratio_min = min(f_ratio_min, config_df['f_ratio_1'].min(), config_df['f_ratio_2'].min())
        
        # Bottom row: Plotting phase difference against UR
        axs[2, 0].plot(config_df['UR'], config_df['phase_difference_CLY1'], marker=marker, linestyle='-', color=color, label=f'Damping={damping}')
        axs[2, 1].plot(config_df['UR'], config_df['phase_difference_CLY2'], marker=marker, linestyle='--', color=color, label=f'Damping={damping}')
        
        # Update y-axis limits for phase difference
        phase_max = max(phase_max, config_df['phase_difference_CLY1'].max(), config_df['phase_difference_CLY2'].max())
        phase_min = min(phase_min, config_df['phase_difference_CLY1'].min(), config_df['phase_difference_CLY2'].min())
    
    # Add margins to y-axis limits
    y_star_margin = (y_star_max - y_star_min) * 0.05  # 5% margin
    f_ratio_margin = (f_ratio_max - f_ratio_min) * 0.05  # 5% margin
    phase_margin = (phase_max - phase_min) * 0.05  # 5% margin
    
    # Set unified y-axis limits for each row with margins
    axs[0, 0].set_ylim(y_star_min - y_star_margin, y_star_max + y_star_margin)
    axs[0, 1].set_ylim(y_star_min - y_star_margin, y_star_max + y_star_margin)
    axs[1, 0].set_ylim(f_ratio_min - f_ratio_margin, f_ratio_max + f_ratio_margin)
    axs[1, 1].set_ylim(f_ratio_min - f_ratio_margin, f_ratio_max + f_ratio_margin)
    axs[2, 0].set_ylim(phase_min - phase_margin, phase_max + phase_margin)
    axs[2, 1].set_ylim(phase_min - phase_margin, phase_max + phase_margin)
    
    # Customize each subplot
    axs[0, 0].set_ylabel('Y*')
    axs[0, 0].set_title(f'Vibration Amplitude Y* (UC) vs UR (SP_U={spu}, SP_D={spd})')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].set_title(f'Vibration Amplitude Y* (DC) vs UR (SP_U={spu}, SP_D={spd})')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].set_ylabel('Frequency Ratio (f/f_n)')
    axs[1, 0].set_title(f'Frequency Ratio (UC) vs UR (SP_U={spu}, SP_D={spd})')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].set_title(f'Frequency Ratio (DC) vs UR (SP_U={spu}, SP_D={spd})')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    axs[2, 0].set_xlabel('UR')
    axs[2, 0].set_ylabel('Phase Difference (degrees)')
    axs[2, 0].set_title(f'Phase Difference (UC) vs UR (SP_U={spu}, SP_D={spd})')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    axs[2, 1].set_xlabel('UR')
    axs[2, 1].set_title(f'Phase Difference (DC) vs UR (SP_U={spu}, SP_D={spd})')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    # Adjust layout, save, and show plot
    plt.tight_layout()
    filename = f"{save_name}_SPU_{spu}_SPD_{spd}.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)  # Close figure to release memory
    print(f"Saved plot as {filename}")

# Example usage
# Assuming `df` is your DataFrame and contains columns 'damping', 'SP_U', 'SP_D', 'UR', 'Y*_1', 'Y*_2', 'f_ratio_1', 'f_ratio_2', 'phase_difference_CLY1', 'phase_difference_CLY2'
plot_vibration_frequency_phase_fixed_spd(df, spu=0.5, spd=1, damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])
plot_vibration_frequency_phase_fixed_spd(df, spu=0.5, spd=2, damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])
plot_vibration_frequency_phase_fixed_spd(df, spu=0.5, spd=3, damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])

plot_vibration_frequency_phase_fixed_spd(df, spu=0.7, spd=1, damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])
plot_vibration_frequency_phase_fixed_spd(df, spu=0.7, spd=2, damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])
plot_vibration_frequency_phase_fixed_spd(df, spu=0.7, spd=3, damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])




# Example usage for multiple downstream spacing ratios and damping values
plot_power_output(df, spu=0.5, spd_values=[1, 2, 3], damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])
plot_power_output(df, spu=0.7, spd_values=[1, 2, 3], damping_values=[0.1, 0.15, 0.2, 0.25, 0.3])

# Example usage for multiple downstream spacing ratios with damping=0 filter
plot_vibration_frequency_phase_combined(df, spu=0.5, spd_values=[1, 2, 3])
plot_vibration_frequency_phase_combined(df, spu=0.7, spd_values=[1, 2, 3])


# Example usage for multiple downstream spacing ratios
plot_cd_cl_combined(df, spu=0.5, spd_values=[1, 2, 3])
plot_cd_cl_combined(df, spu=0.7, spd_values=[1, 2, 3])


# Example usage for multiple downstream spacing ratios
plot_ca(df, spu=0.5, spd_values=[1, 2, 3])
plot_ca(df, spu=0.7, spd_values=[1, 2, 3])


plot_Emean(df, spu=0.5, spd_values=[1, 2, 3])
plot_Emean(df, spu=0.7, spd_values=[1, 2, 3])






def compute_average_power(df, damping, spu, spd):
    filtered_df = df[(df['damping']==damping) &
                     (df['SP_U']==spu) &
                     (df['SP_D']==spd)]
    avg_uc = filtered_df['P_uc'].mean()
    avg_dc = filtered_df['P_dc'].mean()
    return avg_uc, avg_dc, avg_uc + avg_dc

# — load all isolated‐cylinder CSVs into one DataFrame —
def load_isolated_cylinder(ic_root):
    frames = []
    for sub in os.listdir(ic_root):
        subdir = os.path.join(ic_root, sub)
        if not os.path.isdir(subdir) or not sub.startswith('Single_VIV'):
            continue
        for fname in os.listdir(subdir):
            if fname.startswith('computation_results') and fname.endswith('.csv'):
                df_tmp = pd.read_csv(os.path.join(subdir, fname))
                df_tmp['Total_power'] = df_tmp['P_uc']
                frames.append(df_tmp)
    if not frames:
        raise RuntimeError("No isolated‐cylinder data found!")
    return pd.concat(frames, ignore_index=True)

# 1) load your IC data
ic_dir = '/ibex/scratch/projects/c2298/energy_harvesting/single_viv'
df_ic = load_isolated_cylinder(ic_dir)
# keep only true iso geometry
df_ic0 = df_ic[(df_ic['SP_U']==0)&(df_ic['SP_D']==0)]

# 2) load your CPC data

df['P_uc'] = df['P_uc']    # ensure present
df['P_dc'] = df['P_dc']
df['Total_power'] = df['P_uc'] + df['P_dc']

# 3) precompute isolated‐cylinder average lists
damping_values = sorted(df['damping'].unique())
iso_avg_uc   = [ df_ic0[df_ic0['damping']==d]['P_uc'].mean() for d in damping_values ]
iso_avg_dc   = [ u for u in iso_avg_uc ]
iso_avg_total= [ u for u in iso_avg_uc ]  # same as UC

# 4) modified plotting function
def plot_average_power_vs_damping_with_IC(df, spd_values):
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    spu_values = [0.5, 0.7]
    colors = plt.cm.tab10(np.linspace(0,1,len(spd_values)))
    markers = ['o','s','^','D','*']

    for col_idx, spu in enumerate(spu_values):
        # for each CPC SPD
        for spd_idx, spd in enumerate(spd_values):
            uc_list, dc_list, tot_list = [], [], []
            for d in damping_values:
                uc, dc, tot = compute_average_power(df, d, spu, spd)
                uc_list.append(uc); dc_list.append(dc); tot_list.append(tot)
            axs[0,col_idx].plot(damping_values, uc_list, marker=markers[spd_idx],
                                linestyle='-',  color=colors[spd_idx], label=f'SPD={spd}')
            axs[1,col_idx].plot(damping_values, dc_list, marker=markers[spd_idx],
                                linestyle='--', color=colors[spd_idx], label=f'SPD={spd}')
            axs[2,col_idx].plot(damping_values, tot_list, marker=markers[spd_idx],
                                linestyle=':',  color=colors[spd_idx], label=f'SPD={spd}')

        # overlay isolated‐cylinder
        axs[0,col_idx].plot(damping_values, iso_avg_uc,
                            'k--', linewidth=2, marker='X', label='Isolated')
        axs[1,col_idx].plot(damping_values, iso_avg_dc,
                            'k--', linewidth=2, marker='X', label='Isolated')
        axs[2,col_idx].plot(damping_values, iso_avg_total,
                            'k--', linewidth=2, marker='X', label='Isolated')

        # titles / legends / grid
        axs[0,col_idx].set_title(f'UC avg power vs damping (SPU={spu})')
        axs[1,col_idx].set_title(f'DC avg power vs damping (SPU={spu})')
        axs[2,col_idx].set_title(f'Total avg power vs damping (SPU={spu})')
        for ax in axs[:,col_idx]:
            ax.legend()
            ax.grid(True)

    # fixed limits
    for ax in axs[0,:]:
        ax.set_ylim(0,50); ax.set_xlim(min(damping_values), max(damping_values))
    for ax in axs[1,:]:
        ax.set_ylim(0,60); ax.set_xlim(min(damping_values), max(damping_values))
    for ax in axs[2,:]:
        ax.set_ylim(0,100); ax.set_xlim(min(damping_values), max(damping_values))

    # labels
    axs[2,0].set_xlabel('Damping Ratio'); axs[2,1].set_xlabel('Damping Ratio')
    axs[0,0].set_ylabel('Avg Power UC'); axs[1,0].set_ylabel('Avg Power DC')
    axs[2,0].set_ylabel('Total Avg Power')

    plt.tight_layout()
    plt.savefig("average_power_vs_damping_with_IC.png", dpi=300)
    plt.show()
    plt.close(fig)
    print("Saved plot as average_power_vs_damping_with_IC.png")

# 5) call it
plot_average_power_vs_damping_with_IC(df, spd_values=[1,2,3])


def plot_peak_power_vs_damping(df, spd_values):
    """
    Plot the peak power output for UC, DC, and total power against damping ratio for different SPU values (0.5 and 0.7).
    The plot has three rows: top for UC, middle for DC, and bottom for total power.

    Parameters:
    df : DataFrame
        The merged data containing all configurations.
    spd_values : list of float
        The downstream spacing ratios to filter and plot in each subplot.
    """
    save_name = "C-P-C-PeakPower-UR_SPU_0.5_0.7"
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    # Define SPU values for left (0.5) and right (0.7) columns
    spu_values = [0.5, 0.7]
    colors = plt.cm.tab10(np.linspace(0, 1, len(spd_values)))
    markers = ['o', 's', '^', 'D', '*']  # Different marker styles for each SPD value
    
    for col_idx, spu in enumerate(spu_values):
        for spd_idx, spd in enumerate(spd_values):
            peak_power_uc_list = []
            peak_power_dc_list = []
            peak_total_power_list = []
            damping_ratios = sorted(df['damping'].unique())
            
            for damping in damping_ratios:
                df_damping = df[(df['damping'] == damping) & (df['SP_U'] == spu) & (df['SP_D'] == spd)]
                
                if not df_damping.empty:
                    # Compute peak values for UC, DC, and total power
                    peak_power_uc = df_damping['P_uc'].max()
                    peak_power_dc = df_damping['P_dc'].max()
                    peak_total_power = df_damping['Total_power'].max()
                    
                    peak_power_uc_list.append(peak_power_uc)
                    peak_power_dc_list.append(peak_power_dc)
                    peak_total_power_list.append(peak_total_power)

            # Plot peak UC power in the upper panel
            axs[0, col_idx].plot(damping_ratios, peak_power_uc_list, marker=markers[spd_idx], linestyle='-', color=colors[spd_idx], label=f'SPD={spd}')
            # Plot peak DC power in the middle panel
            axs[1, col_idx].plot(damping_ratios, peak_power_dc_list, marker=markers[spd_idx], linestyle='--', color=colors[spd_idx], label=f'SPD={spd}')
            # Plot total peak power in the lower panel
            axs[2, col_idx].plot(damping_ratios, peak_total_power_list, marker=markers[spd_idx], linestyle=':', color=colors[spd_idx], label=f'SPD={spd}')
        
        # Set titles and legends for each SPU column
        axs[0, col_idx].set_title(f'Peak Power Output (UC) vs Damping Ratio for SPU={spu}')
        axs[1, col_idx].set_title(f'Peak Power Output (DC) vs Damping Ratio for SPU={spu}')
        axs[2, col_idx].set_title(f'Total Peak Power Output vs Damping Ratio for SPU={spu}')
        for ax in axs[:, col_idx]:
            ax.legend()
            ax.grid(True)

    # Set specific y-axis limits
    for ax in axs[0, :]:  # Upper panel (UC)
        ax.set_ylim(0, 100)
        ax.set_xlim(0.05, 0.35)
    for ax in axs[1, :]:  # Middle panel (DC)
        ax.set_ylim(0, 150)
        ax.set_xlim(0.05, 0.35)
    for ax in axs[2, :]:  # Lower panel (Total)
        ax.set_ylim(0, 200)
        ax.set_xlim(0.05, 0.35)

    # Set labels
    axs[2, 0].set_xlabel('Damping Ratio')
    axs[2, 1].set_xlabel('Damping Ratio')
    axs[0, 0].set_ylabel('Peak Power UC')
    axs[1, 0].set_ylabel('Peak Power DC')
    axs[2, 0].set_ylabel('Total Peak Power')

    plt.tight_layout()
    plt.savefig(f"{save_name}_combined.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to release memory
    print(f"Saved plot as {save_name}_combined.png")

# Example usage
plot_peak_power_vs_damping(df, spd_values=[1, 2, 3])



# ===================================================================
# === NEW ADDITION === OAOP percentage & OAOE percentage breakdown
# ===================================================================


def plot_OAOP_and_OAOE_breakdown(
    df,
    spu_values,
    spd_values,
    damping_values,
    ur_min=3,
    ur_max=30,
    save_prefix="CPC_barplots"
):
    """
    Creates two grouped bar plots:
      (1) % contribution of UC/DC to total OAOP
      (2) % contribution of UC/DC to total OAOE (efficiency-based)
    For each SPU (0.5, 0.7) and SPD (1, 2, 3).
    """

    fig1, axes1 = plt.subplots(3, len(spu_values), figsize=(14, 10), sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(3, len(spu_values), figsize=(14, 10), sharex=True, sharey=True)

    for col, spu in enumerate(spu_values):
        for row, spd in enumerate(spd_values):
            uc_OAOP_pct, dc_OAOP_pct = [], []
            uc_OAOE_pct, dc_OAOE_pct, tot_OAOE_pct = [], [], []

            for d in damping_values:
                # --- Integrated OAOPs ---
                oaop_uc  = compute_oaop(df, spu, spd, d, ur_min, ur_max, 'P_uc')
                oaop_dc  = compute_oaop(df, spu, spd, d, ur_min, ur_max, 'P_dc')
                oaop_tot = compute_oaop(df, spu, spd, d, ur_min, ur_max, 'Total_power')

                # --- Weighted average efficiencies ---
                oaoe_uc  = compute_avg_efficiency(df, spu, spd, d, ur_min, ur_max, 'efficiency_uc')
                oaoe_dc  = compute_avg_efficiency(df, spu, spd, d, ur_min, ur_max, 'efficiency_dc')
                oaoe_tot = compute_avg_efficiency(df, spu, spd, d, ur_min, ur_max, 'Total_efficiency')

                # OAOP % contributions
                if oaop_tot and not np.isnan(oaop_tot) and oaop_tot > 0:
                    uc_OAOP_pct.append(100 * oaop_uc / oaop_tot)
                    dc_OAOP_pct.append(100 * oaop_dc / oaop_tot)
                else:
                    uc_OAOP_pct.append(np.nan)
                    dc_OAOP_pct.append(np.nan)

                # OAOE % contributions (UC/DC vs total OAOE)
                if (oaoe_uc + oaoe_dc) > 0:
                    uc_OAOE_pct.append(100 * oaoe_uc / (oaoe_uc + oaoe_dc))
                    dc_OAOE_pct.append(100 * oaoe_dc / (oaoe_uc + oaoe_dc))
                else:
                    uc_OAOE_pct.append(np.nan)
                    dc_OAOE_pct.append(np.nan)

                tot_OAOE_pct.append(100 * oaoe_tot)  # absolute efficiency (%)

            # Bar positions
            x = np.arange(len(damping_values))
            width = 0.35

            # --- FIGURE 1: OAOP contribution ---
            ax1 = axes1[row, col]
            ax1.bar(x - width/2, uc_OAOP_pct, width, label='UC', color='forestgreen')
            ax1.bar(x + width/2, dc_OAOP_pct, width, label='DC', color='limegreen', alpha=0.7)
            ax1.set_ylim(0, 120)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{ζ:.2f}" for ζ in damping_values])
            ax1.set_ylabel(f"SPD={spd}")
            if row == 0:
                ax1.set_title(f"SPU={spu}")
            if row == len(spd_values) - 1:
                ax1.set_xlabel("Damping Ratio ζ")
            if col == 0 and row == 1:
                ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

            # --- FIGURE 2: OAOE contribution ---
            ax2 = axes2[row, col]
            ax2.bar(x - width/2, uc_OAOE_pct, width, label='UC', color='royalblue')
            ax2.bar(x + width/2, dc_OAOE_pct, width, label='DC', color='skyblue', alpha=0.7)
            ax2.plot(x, tot_OAOE_pct, 'ko--', lw=1.5, ms=4, label='Total OAOE (%)')
            ax2.set_ylim(0, 120)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{ζ:.2f}" for ζ in damping_values])
            ax2.set_ylabel(f"SPD={spd}")
            if row == 0:
                ax2.set_title(f"SPU={spu}")
            if row == len(spd_values) - 1:
                ax2.set_xlabel("Damping Ratio ζ")
            if col == 0 and row == 1:
                ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Layout and save
    fig1.suptitle("Percentage Contribution of UC/DC to Total OAOP", fontsize=16, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig1.savefig(f"{save_prefix}_OAOP_contribution.png", dpi=300)
    plt.close(fig1)
    print(f"✅ Saved OAOP contribution chart as {save_prefix}_OAOP_contribution.png")

    fig2.suptitle("Percentage Contribution of UC/DC to Total OAOE (and absolute total %)", fontsize=16, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig2.savefig(f"{save_prefix}_OAOE_contribution.png", dpi=300)
    plt.close(fig2)
    print(f"✅ Saved OAOE contribution chart as {save_prefix}_OAOE_contribution.png")


# ── Example usage ─────────────────────────────────────────────────────────────
plot_OAOP_and_OAOE_breakdown(
    df,
    spu_values=[0.5, 0.7],
    spd_values=[1, 2, 3],
    damping_values=[0.10, 0.15, 0.20, 0.25, 0.30],
    ur_min=3,
    ur_max=30,
    save_prefix="CPC_barplot"
)
