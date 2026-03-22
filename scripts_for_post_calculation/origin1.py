
# Define the base directory and post directory
base_dir = "/ibex/user/gongm/projects/c-sp-c/IC"

import pandas as pd
import os
import shutil

# 1) Define directories

post_dir = os.path.join(base_dir, "post")
results_dir = os.path.join(post_dir, "results")

# ensure post/ and post/results/ both exist
os.makedirs(results_dir, exist_ok=True)

def is_target_file(filename):
    """
    Returns True if filename matches 'computation_results3_<SPU>_<SPD>_<UR>_<damping>.csv'
    """
    if not (filename.startswith('computation_results3_') and filename.endswith('.csv')):
        return False
    parts = filename[len('computation_results3_'):-len('.csv')].split('_')
    if len(parts) != 4:
        return False
    try:
        [float(x) for x in parts]
        return True
    except ValueError:
        return False

# 2) Walk base_dir (but skip post/ and post/results/) and move matching files
for root, dirs, files in os.walk(base_dir):
    # prevent recursing into post_dir and results_dir
    if 'post' in dirs:
        dirs.remove('post')
    if 'results' in dirs:
        dirs.remove('results')

    for fname in files:
        if is_target_file(fname):
            src = os.path.join(root, fname)
            dst = os.path.join(results_dir, fname)
            # only copy if it's not already in results_dir
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                print(f"Copied: {src} → {dst}")

# 3) Helpers to parse and read
def parse_filename(filename):
    """Extract SPU, SPD, UR, damping from 'computation_results3_<SPU>_<SPD>_<UR>_<damping>.csv'"""
    parts = filename[len('computation_results3_'):-len('.csv')].split('_')
    return tuple(float(x) for x in parts)

def read_and_tag_file(filepath):
    """Read CSV and add columns SP_U, SP_D, UR, damping based on filename"""
    df = pd.read_csv(filepath)
    SPU, SPD, UR, damping = parse_filename(os.path.basename(filepath))
    df['SP_U'] = SPU
    df['SP_D'] = SPD
    df['UR']   = UR
    df['damping'] = damping
    return df

# 4) Aggregate all moved files into a single DataFrame
all_data = []
for fname in os.listdir(results_dir):
    if is_target_file(fname):
        fp = os.path.join(results_dir, fname)
        all_data.append(read_and_tag_file(fp))

if not all_data:
    raise RuntimeError("No computation_results3_ files found in post/results/")

full_data = pd.concat(all_data, ignore_index=True)

# 5) Sort and write final_results.csv into post_dir
sorted_data = full_data.sort_values(by=['SP_U', 'SP_D', 'UR'])
final_fp = os.path.join(post_dir, 'final_results.csv')
sorted_data.to_csv(final_fp, index=False)
print(f"Merged data written to {final_fp}")

