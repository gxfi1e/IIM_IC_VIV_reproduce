import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.fft import fft, fftfreq
from scipy.signal import (
    argrelextrema,
    savgol_filter,
    hilbert,
    find_peaks,
    butter,
    filtfilt
)
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from numpy import trapz
import matplotlib.gridspec as gridspec
import scipy.integrate
from scipy.integrate import simpson
from scipy.signal import detrend
from matplotlib.ticker import MaxNLocator


red1 = "#f03e3e"
red2 = "#ff6b6b"
red3 = "#ffa8a8"

yellow1 = "#a97426"
yellow2 = "#d29c2f"
yellow3 = "#f5df7a"

blue1 = "#3163eb"
blue2 = "#84a1f9"
blue3 = "#adbffb"

green1 = "#2f9969"
green2 = "#52b785"
green3 = "#c0f1d3"

black1 = "#646464"
black2 = "#696969"
black3 = "#cbcdd1"

def read_and_evaluate_parameters(filename):
    params = {}
    line_count = 0
    with open(filename, 'r') as file:
        for line in file:
            if line_count >= 60:
                break
            original_line = line.strip()
            line = line.split('//')[0].strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                try:
                    value = eval(value.strip(), {"__builtins__": None, "sqrt": math.sqrt, "PI": math.pi}, params)
                    params[key] = value
                except Exception as e:
                    print(f"Error processing line: {original_line}")
                    print(f"Error message: {e}")
            line_count += 1
    return params

def filter_last_n_seconds(data, end_time):
    if end_time is not None:
        start_time = end_time - sampling_duration
        return data[(data[0] >= start_time) & (data[0] <= end_time)]
    return data

def compute_oscillation_frequency(displacement, sampling_interval):
    displacement_fft = fft(displacement)
    freq = fftfreq(len(displacement), d=sampling_interval)
    fosc = freq[np.argmax(np.abs(displacement_fft))]
    return fosc

def plot_fft_spectrum(signal, sampling_interval, label, color, ax, f_n, x_range=None, linestyle='-', fontsize=14):
    # Compute FFT of the signal
    signal_fft = fft(signal)
    # Compute frequency values for the x-axis
    freq = fftfreq(len(signal), d=sampling_interval)
    # Compute amplitude spectrum (absolute value of FFT)
    amplitude = np.abs(signal_fft)
    # Nondimensionalize the amplitude by dividing by its maximum value
    normalized_amplitude = amplitude / np.max(amplitude)
    # Plot only the positive frequencies (symmetric in FFT)
    positive_freqs = freq[:len(freq)//2]
    positive_normalized_amplitude = normalized_amplitude[:len(amplitude)//2]
    # Nondimensionalize the frequency
    nondimensional_freqs = positive_freqs / f_n  # Nondimensionalizing frequency
    # Plot the FFT amplitude spectrum
    ax.plot(nondimensional_freqs, positive_normalized_amplitude, label=label, color=color, linestyle=linestyle)
    ax.set_xlabel(r'Frequency ratio (f/f$_n$)', fontsize=fontsize)
    ax.set_ylabel('Normalized PSD amplitude', fontsize=fontsize)
    ax.legend()
    # Set the y-axis to logarithmic scale
    ax.set_yscale('log')
        # Automatically set x-axis limits if not provided
    if x_range is None:
        ax.set_xlim([nondimensional_freqs[0], nondimensional_freqs[-1]])
    else:
        ax.set_xlim(x_range)
    # Automatically set y-axis limits
    y_min = np.min(positive_normalized_amplitude[positive_normalized_amplitude > 0])  # Avoid log(0)
    y_max = np.max(positive_normalized_amplitude)
    ax.set_ylim([y_min*10, y_max * 1.2])  # Slightly below min and slightly above max for better visibility

def compute_peak_frequency(signal, sampling_interval):
    signal_fft = fft(signal)
    freq = fftfreq(len(signal), d=sampling_interval)
    positive_freqs = freq[:len(freq)//2]
    positive_amplitudes = np.abs(signal_fft[:len(signal)//2])
    peak_frequency = positive_freqs[np.argmax(positive_amplitudes)]
    return peak_frequency

def plot_fft_spectrum_custom(signal, sampling_interval, label, color, ax, ref_frequency, x_range=None, linestyle='-', fontsize=14):
    
    # Compute FFT of the signal
    signal_fft = fft(signal)
    freq = fftfreq(len(signal), d=sampling_interval)
    amplitude = np.abs(signal_fft)

    # Nondimensionalize the amplitude
    normalized_amplitude = amplitude / np.max(amplitude)

    # Plot only the positive frequencies
    positive_freqs = freq[:len(freq)//2]
    positive_normalized_amplitude = normalized_amplitude[:len(amplitude)//2]

    # Nondimensionalize the frequency using custom reference frequency
    nondimensional_freqs = positive_freqs / ref_frequency

    # Plot the FFT amplitude spectrum
    ax.plot(nondimensional_freqs, positive_normalized_amplitude, label=label, color=color, linestyle=linestyle)
    ax.set_xticks(np.arange(0, 10, 1))
    ax.grid(True, which='major', axis='x', linestyle='--', color='gray', linewidth=0.5)
    ax.legend()

    # Set the y-axis to logarithmic scale
    ax.set_yscale('log')

    # Automatically set x-axis limits if not provided
    if x_range is None:
        ax.set_xlim([nondimensional_freqs[0], nondimensional_freqs[-1]])
    else:
        ax.set_xlim(x_range)

    # Automatically set y-axis limits
    y_min = np.min(positive_normalized_amplitude[positive_normalized_amplitude > 0])
    y_max = np.max(positive_normalized_amplitude)
    ax.set_ylim([y_min * 10, y_max * 5])


def smooth_signal(signal, method="savgol", **kwargs):
    signal = np.asarray(signal)

    if method == "savgol":
        # Savitzky-Golay filter
        window_length = kwargs.get("window_length", 51)
        polyorder = kwargs.get("polyorder", 3)
        mode = kwargs.get("mode", "interp")  # interp handles edges better
        if window_length % 2 == 0 or window_length <= 0:
            raise ValueError("window_length must be a positive odd integer.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")
        return savgol_filter(signal, window_length, polyorder, mode=mode)

    elif method == "gaussian":
        # Gaussian filter
        sigma = kwargs.get("sigma", 2)
        mode = kwargs.get("mode", "nearest")  # safe edge handling
        return gaussian_filter1d(signal, sigma=sigma, mode=mode)

    elif method == "moving_average":
        # Moving average
        window_size = kwargs.get("window_size", 5)
        mode = kwargs.get("mode", "same")  # keep same length
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode=mode)

    elif method == "exponential":
        # Exponential smoothing
        alpha = kwargs.get("alpha", 0.3)
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in the range (0, 1].")
        smoothed = np.zeros_like(signal, dtype=float)
        smoothed[0] = signal[0]
        for t in range(1, len(signal)):
            smoothed[t] = alpha * signal[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed

    else:
        raise ValueError(f"Invalid method '{method}'. "
                         "Choose from 'savgol', 'gaussian', 'moving_average', or 'exponential'.")



def compute_vibration_amplitude(displacement):
    """
    Compute the vibration amplitude based on the RMS value of the displacement.
    Amplitude A is calculated as A = Yrms × sqrt(2)
    
    Parameters:
    displacement (array-like): Displacement time history (1D array)
    
    Returns:
    float: Vibration amplitude
    """
    displacement = np.asarray(displacement)
    
    # Compute RMS value of displacement
    yrms = np.sqrt(np.mean(displacement**2))
    
    # Compute amplitude
    amplitude = yrms * np.sqrt(2)
    
    return amplitude

def compute_instantaneous_power(y_displacement, mass, damping_ratio, natural_frequency, sampling_interval):
    # Calculate the term 4 * π * m * damping_ratio * fn
    constant_term = 4 * np.pi * mass * damping_ratio * (natural_frequency)
    
    # Calculate power at each time step based on displacement differences
    displacement_diffs = np.diff(y_displacement)
    squared_diffs = (displacement_diffs / sampling_interval) ** 2
    instantaneous_power = constant_term * squared_diffs /0.4985  # in mw
    
    # Pad the result with one zero at the beginning to match the length of the original time array
    instantaneous_power = np.insert(instantaneous_power, 0, 0)
    # Calculate the average power over the total time (if needed)
    average_power = np.mean(instantaneous_power) # Average power
    
    return instantaneous_power, average_power


def compute_stouhal(cl_data, sampling_interval, L):
    lift_fft = fft(cl_data.to_numpy())
    freq = fftfreq(len(cl_data), d=sampling_interval)
    dominant_frequency = freq[np.argmax(np.abs(lift_fft))]
    velocity = U_MAX
    characteristic_length = L
    stouhal_number = dominant_frequency * characteristic_length / velocity
    return stouhal_number
    
def plot_data(time, data, average, title, color, avg_label, subplot):
    subplot.plot(time, data, label=f'{title}', color=color)
    subplot.axhline(y=average, color=color, linestyle='--', label=avg_label)
    subplot.set_xlabel('T*')
    subplot.set_ylabel(f'{title}')
    legend = subplot.legend(
    fontsize=14,
    loc='upper right',  # Top-right corner
    frameon=True        # Frame around the legend
    )
    legend.get_frame().set_alpha(0.7)  # Set transparency to 70%
    legend.get_frame().set_edgecolor('gray')  # Optional: Add border color

def compute_detrended_phase_difference_hilbert(signal_1, signal_2, smoothing_window=71, polyorder=7):
    # Compute the analytic signal using the Hilbert transform
    analytic_signal_1 = hilbert(signal_1)
    analytic_signal_2 = hilbert(signal_2)
    
    instantaneous_phase_1 = np.unwrap(np.angle(analytic_signal_1))
    instantaneous_phase_2 = np.unwrap(np.angle(analytic_signal_2))
    
    instantaneous_phase_difference = instantaneous_phase_1 - instantaneous_phase_2

    instantaneous_phase_difference_deg = np.degrees(instantaneous_phase_difference)
    mean_phase_difference_deg = np.mean(instantaneous_phase_difference_deg)
    
    if mean_phase_difference_deg <= 0:
        instantaneous_phase_difference_deg = -instantaneous_phase_difference_deg 
    # Ensure phase difference is wrapped to [0, 360)
    instantaneous_phase_difference_deg = (instantaneous_phase_difference_deg) % 360

    phase_lag = np.array(instantaneous_phase_difference_deg)  # Assuming `instantaneous_phase_difference_deg` is a NumPy array

    mask_0_180 = (phase_lag >= 0) & (phase_lag < 180)
    mask_180_360 = (phase_lag >= 180) & (phase_lag < 360)

    mean_phase_lag_0_180 = np.mean(phase_lag[mask_0_180]) if np.any(mask_0_180) else 0
    mean_phase_lag_180_360 = np.mean(phase_lag[mask_180_360]) if np.any(mask_180_360) else 0

    if abs(mean_phase_lag_180_360 - 360) < abs(mean_phase_lag_180_360 - 180) and abs(mean_phase_lag_0_180 - 0) < abs(mean_phase_lag_0_180 - 180):
        phase_lag[mask_180_360] -= 360

    instantaneous_phase_difference_deg = phase_lag

    mean_phase_difference_deg = np.mean(instantaneous_phase_difference_deg)
    if mean_phase_difference_deg >= 180:
        instantaneous_phase_difference_deg = 360-instantaneous_phase_difference_deg   

    if isinstance(smoothing_window, int) and smoothing_window > 1:
        if len(instantaneous_phase_difference_deg) > smoothing_window:
            if smoothing_window % 2 == 0:
                smoothing_window += 1  # Ensure the window length is odd for the Savitzky-Golay filter
            instantaneous_phase_difference_deg = savgol_filter(
                instantaneous_phase_difference_deg, smoothing_window, polyorder
            )

    mean_phase_difference_deg = np.mean(instantaneous_phase_difference_deg)
    if mean_phase_difference_deg <= 0:
        instantaneous_phase_difference_deg = -instantaneous_phase_difference_deg 

    mean_phase_difference_deg = np.mean(instantaneous_phase_difference_deg)

    return instantaneous_phase_difference_deg, mean_phase_difference_deg

params = read_and_evaluate_parameters('input2d')
# Parameters
UR = params.get('UR', None)
mr1 = params.get('M_RATIO_1', None)
mr2 = params.get('M_RATIO_2', None)
rho = params.get('RHO', None)
pi = params.get('PI', None)
# First try specific keys
specific_keys = ['Damping_ratio_1', 'Damping_ratio_2', 'Damping_ratio_3']

# collect whichever specific ratios are defined
specific_vals = [params[k] for k in specific_keys if k in params]

if not specific_vals:
    # no specific key => fall back to generic
    if 'damping' not in params:
        raise ValueError("No damping information found.")
    damping_ratio = params['damping']
    damping_ratio1 = damping_ratio2 = damping_ratio3 = damping_ratio
else:
    # at least one specific key; ensure they’re all equal
    first = specific_vals[0]
    if any(val != first for val in specific_vals):
        raise ValueError("Inconsistent specific damping ratios in input.")
    damping_ratio = first
    # and assign out 1,2,3 (even if some weren’t in the file)
    damping_ratio1 = damping_ratio2 = damping_ratio3 = damping_ratio

damping_ratio = damping_ratio1
natural_frequency = params.get('Fn_1', None)
total_time = params.get('sampling_duration', None)
SP_U = params.get('SPU', None)
SP_D = params.get('SPD', None)
ref_length_1 = params.get('D_1', None) * params.get('M_1', None)
ref_length_2 = params.get('D_2', None) * params.get('M_2', None)
ref_length_3 = params.get('D_3', None) * params.get('M_3', None)
ref_length_3_ST = params.get('D_3', None)
U_MAX = params.get('U_MAX', None)
mass1 = mr1 * rho * 0.25 * pi * (ref_length_1 ** 2)
mass2 = mr2 * rho * 0.25 * pi * (ref_length_2 ** 2)
RE = params.get('Re', None)
fn_1 = params.get('Fn_1', None)
fn_2 = params.get('Fn_2', None)
end_time = params.get('TIMEending', None)
sampling_duration = params.get('sampling_duration', None)
start_time = end_time - sampling_duration
round_num = params.get('round_num', None)


cd_1 = pd.read_csv('CTD_1.curve', header=None, sep='\s+')

cd_1 = filter_last_n_seconds(cd_1, end_time)

cd_1[1] = smooth_signal(cd_1[1], method="gaussian", sigma=70, mode="nearest")

cd_ave_1 = round(cd_1[1].mean(), round_num)

cl_1 = pd.read_csv('CTL_1.curve', header=None, sep='\s+')


cl_1 = filter_last_n_seconds(cl_1, end_time)

cl_1[1] = smooth_signal(cl_1[1], method="gaussian", sigma=70, mode="nearest")



time_1 = cl_1[0].values
cl_coeff_1 = cl_1[1].values


# Sort by time to ensure correct gradient calculations
sorted_indices_1 = np.argsort(time_1)
time_1 = time_1[sorted_indices_1]
cl_coeff_1 = cl_coeff_1[sorted_indices_1]




# Calculate time derivatives of lift coefficients
d_cl_dt_1 = np.gradient(cl_coeff_1, time_1)  # Time derivative for cl_1


cl_rms_1 = round(np.sqrt(np.mean(np.square(cl_1[1]))), round_num)

sampling_interval = cl_1.iloc[1, 0] - cl_1.iloc[0, 0]

stouhal_number_1 = round(compute_stouhal(cl_1[1], sampling_interval, ref_length_1), round_num)



disp_1 = pd.read_csv('CDisplacement_1.curve', header=None, sep='\s+')

disp_1 = filter_last_n_seconds(disp_1, end_time)

time = disp_1[0].to_numpy()

# Calculate phase differences with detrending
detrended_phase_difference_CLY1t, detrended_mean_phase_difference_CLY1 = compute_detrended_phase_difference_hilbert(cl_1[1].to_numpy(), disp_1[2].to_numpy(), time)

# Calculate the absolute mean phase differences and round them
abs_phase_difference_CLY1 = round(abs(detrended_mean_phase_difference_CLY1), round_num - 2)

print(f"Phase difference between CL1 and Y1: {abs_phase_difference_CLY1} degrees")

y_displacement_1 = disp_1[2].to_numpy()

y_dot_1 = disp_1[4].to_numpy()

Y_star_1 = round(compute_vibration_amplitude(y_displacement_1)/ref_length_1, round_num)

fosc_1 = round(compute_oscillation_frequency(y_displacement_1, sampling_interval), round_num)

f_ratio_1 = round(fosc_1 / fn_1, round_num)
 
L = 1 
def convert_lift_coefficient_to_force(lift_coefficient, rho, U, D, L=1):

    # Calculate lift force using the formula f_L = C_L * 0.5 * rho * U^2 * D * L
    lift_force = lift_coefficient * 0.5 * rho * U**2 * D * L
    return lift_force


lift_force_1 = convert_lift_coefficient_to_force(cl_1[1].to_numpy(), rho, U_MAX, ref_length_1, L)
drag_force_1 = convert_lift_coefficient_to_force(cd_1[1].to_numpy(), rho, U_MAX, ref_length_1, L)






#lift decompose to sub harmonics
def get_dominant_harmonic(time, signal):
    t = np.asarray(time)
    x = np.asarray(signal)
    if t.shape != x.shape:
        raise ValueError("time and signal must have the same shape")
    
    N = x.size
    dt = t[1] - t[0]
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, dt)
    
    mags = np.abs(X)
    mags[0] = 0
    idx = mags.argmax()
    
    f_dom   = freqs[idx]
    A_dom   = 2 * mags[idx] / N
    phi_dom = np.angle(X[idx], deg=True)
    return f_dom, A_dom, phi_dom

def decompose_harmonics(time, signal, num_harmonics=3):
    """
    Returns a list of (f, A, phi_rad, component_time_series)
    for the top `num_harmonics` FFT lines.
    """
    t = np.asarray(time)
    x = np.asarray(signal)
    N = x.size
    dt = t[1] - t[0]
    
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, dt)
    
    mags = np.abs(X)
    mags[0] = 0
    # pick the indices of the largest `num_harmonics` peaks
    idxs = np.argsort(mags)[-num_harmonics:][::-1]
    
    harmonics = []
    for idx in idxs:
        f_n   = freqs[idx]
        A_n   = 2 * mags[idx] / N
        phi_n = np.angle(X[idx])   # in radians
        # reconstruct that single harmonic in time domain
        comp = A_n * np.sin(2*np.pi * f_n * t + phi_n)
        harmonics.append((f_n, A_n, phi_n, comp))
    return harmonics


# 1) compute
f_disp1, A_disp1, phi_disp1 = get_dominant_harmonic(
    disp_1[0].to_numpy(),
    disp_1[2].to_numpy()
)
f_lift1, A_lift1, phi_lift1 = get_dominant_harmonic(
    cl_1[0].to_numpy(),
    cl_1[1].to_numpy()
)



# 5) decompose into first three harmonics
harm_disp1 = decompose_harmonics(disp_1[0].to_numpy(), disp_1[2].to_numpy(), num_harmonics=1)
harm_lift1 = decompose_harmonics(cl_1[0].to_numpy(), cl_1[1].to_numpy(), num_harmonics=1)



# 2) shift reference so φ_disp = 0°
phi_disp_ref1 = phi_disp1- phi_disp1
phi_lift_ref1 = abs((phi_lift1 - phi_disp1 + 180) % 360 - 180)

# 3) optionally check you’re on the same harmonic
if abs(f_disp1 - f_lift1) > 1e-6:
    print(f"Warning: dominant freqs differ: disp {f_disp1:.4f} Hz vs lift {f_lift1:.4f} Hz")

# 4) print everything
print(f"Disp → f = {f_disp1:.4f} Hz, A = {A_disp1:.3e}, φ = {phi_disp_ref1:.1f}°")
print(f"Lift → f = {f_lift1:.4f} Hz, A = {A_lift1:.3e}, φ = {phi_lift_ref1:.1f}°")
print(f"Phase lag UC (lift – disp) = {phi_lift_ref1:.1f}°")



# Grab the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(10, 5))
for i, ((f_d, _, _, comp_d), (f_l, _, _, comp_l)) in enumerate(zip(harm_disp1, harm_lift1)):
    color = colors[i % len(colors)]
    # Displacement: solid
    plt.plot(disp_1[0].to_numpy(), comp_d,
             linestyle='-',
             color=color,
             label=f'Harmonic {i+1} Disp ({f_d:.2f} Hz)')
    # Lift: dashed
    plt.plot(disp_1[0].to_numpy(), comp_l,
             linestyle='--',
             color=color,
             label=f'Harmonic {i+1} Lift ({f_l:.2f} Hz)')

    
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("First Three Harmonics: Displacement (solid) vs. Lift (dashed) UC")
plt.legend(ncol=2, fontsize='small')
plt.xlim([end_time-0.75*sampling_duration,end_time])
plt.tight_layout()
plt.savefig('harmonics_overlayUC.png')





def extract_harmonic_component_at_f(
    signal: np.ndarray,
    sampling_interval: float,
    f_target: float,
    apply_window: bool = False
) -> tuple[np.ndarray, float, float]:
    """
    Extract the single‐frequency component of `signal(t)` at f_target via FFT,
    with optional windowing and proper de‐meaning/detrending.

    Returns
    -------
    component_time_series : np.ndarray
        Reconstructed A_target * sin(2π f_target t + phase_target).
    A_target : float
        Amplitude of that component.
    phase_target : float
        Phase (in radians), wrapped into [−π, +π].
    """
    # 0) De‐mean & detrend
    y = signal - np.mean(signal)
    y = detrend(y)

    N = y.size

    # 1) Optional Hanning window to reduce leakage
    if apply_window:
        w = np.hanning(N)
        y *= w
        # amplitude compensation for Hanning window
        window_comp = 1.0 / np.mean(w)
    else:
        window_comp = 1.0

    # 2) FFT
    X = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=sampling_interval)

    # 3) Pick nearest bin to f_target
    k = np.argmin(np.abs(freqs - f_target))
    Xk = X[k]

    # 4) Compute amplitude and phase
    A_target = 2.0 * np.abs(Xk) * window_comp / N
    phase_target = np.angle(Xk)
    # wrap to [−π, +π]
    phase_target = (phase_target + np.pi) % (2*np.pi) - np.pi

    # 5) Reconstruct the pure sinusoid
    t = np.arange(N) * sampling_interval
    component_time_series = A_target * np.sin(2.0*np.pi*f_target*t + phase_target)

    return component_time_series, A_target, phase_target


# --- Example usage for UC and DC ---

# Convert lift coefficients to force arrays beforehand:
lift_force_1 = convert_lift_coefficient_to_force(
    cl_1[1].to_numpy(), rho, U_MAX, ref_length_1, L
)


# Extract fundamental components
lift_comp_uc, A_lift_uc, φ_lift_uc = extract_harmonic_component_at_f(
    signal=lift_force_1,
    sampling_interval=sampling_interval,
    f_target=fosc_1,
    apply_window=True
)
disp_comp_uc, A_disp_uc, φ_disp_uc = extract_harmonic_component_at_f(
    signal=y_displacement_1,
    sampling_interval=sampling_interval,
    f_target=fosc_1,
    apply_window=True
)

# Compute signed phase‐lag in radians, wrapped to [−π, +π]
phase_lag_uc_rad = φ_lift_uc - φ_disp_uc
phase_lag_uc_rad = (phase_lag_uc_rad + np.pi) % (2 * np.pi) - np.pi

# Convert to degrees (range [−180°, +180°])
phase_lag_uc_deg = np.degrees(phase_lag_uc_rad)

print(f"UC:  A_lift = {A_lift_uc:.3e}, A_disp = {A_disp_uc:.3e}")
print(f"UC: phase_lift = {np.degrees(φ_lift_uc):.1f}°, phase_disp = {np.degrees(φ_disp_uc):.1f}°")
print(f"UC: Phase‐lag (lift – disp) = {phase_lag_uc_deg:.1f}°")





# (1) Build a time‐vector that matches the length of those “component” arrays
N_uc = lift_comp_uc.size  # should equal disp_comp_uc.size
t_uc = np.arange(N_uc) * sampling_interval

# (2) Plot Upstream (UC) “pure‐harmonic” lift & disp on the same axes
plt.figure(figsize=(8, 4))
plt.plot(t_uc, disp_comp_uc, label='UC: displacement @ f$_{osc}$', color='blue', linewidth=1.5)
plt.plot(t_uc, lift_comp_uc, label='UC: lift @ f$_{osc}$',       color='red',  linestyle='--', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Upstream Cylinder: Harmonic Components at f$_{osc}$')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('UC_fosc_components.png', dpi=300)
plt.show()



def compute_work_per_cycle(F_amp: float, Y_amp: float, phase_rad: float) -> float:
    """
    Compute W_f = \int_{t}^{t+T} F dY for a pure sinusoid at the same frequency.

    Parameters
    ----------
    F_amp : float
        Amplitude of the lift force harmonic, F_{L0}.
    Y_amp : float
        Amplitude of the displacement harmonic, Y_{0}.
    phase_rad : float
        Phase lag φ between F(t) = F_{L0} sin(ω t + φ) and 
        Y(t) = Y_{0} sin(ω t), in radians.

    Returns
    -------
    Wf : float
        The work done by F over one period: W_f = π * F_amp * Y_amp * sin(φ).
    """
    return np.pi * F_amp * Y_amp * np.sin(phase_rad)

#   3) Compute the signed phase‐lag φ between lift and disp (in radians)
#    If you prefer to wrap to [−π, π):
sqrt2 = math.sqrt(2)
Fl_1rms = np.sqrt(np.mean(lift_force_1**2))
F01= sqrt2 * Fl_1rms

# Now compute W_f (Upstream cylinder):
W_f_uc = compute_work_per_cycle(F01, Y_star_1, phase_lag_uc_rad)
print(f"UC: W_f = {W_f_uc:.4e} (same units as F·Y).")



# ─── User data (replace these with your actual arrays) ───────
Uy1 = disp_1[4].to_numpy()
Fx1 = drag_force_1
Fy1 = lift_force_1




def compute_added_damping_ratio(
    force:    np.ndarray,  # F(t)
    t_force:  np.ndarray,  # timestamps for F
    displacement: np.ndarray,  # Y(t)
    t_disp:   np.ndarray,  # timestamps for Y
    velocity: np.ndarray,  # Ẏ(t)
    t_vel:    np.ndarray,  # timestamps for Ẏ
    mass:     float,       # m
    fn:       float,       # natural freq (Hz)
    fosc:     float,       # osc freq (Hz)
    Y0:       float,       # peak amplitude
    φ1:       float, 
    φ2:       float, 
    name:     str = "UC"
) -> float:
    omega_n = 2*np.pi*fosc

    # 1) find NEG→POS zero‐crossings in **displacement** Y(t)
    sign_y   = np.sign(displacement)
    zero_idx = np.where((sign_y[:-1] < 0) & (sign_y[1:] >= 0))[0]
    if len(zero_idx) < 2:
        raise ValueError(f"{name}: not enough NEG→POS zero‐crossings in displacement")

    # interpolate exact crossing times
    t_cross = (t_disp[zero_idx] + t_disp[zero_idx+1]) / 2
    t0, t1  = t_cross[0], t_cross[-1]

    # 2) slice F and Ẏ over [t0, t1]
    mask_F    = (t_force >= t0) & (t_force <= t1)
    mask_Ydot = (t_vel   >= t0) & (t_vel   <= t1)

    tF    = t_force[mask_F]
    Fseg  = force[mask_F]
    Ydseg = np.interp(tF, t_vel[mask_Ydot], velocity[mask_Ydot])

    # 3) remove any DC
    Fseg  -= Fseg.mean()
    Ydseg -= Ydseg.mean()

    # 4) plot instantaneous power
    plt.figure(figsize=(8,4))
    plt.plot(tF, Fseg * Ydseg, label=f'Power {name}')
    plt.axhline(0, color='k', ls='--')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$F\,\dot Y$')
    plt.title(f'Instantaneous Power Input ({name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'instantaneous_power_{name}.png', dpi=300)
    plt.close()

    # 5) integrate & normalize
    integral = simpson(Fseg * Ydseg, x=tF)
    window   = t1 - t0
    denom    = window * mass * 2 * np.pi * fn * (omega_n * Y0)**2
    
    FL_rms = np.sqrt(np.mean(force**2))
    FL0    = np.sqrt(2) * FL_rms
    ca1     = - FL0 * np.sin(φ1) / (omega_n * Y0)
    ca2     = - FL0 * np.sin(φ2) / (omega_n * Y0)
    cc       = 4*np.pi*mass*fn
    zeta_harmonic1 = ca1 / cc
    zeta_harmonic2 = ca2 / cc
    addedmass1 =  FL0 * np.cos(φ1) / (omega_n**2) / Y0 / (0.25*np.pi*0.997)
    addedmass2 =  FL0 * np.cos(φ2) / (omega_n**2) / Y0 / (0.25*np.pi*0.997)
    return - integral / denom , zeta_harmonic1, addedmass1, zeta_harmonic2, addedmass2


zeta_a_uc1, zeta_a_uc2, m_star_uc, zeta_a_uc22, m_star_uc1 = compute_added_damping_ratio(
    force        = lift_force_1,
    t_force      = cl_1[0].to_numpy(),
    displacement = disp_1[2].to_numpy(),  # Y(t)
    t_disp       = disp_1[0].to_numpy(),
    velocity     = disp_1[4].to_numpy(),  # Ẏ(t)
    t_vel        = disp_1[0].to_numpy(),
    mass         = mass1,
    fn           = fn_1,
    fosc         = fosc_1,
    Y0           = Y_star_1,
    φ1            = phase_lag_uc_rad,
    φ2            = np.radians(abs_phase_difference_CLY1),
    name         = "UC"
)



zeta_total_uc1 = damping_ratio1 + zeta_a_uc1

zeta_total_uc2 = damping_ratio1 + zeta_a_uc2
zeta_total_uc3 = damping_ratio1 + zeta_a_uc22
print(f"UC total ζ1 = {zeta_total_uc1:.4f} →", 
      "galloping" if zeta_total_uc1 < 0 else "no galloping")


print(f"UC total ζ2 = {zeta_total_uc2:.4f} →", 
      "galloping" if zeta_total_uc2 < 0 else "no galloping")


print(f"UC total ζ3 = {zeta_total_uc3:.4f} →", 
      "galloping" if zeta_total_uc3 < 0 else "no galloping")


# ─── Write summary into CSV ───────────────────────────────────

galloping_summary_filename = "galloping_indication_summary.csv"
header = [
    'SPU','SPD','UR','damping',
    'UC_gallop_slope','DC_gallop_slope',
    'UC_gallop_damping1','DC_gallop_damping1',
    'UC_gallop_damping2','DC_gallop_damping2',
    'UC_gallop_damping3','DC_gallop_damping3'
]
row = {
    'SPU':               SP_U,
    'SPD':               SP_D,
    'UR':                UR,
    'damping':           damping_ratio,
    # now use the boolean check (1 if galloping, 0 otherwise)
    'UC_gallop_damping1': int(zeta_total_uc1 < 0),
    'UC_gallop_damping2': int(zeta_total_uc2 < 0),
    'UC_gallop_damping3': int(zeta_total_uc3 < 0),
}

# 1) Read existing rows into a list of dicts, skipping bad ones
rows = []
if os.path.exists(galloping_summary_filename):
    with open(galloping_summary_filename, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # only keep rows with exactly the expected keys
            if set(r.keys()) == set(header):
                rows.append(r)

# 2) Look for a match on the key columns and update or append
match_idx = next(
    (i for i,r in enumerate(rows)
     if (r['SPU']==str(row['SPU']) and
         r['SPD']==str(row['SPD']) and
         r['UR']==str(row['UR']) and
         r['damping']==str(row['damping']))),
    None
)
if match_idx is not None:
    # update existing
    rows[match_idx] = {k:str(v) for k,v in row.items()}
else:
    # append new
    rows.append({k:str(v) for k,v in row.items()})

# 3) Write them all back with a clean header
with open(galloping_summary_filename, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"Galloping summary updated in {galloping_summary_filename}")



def compute_added_mass_ratio(
    force:        np.ndarray,  # F(t)
    t_force:      np.ndarray,  # timestamps for F
    displacement: np.ndarray,  # Y(t)
    t_disp:       np.ndarray,  # timestamps for Y
    velocity:     np.ndarray,  # Ẏ(t)
    t_vel:        np.ndarray,  # timestamps for Ẏ
    rho:          float,       # fluid density
    D:            float,       # diameter
    l:            float,       # span
    fn:           float,       # natural freq (Hz)
    Y0:           float,       # peak amplitude
    name:         str = "UC"   # for titles & filenames
) -> float:
    """
    m_a* = –8 / [nT ρ π D² l (ω² Y0)²] ∫ F(t)·Ẍ(t) dt
    where ω = 2πfn, and the integration window runs
    from the first to the last NEG→POS crossing of Y(t).
    """
    ω   = 2*np.pi*fn

    # 1) find NEG→POS zero crossings in Y(t)
    sign_y   = np.sign(displacement)
    zero_idx = np.where((sign_y[:-1] < 0) & (sign_y[1:] >= 0))[0]
    if len(zero_idx) < 2:
        raise ValueError(f"{name}: need ≥2 NEG→POS zero‐crossings in displacement")

    # 2) approximate their times by linear interp
    t_cross = (t_disp[zero_idx] + t_disp[zero_idx+1]) / 2
    t0, t1  = t_cross[0], t_cross[-1]

    # 3) slice F & compute accel = d(Ẏ)/dt
    mask_F    = (t_force >= t0) & (t_force <= t1)
    mask_Ydot = (t_vel   >= t0) & (t_vel   <= t1)

    tF    = t_force[mask_F]
    Fseg  = force[mask_F]
    Ydot_seg = velocity[mask_Ydot]
    tYdot    = t_vel[mask_Ydot]

    # compute acceleration
    dt = tYdot[1] - tYdot[0]
    Accel = np.gradient(Ydot_seg, dt)
    # interpolate accel onto tF
    a_seg = np.interp(tF, tYdot, Accel)

    # 4) remove any DC bias
    Fseg -= Fseg.mean()
    a_seg -= a_seg.mean()

    # 5) plot instantaneous F·a to verify sign
    plt.figure(figsize=(8,4))
    plt.plot(tF, Fseg * a_seg, label=f'F·Ẍ ({name})')
    plt.axhline(0, color='k', ls='--')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$F(t)\,\ddot Y(t)$')
    plt.title(f'Instantaneous F·ä ({name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'instantaneous_Fa_{name}.png', dpi=300)
    plt.close()

    # 6) integrate and nondimensionalize
    integral = simpson(Fseg * a_seg, x=tF)
    window   = t1 - t0
    denom    = window * rho * np.pi * D**2 * l * (ω**2 * Y0)**2

    # final non-dim added-mass ratio
    m_a_star = -8 * integral / denom
    return m_a_star



# Upstream cylinder
m_star_uco = compute_added_mass_ratio(
    force        = lift_force_1,
    t_force      = cl_1[0].to_numpy(),
    displacement = disp_1[2].to_numpy(),
    t_disp       = disp_1[0].to_numpy(),
    velocity     = disp_1[4].to_numpy(),
    t_vel        = disp_1[0].to_numpy(),
    rho          = rho,
    D            = ref_length_1,
    l            = L,
    fn           = fosc_1,
    Y0           = Y_star_1,
    name         = "UC"
)

# Downstream cylinder



print(f"Added‐mass ratio UC(o): {m_star_uco:.4f}")




def compute_lift_velocity_acceleration(C_l, v, time, T, U_max, D):
    lift_coe= np.asarray(C_l)
    vel = np.asarray(v)
    time_array = np.asarray(time)
    dt = time_array [1] - time_array [0]
    sampling_dur = T
    U = U_max
    acce = np.gradient(v, dt)
    Cl_v1 = np.trapezoid(lift_coe * vel, time)/sampling_dur
    Cl_v2 = np.sqrt ((2/sampling_dur) * np.trapezoid(vel**2, time))
    Cl_vt = (lift_coe * vel/Cl_v2)
    Cl_v =  Cl_v1/Cl_v2

    Cm_a1 = np.trapezoid(lift_coe * acce, time)
    Cm_a2 = np.trapezoid(acce**2, time)
    Cm_a = (-2*U**2)/(np.pi*D**2)*(Cm_a1/Cm_a2)
    return Cl_vt, Cl_v, Cm_a




def compute_Ca(Cy, zeta, time_array, eps=1e-12):
    """
    Compute added‐mass coefficient Cm given:
      Cy    : array of cross‐flow force coefficients C_y
      zeta  : array of displacement values ζ
      dzeta : array of velocity gradients dζ/dt
    Returns:
      Cm    : array of Cm values at each time step
    """
    Cy    = np.asarray(Cy)
    zeta  = np.asarray(zeta)
    dt = time_array [1] - time_array [0]
    dzeta = np.gradient(zeta, dt)
    # denominator: avoid exact zeros
    denom = np.mean(dzeta**2)
    denom = np.where(denom < eps, np.nan, denom)
    nom = np.mean(dzeta*Cy)
    Ca = -2/np.pi* nom / denom
    return Ca




Ekut, Eku, C_a_1 = compute_lift_velocity_acceleration (cl_1[1].to_numpy(), disp_1[4].to_numpy(), time, sampling_duration, U_MAX, ref_length_1)



C_a_1111 = compute_Ca(cl_coeff_1, disp_1[4].to_numpy(), time, eps=1e-12)




def compute_ce(C_l, v, time, T):
    lift_coe= np.asarray(C_l)
    vel = np.asarray(v)
    time_array = np.asarray(time)
    Ce = np.trapezoid(lift_coe * vel, time_array)/T
    return Ce



Ek11 = compute_ce (cl_1[1].to_numpy(), disp_1[4].to_numpy(), time, sampling_duration)

print(f"Time-mean energy transfer coefficient for UC E_mean:  {Eku}" )


print(f"Time-mean energy transfer coefficient for UC E_mean11:  {Ek11}" )

print(f"Added mass coefficient for UC C_a1: {C_a_1}")


print(f"Added mass coefficient for UC C_a_1111: {C_a_1111}")


# Compute time-varying P_Mech for each cylinder
P_mech_time_1, average_power_1 = compute_instantaneous_power(y_displacement_1, mass1, damping_ratio1, fn_1, sampling_interval)
# Generate time array for plotting
time_array = np.linspace(0, total_time, len(P_mech_time_1))

P_fluid_1 = 0.5*rho*U_MAX*U_MAX*U_MAX*(ref_length_1+Y_star_1*2)/0.4985

Efficiency_1 = average_power_1 /P_fluid_1/(16/27)* 100


print("Efficiency for Cylinder 1 in %:", Efficiency_1)


# Compute total time-varying P_Mech
P_mech_total_time = P_mech_time_1
P_mech_total_time_ave = average_power_1 

Y_max = Y_star_1
P_fluid_max = P_fluid_1
Efficiency_total = P_mech_total_time_ave / P_fluid_max * 100

print("Total Efficiency in %:", Efficiency_total)

print("Time varying Pmeach for Cylinder 1 in mw:", average_power_1)
print("Time varying Pmeach in total in mw:", P_mech_total_time_ave) 

print("Non-dimensional vibration amplitude Y* for Cylinder 1:", Y_star_1)
print("Oscillation frequency fosc and f* / fn for Cylinder 1:", fosc_1, f_ratio_1)
print("Stouhal number for cylinder 1:", stouhal_number_1)
print("Root mean square lift coefficient for cylinder 1:", cl_rms_1)

print("Time-averaged drag coefficient for cylinder 1:", cd_ave_1)


TEST1 = 8*pi*pi*pi*mass1*damping_ratio1*((Y_star_1*fosc_1)**2)*fn_1/0.4985

print("TEST Cylinder 1 in mw:", TEST1)

# after computing zeta_total_uc1, zeta_total_dc1, zeta_total_uc2, zeta_total_dc2

uc_gallop_damping1  = int(zeta_total_uc1  < 0)
uc_gallop_damping2  = int(zeta_total_uc2  < 0)

votes_uc =  uc_gallop_damping1 + uc_gallop_damping2

uc_gallop_final = int(votes_uc >= 2)

UC_uncertain    = int(votes_uc == 1)

data = [
    SP_U, SP_D, UR, damping_ratio,
    cd_ave_1, cl_rms_1, stouhal_number_1,
    Y_star_1, fosc_1, round(fn_1, round_num), f_ratio_1,
    abs_phase_difference_CLY1,
    round(average_power_1, round_num), round(P_mech_total_time_ave, round_num),
    round(Efficiency_1, round_num), round(Efficiency_total, round_num),
    round(Eku, round_num + 6), 
    round(Ek11, round_num + 6), 
    round(W_f_uc, round_num + 6), 
    round(C_a_1111, round_num + 3),
    round(m_star_uc, round_num + 3),
    round(zeta_a_uc2, round_num + 3)
]

filename = f"computation_results3_{SP_U}_{SP_D}_{UR}_{damping_ratio}.csv"

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'SP_U', 'SP_D', 'UR', 'damping', 
        'Cd-ave1', 'Cl-rms1', 'St_1', 
        'Y*_1', 'fosc_1', 'fn_1', 'f_ratio_1',
        'phase_difference_CLY1',
        'P_uc', 'P_total',
        'efficiency_uc',  'efficiency_total', 
        'Emean_uc1', 
        'Emean_uc2', 
        'Emean_uc3', 
        'Ca_uc1', 
        'Ca_uc2', 
        'adddamping1'
    ])
    writer.writerow(data)

print(f"Results exported successfully to {filename}.")

fig1, axs = plt.subplots(3, 2, figsize=(20, 9))

plot_data(cd_1[0].to_numpy(), cd_1[1].to_numpy(), cd_ave_1, '$C_{{\\mathrm{{D}}}}$', black1, '$C_{\\mathrm{D-ave}}$', axs[0, 0])
plot_data(cl_1[0].to_numpy(), cl_1[1].to_numpy(), cl_rms_1, '$C_{{\\mathrm{{L}}}}$', red1, '$C_{\\mathrm{L-rms}}$', axs[1, 0])
plot_data(disp_1[0].to_numpy(), disp_1[2].to_numpy(), Y_star_1, 'Y', blue1, 'Y*', axs[2, 0])
plt.tight_layout()
filename1 = f"CY_CDCLY_{SP_U}_{SP_D}_{UR}_{damping_ratio}.jpg"
fig1.savefig(filename1, dpi=600, format='jpg')



# Create the plot
fig5, ax = plt.subplots(figsize=(6, 5))
# Plot FFT spectrum for lift
peak_frequency_uc = compute_peak_frequency(cl_1[1].to_numpy(), sampling_interval)
plot_fft_spectrum_custom(cl_1[1].to_numpy(), sampling_interval, 'UC',  red1, ax, fn_1, x_range=(0, 10))
# Customize the font size of labels, legend, and ticks
ax.set_xlabel('Frequency ratio (f$_{s}$/f$_{n}$)', fontsize=16)
ax.set_ylabel('Normalized PSD Amplitude', fontsize=16)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.minorticks_off()
legend = ax.legend(
    fontsize=14,
    loc='upper right',  # Top-right corner
    frameon=True        # Frame around the legend
)
legend.get_frame().set_alpha(0.7)  # Set transparency to 70%
legend.get_frame().set_edgecolor('gray')  # Optional: Add border color
# Middle subplot: FFT Amplitude of Lift and Displacement
ax.tick_params(axis='both', labelsize=14)  # Adjust tick label font size
plt.tight_layout()
# Show and save the plot
plt.show()
filename5 = f"FFT_{SP_U}_{SP_D}_{UR}_{damping_ratio}.jpg"
fig5.savefig(filename5, dpi=600, format='jpg')


import re
ic_base_dir = '/ibex/user/gongm/projects/c-sp-c/IC'
curve_file_disp = None
curve_file_lift = None

for case_dir in os.listdir(ic_base_dir):
    if not case_dir.startswith("Single_VIV_"):
        continue

    parts = case_dir.split('_')
    ic_case_dir = os.path.join(ic_base_dir, case_dir)

    # Initialize placeholders
    ur_value = damp_value = durr = end_time_ic = None

    # Parse from directory name if possible
    if len(parts) >= 7:
        try:
            ur_value    = float(parts[3])
            damp_value  = float(parts[4])
            durr        = float(parts[5])
            end_time_ic = float(parts[6])
        except ValueError:
            pass  # Fall back to input2d parsing below
    else:
        print(f"Short name, falling back to input2d for: {case_dir}")

    # If missing duration or end_time, read input2d file
    if durr is None or end_time_ic is None:
        input2d_path = os.path.join(ic_case_dir, 'input2d')
        if os.path.isfile(input2d_path):
            with open(input2d_path) as f:
                for line in f:
                    # Parse TIMEending
                    m = re.search(r'TIMEending\s*=\s*([\d\.]+)', line)
                    if m:
                        end_time_ic = float(m.group(1))
                    # Parse sampling_duration; keep the legacy typo as a fallback
                    m2 = re.search(r'samp(?:l|i)ng_duration\s*=\s*([\d\.]+)', line)
                    if m2:
                        durr = float(m2.group(1))
        else:
            print(f"No input2d found in {ic_case_dir}, cannot get duration/end_time.")

    # At this point, we should have ur_value, damp_value, durr, end_time_ic
    # Attempt to parse UR and damping if still missing from directory name
    if ur_value is None or damp_value is None:
        # parts = ["Single","VIV","<Re>","<UR>","<damping>",...]
        try:
            ur_value   = float(parts[3])
            damp_value = float(parts[4])
        except (IndexError, ValueError):
            print(f"Cannot parse UR/damping from {case_dir}, skipping")
            continue

    print(f"Checking folder {case_dir}: UR={ur_value}, damping={damp_value}, dur={durr}, end={end_time_ic}")

    # only proceed if both UR and damping match workdir parameters
    if ur_value == UR and damp_value == damping_ratio1:
        print(f">>> Using IC case: {ic_case_dir}")
        # walk to find the .curve files
        for root, _, files in os.walk(ic_case_dir):
            for fname in files:
                if fname == 'CDisplacement_1.curve':
                    curve_file_disp = os.path.join(root, fname)
                elif fname == 'CTL_1.curve':
                    curve_file_lift = os.path.join(root, fname)
                if curve_file_disp and curve_file_lift:
                    break
            if curve_file_disp and curve_file_lift:
                break
        break  # found matching case

# Verify
if curve_file_disp and curve_file_lift:
    print("Found IC files:")
    print("  Displacement:", curve_file_disp)
    print("  Lift:        ", curve_file_lift)
else:
    print(f"ERROR: IC files not found for UR={UR}, damping={damping_ratio1}")



def filter(data, durrr, end_time):
    if end_time is not None:
        start_time = end_time - durrr
        return data[(data[0] >= start_time) & (data[0] <= end_time)]
    return data


# Conversion factor from centimeters to inches
cm_to_inch = 1 / 2.54
# Create figure with customized subplots using GridSpec
fig8 = plt.figure(figsize=(22 * cm_to_inch * 3, 14 * cm_to_inch * 3 * 0.38))  # Set figure width to 14 cm
gs = gridspec.GridSpec(1, 3, figure=fig8, width_ratios=[2.2, 1.2, 1.1], wspace=0.2)  # Define width ratios and space between subplots
plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.12, wspace=0.1)
font_size = 8 * 2 # Set font size equivalent to Word's size 8
# Left subplot: Displacement and Lift with Time
axs_left = fig8.add_subplot(gs[0])

x_min, x_max = np.min(y_displacement_1), np.max(y_displacement_1)
y_min, y_max = np.min(cl_1[1]), np.max(cl_1[1])

ydata_range = y_max - y_min
xdata_range = x_max - x_min
ranges = np.max([ydata_range, xdata_range])  # Pass as a list or tuple
ltick_step = tick_step = max(round(ranges / 5, 1), 0.1)

ylim_l = np.max([x_max, -1*x_min, y_max, -1*y_min]) * 1.2

xlim_r = np.max([x_max, -1*x_min]) * 1.2
ylim_r = np.max([y_max, -1*y_min]) * 1.2

axs_left.plot(disp_1[0].to_numpy(), disp_1[2].to_numpy(), label='Y', color= blue1)
axs_left.plot(cl_1[0].to_numpy(), cl_1[1].to_numpy(), label='$C_{{\\mathrm{{L}}}}$', color= red1, linestyle='--')
axs_left.yaxis.set_major_locator(MultipleLocator(ltick_step))

axs_left.set_xlabel('T*', fontsize= font_size * 1.4)
axs_left.set_ylabel('Y and $C_{{\\mathrm{{L}}}}$', color='black', fontsize=font_size * 1.4)
axs_left.set_xlim([end_time-0.5*sampling_duration,end_time])
axs_left.set_ylim([-1*ylim_l,ylim_l])
axs_left.xaxis.set_major_locator(MaxNLocator(nbins=5))
axs_left.yaxis.set_major_locator(MaxNLocator(nbins=5))
axs_left.minorticks_off()
axs_left.tick_params(axis='y', labelcolor='black', labelsize=font_size*1.1)
axs_left.tick_params(axis='x', labelcolor='black', labelsize=font_size*1.1)
legend = axs_left.legend(
    fontsize=font_size * 1.1,
    loc='upper right',  # Top-right corner
    frameon=True        # Frame around the legend
)
legend.get_frame().set_alpha(0.7)  # Set transparency to 70%
legend.get_frame().set_edgecolor('gray')  # Optional: Add border color
# Middle subplot: FFT Amplitude of Lift and Displacement
axs_middle = fig8.add_subplot(gs[1])
plot_fft_spectrum(y_displacement_1, sampling_interval, 'Y', blue1, axs_middle, fn_1, x_range=(0, 10), fontsize=font_size * 1.5)
plot_fft_spectrum(cl_1[1].to_numpy(), sampling_interval, '$C_{{\\mathrm{{L}}}}$', red1, axs_middle, fn_1, x_range=(0, 10), linestyle='--', fontsize=font_size * 1.5)

axs_middle.tick_params(axis='x', labelcolor='black', labelsize=font_size*1.1)
axs_middle.tick_params(axis='y', labelcolor='black', labelsize=font_size*1.1)
axs_middle.minorticks_off()
axs_middle.xaxis.set_major_locator(MaxNLocator(nbins=5))
axs_middle.grid(True, which='major', axis='x', linestyle='--', color='gray', linewidth=0.5)
legend = axs_middle.legend(
    fontsize=font_size * 1.1,
    loc='upper right',  # Top-right corner
    frameon=True        # Frame around the legend
)
legend.get_frame().set_alpha(0.7)  # Set transparency to 70%
legend.get_frame().set_edgecolor('gray')  # Optional: Add border color

# Right subplot: Displacement-Velocity Plot
axs_right = fig8.add_subplot(gs[2])
# Compute 5% start and end indices
start_idx = int(0.03 * len(y_displacement_1))
end_idx   = int(0.97 * len(y_displacement_1))

# Slice both arrays consistently
y_mid  = y_displacement_1[start_idx:end_idx]
cl_mid = cl_1[1][start_idx:end_idx]

# Plot only the middle 94%
axs_right.plot(y_mid, cl_mid, color=black1)


ydata_range = y_max - y_min
xdata_range = x_max - x_min
xtick_step = max(round(xdata_range / 5, 1), 0.1)  # Divide range into ~10 steps and round to 1 decimal
ytick_step = max(round(ydata_range / 5, 1), 0.1)  # Divide range into ~10 steps and round to 1 decimal


# Set the major locator dynamically
axs_right.yaxis.set_major_locator(MultipleLocator(ytick_step))
axs_right.xaxis.set_major_locator(MultipleLocator(xtick_step))
axs_right.xaxis.set_major_locator(MaxNLocator(nbins=6))
axs_right.yaxis.set_major_locator(MaxNLocator(nbins=6))
axs_right.minorticks_off()

axs_right.set_xlabel('Displacement Y', color='black', fontsize=font_size * 1.4)
axs_right.set_ylabel('Lift $C_{{\\mathrm{{L}}}}$', color='black', fontsize=font_size * 1.4)
axs_right.tick_params(axis='x', labelcolor='black', labelsize=font_size*1.1)
axs_right.tick_params(axis='y', labelcolor='black', labelsize=font_size*1.1)
axs_right.set_xlim([-1*xlim_r,xlim_r])
axs_right.set_ylim([-1*ylim_r,ylim_r])
# Save the figure
filename8 = f"combined_plot_UC_{SP_U}_{SP_D}_{UR}_{damping_ratio}.jpg"
fig8.savefig(filename8, dpi=600, format='jpg')


