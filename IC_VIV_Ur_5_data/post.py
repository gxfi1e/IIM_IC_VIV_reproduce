import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import re
import math  # Import the math module to use its functions
import csv
from scipy.signal import correlate
from scipy.signal import argrelextrema
def compute_phase_difference(disp_1, disp_2, sampling_frequency):
    # Cross-correlation
    corr = correlate(disp_1.iloc[:, 1], disp_2.iloc[:, 1], mode='full', method='auto')
    lag = np.argmax(corr) - (len(disp_1) - 1)
    
    # Time shift
    time_shift = lag / sampling_frequency
    
    # Frequency calculation (assumes periodic signal)
    time_period = np.mean(np.diff(disp_1[0]))
    frequency = 1 / time_period
    
    # Phase shift in radians
    phase_shift_rad = 1 * np.pi * frequency * time_shift
    
    # Convert radians to degrees
    phase_shift_deg = np.degrees(phase_shift_rad) % 360
    
    return phase_shift_deg

def read_and_evaluate_parameters(filename):
    params = {}
    line_count = 0  # Initialize a counter to keep track of the number of lines read
    with open(filename, 'r') as file:
        for line in file:
            if line_count >= 60:  # Stop reading after 50 lines
                break
            original_line = line.strip()  # Keep a copy of the original line for debugging
            line = line.split('//')[0].strip()  # Remove comments and strip whitespace
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                try:
                    # Attempt to evaluate the expression within a safe limited context
                    value = eval(value.strip(), {"__builtins__": None, "sqrt": math.sqrt, "PI": math.pi}, params)
                    params[key] = value
                except Exception as e:
                    # Print the error and the line that caused it for debugging
                    print(f"Error processing line: {original_line}")
                    print(f"Error message: {e}")
            line_count += 1  # Increment the line counter
    return params
# Use the function to read parameters
params = read_and_evaluate_parameters('input2d')
# Access the UR parameter
UR = params.get('UR', None)  # Using .get() for safe access; returns None if 'UR' is not found
# Access the configuration
SP_U = params.get('resolution', None)  # Using .get() for safe access; returns None if 'UR' is not found
SP_D = params.get('resolution', None)  # Using .get() for safe access; returns None if 'UR' is not found
# Access the Reference length
ref_length_1 = params.get('D_1', None) * params.get('M_1', None) # Using .get() for safe access; returns None if 'UR' is not found
ref_length_2 = params.get('D_2', None) * params.get('M_2', None) # Using .get() for safe access; returns None if 'UR' is not found
ref_length_3 = params.get('D_3', None)  # Using .get() for safe access; returns None if 'UR' is not found
ref_length_3_ST = params.get('D_3', None) / params.get('Ratio', None)
# Access the velocity
U_MAX = params.get('U_MAX', None)  # Using .get() for safe access; returns None if 'UR' is not found
# Access the Reynolds number AND PI
RE = params.get('Re', None)  # Using .get() for safe access; returns None if 'UR' is not found
PI = params.get('PI', None)
# Access the natural frequency
RE = params.get('Re', None)  # Using .get() for safe access; returns None if 'UR' is not found
fn_1 = params.get('Fn_1', None)
fn_2 = params.get('Fn_2', None)
# Access the sampling range
end_time = params.get('TIMEending', None)  # End time of the simulation
smapling_duration = params.get('samping_duration', None)
round_num = params.get('round_num', None)

# Modify this function to filter data for the last 100 seconds
def filter_last_n_seconds(data, end_time):
    if end_time is not None:
        start_time = end_time - smapling_duration  # Calculate the start time of the last 100 seconds
        return data[(data[0] >= start_time) & (data[0] <= end_time)]
    return data
    
def compute_oscillation_frequency(displacement, sampling_interval):
    displacement_fft = fft(displacement.to_numpy())
    freq = fftfreq(len(displacement), d=sampling_interval)
    fosc = freq[np.argmax(np.abs(displacement_fft))]
    return fosc

def compute_vibration_amplitude(displacement):
    # Finding local maxima and minima
    local_max_indices = argrelextrema(displacement.values, np.greater)[0]
    local_min_indices = argrelextrema(displacement.values, np.less)[0]
    
    # Computing average of local maxima and minima
    ymax = displacement.iloc[local_max_indices].mean()
    ymin = displacement.iloc[local_min_indices].mean()
    
    return (ymax - ymin)

def compute_stouhal_1(cl_data, sampling_interval):
    # Convert to numpy array before performing FFT
    lift_fft = fft(cl_data.to_numpy())
    freq = fftfreq(len(cl_data), d=sampling_interval)

    # Find the dominant frequency
    dominant_frequency = freq[np.argmax(np.abs(lift_fft))]
    
    # Assuming you have velocity and characteristic length to compute Stouhal number
    velocity = U_MAX  # Replace with your value
    characteristic_length = ref_length_1  # Replace with your value
    stouhal_number = dominant_frequency * characteristic_length / velocity
    
    return stouhal_number
def compute_stouhal_2(cl_data, sampling_interval):
    # Convert to numpy array before performing FFT
    lift_fft = fft(cl_data.to_numpy())
    freq = fftfreq(len(cl_data), d=sampling_interval)

    # Find the dominant frequency
    dominant_frequency = freq[np.argmax(np.abs(lift_fft))]
    
    # Assuming you have velocity and characteristic length to compute Stouhal number
    velocity = U_MAX  # Replace with your value
    characteristic_length = ref_length_2  # Replace with your value
    stouhal_number = dominant_frequency * characteristic_length / velocity
    
    return stouhal_number
    
def compute_stouhal_3(cl_data, sampling_interval):
    # Convert to numpy array before performing FFT
    lift_fft = fft(cl_data.to_numpy())
    freq = fftfreq(len(cl_data), d=sampling_interval)

    # Find the dominant frequency
    dominant_frequency = freq[np.argmax(np.abs(lift_fft))]
    
    # Assuming you have velocity and characteristic length to compute Stouhal number
    velocity = U_MAX  # Replace with your value
    characteristic_length = ref_length_3_ST  # Replace with your value
    stouhal_number = dominant_frequency * characteristic_length / velocity
    
    return stouhal_number
def plot_data(time, data, average, title, color, avg_label, subplot):
    # Convert pandas Series to numpy arrays explicitly
    time = time.to_numpy()
    data = data.to_numpy()
    subplot.plot(time, data, label=f'{title}', color=color)
    subplot.axhline(y=average, color=color, linestyle='--', label=avg_label)
    #subplot.set_title(f'{title} vs Time')
    subplot.set_xlabel('Time (s)')
    subplot.set_ylabel(f'{title}')
    subplot.legend()
  
# Load data using the updated method for whitespace
cd_1 = pd.read_csv('CTD_1.curve', header=None, sep='\s+')

cd_1 =filter_last_n_seconds(cd_1, end_time)


# Compute time-averaged drag coefficients
cd_ave_1 =  round(cd_1[1].mean(), round_num)


cl_1 = pd.read_csv('CTL_1.curve', header=None, sep='\s+')


cl_1 =filter_last_n_seconds(cl_1, end_time)

# Compute RMS lift coefficients
cl_rms_1 = round(np.sqrt(np.mean(np.square(cl_1[1]))), round_num)


# Example usage for CL_1
sampling_interval_1 = cl_1.iloc[1, 0] - cl_1.iloc[0, 0]  # Time difference between two samples
stouhal_number_1 = round(compute_stouhal_1(cl_1[1], sampling_interval_1), round_num)


# Read displacement data for Cylinder 1 and 2
disp_1 = pd.read_csv('CDisplacement_1.curve', header=None, sep='\s+')


disp_1 = filter_last_n_seconds(disp_1, end_time)

# Assuming the second column is the y-position (since x and z are zero)
y_displacement_1 = disp_1[2]


# Compute non-dimensional vibration amplitude Y*
Y_star_1 = round(compute_vibration_amplitude(y_displacement_1) / (2*ref_length_1), round_num)


# Compute the oscillation frequency fosc for the y-direction
sampling_interval_1 = disp_1.iloc[1, 0] - disp_1.iloc[0, 0]


#oscillation frequency
fosc_1 = round(compute_oscillation_frequency(y_displacement_1, sampling_interval_1), round_num)

# oscillation frequency ratio
f_ratio_1 = round(fosc_1 /fn_1, round_num) 


print("Non-dimensional vibration amplitude Y* for Cylinder 1:", Y_star_1)

print("Oscillation frequency fosc and f* / fn for Cylinder 1:", fosc_1,   f_ratio_1 )

print("Stouhal number for cylinder 1:", stouhal_number_1)

print("Root mean square lift coefficient for cylinder 1:", cl_rms_1)

print("Time-averaged drag coefficient for cylinder 1:", cd_ave_1)


data = [
    SP_U, SP_D, UR,
    cd_ave_1, cl_rms_1, stouhal_number_1, Y_star_1, fosc_1, round(fn_1, round_num), f_ratio_1,
    
    ]
# Dynamic filename based on SPU, SPD, and UR
filename = f"computation_results1_{SP_U}_{SP_D}_{UR}.csv"

# Export to CSV
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'SP_U', 'SP_U', 'UR', 'Cd-ave1', 'Cl-rms1', 'St_1', 'Y*_1', 'fosc_1', 
        'fn_1', 'f_ratio_1'
    ])
    writer.writerow(data)

print(f"Results exported successfully to {filename}.")




