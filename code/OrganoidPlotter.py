import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set Arial font for all matplotlib text and double the font size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 20  # Double the default font size (default is usually 10)
plt.rcParams['axes.titlesize'] = 24  # Larger title size
plt.rcParams['axes.labelsize'] = 20  # Axis label size
plt.rcParams['xtick.labelsize'] = 18  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 18  # Y-axis tick label size

# Savitzky–Golay filter for smoothing
try:
    from scipy.signal import savgol_filter  # noqa: F401
except ImportError as e:
    raise ImportError(
        "SciPy is required for Savitzky–Golay filtering. Install it via 'pip install scipy' or 'conda install scipy'."
    ) from e

# ------------------ Configuration ------------------
SHOW_RAW = 0          # 0 = hide raw traces, 1 = include raw traces
APPLY_ROLLING = 1     # 1 = apply an additional rolling‑average filter after SavGol
ROLLING_WINDOW = 50    # window size (in data points) for rolling average

Y_PAD_FRAC = .5     # Fractional padding added above & below data range on y‑axis

# ------------------ File Paths ------------------
output_folder = "/Coding/DrewOrganoidTrackingandGraphing/output"
metrics_path = os.path.join(output_folder, "white_organoid_metrics.csv")

# ------------------ Load Data -------------------
# Parse the timestamp column as datetime and set it as index

df = pd.read_csv(metrics_path, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# ------------------ Metrics ---------------------
metrics = [
    "area",
    "solidity",
]

# ------------------ Outlier Removal & Smoothing ------------------
# Parameters for Savitzky–Golay
z_thresh = 3           # Z‑score threshold for outlier detection
window_length = 10    # Savitzky–Golay window (must be odd)
poly_order = 7         # Polynomial order for Savitzky–Golay

# Create a cleaned/smoothed DataFrame
clean_df = df.copy()

for metric in metrics + ["centroid_x", "centroid_y"]:
    series = clean_df[metric]

    # Remove centroid outlier below (650, 650)
    if metric in ["centroid_x", "centroid_y"]:
        mask = ~((clean_df["centroid_x"] < 650) & (clean_df["centroid_y"] < 650))
        series = series[mask]

    # ----- Outlier removal (Z‑score) -----
    z_scores = (series - series.mean()) / series.std(ddof=0)
    series_filtered = series.mask(z_scores.abs() > z_thresh)

    # Interpolate removed outliers and fill edges
    series_filtered = series_filtered.interpolate()
    series_filtered = series_filtered.bfill().ffill()

    # ----- Savitzky–Golay smoothing -----
    n_points = len(series_filtered)
    win = min(window_length, n_points if n_points % 2 == 1 else n_points - 1)
    if win < 5:
        win = 5 if n_points >= 5 else (n_points | 1)

    sg_series = savgol_filter(series_filtered, window_length=win, polyorder=min(poly_order, win - 1))

    # Optional rolling‑average smoothing on top of SavGol
    if APPLY_ROLLING and ROLLING_WINDOW >= 2:
        sg_series = pd.Series(sg_series, index=series_filtered.index)
        ra_series = sg_series.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
        smoothed = ra_series.values
    else:
        smoothed = sg_series

    # Store back in DataFrame, aligning to original index
    smoothed_series = pd.Series(smoothed, index=series_filtered.index)
    clean_df[f"{metric}_smooth"] = smoothed_series.reindex(clean_df.index)

# ------------------ Conversion ------------------
# Conversion factor: 1 pixel = 0.00180 mm = 1.8 um
PIXEL_TO_UM = 1.8

# Convert centroid and metric values to micrometers and square millimeters for plotting
clean_df['centroid_x_um'] = clean_df['centroid_x'] * PIXEL_TO_UM
clean_df['centroid_y_um'] = clean_df['centroid_y'] * PIXEL_TO_UM
clean_df['centroid_x_mm'] = clean_df['centroid_x'] * 0.0018
clean_df['centroid_y_mm'] = clean_df['centroid_y'] * 0.0018
clean_df['area_um'] = clean_df['area'] * (PIXEL_TO_UM ** 2)
clean_df['area_smooth_um'] = clean_df['area_smooth'] * (PIXEL_TO_UM ** 2)
clean_df['area_mm2'] = clean_df['area'] * (0.0018 ** 2)
clean_df['area_smooth_mm2'] = clean_df['area_smooth'] * (0.0018 ** 2)
# If solidity is unitless, do not convert

# ------------------ Plotting --------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1.2]})

# Plot area and solidity as time series
n_points = len(clean_df)
total_days = 14
x_days = np.linspace(0, total_days, n_points)
day_ticks = np.arange(0, total_days + 1, 2)

for i, (ax, metric) in enumerate(zip(axes[:2], metrics)):
    if metric == 'area':
        y_data = clean_df['area_smooth_mm2']
        ylabel = 'Area (mm²)'
    else:
        y_data = clean_df[f'{metric}_smooth']
        ylabel = f'{metric.title()} (A.U.)'
    ax.plot(x_days, y_data)
    y_min = y_data.min()
    y_max = y_data.max()
    pad = (y_max - y_min) * Y_PAD_FRAC
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(metric.replace('_', ' ').title(), fontfamily='Arial', fontsize=24)
    ax.set_ylabel(ylabel, fontfamily='Arial', fontsize=20)    # Remove grid
    ax.grid(False)
    # Major ticks at every 2 days, minor ticks at every 1 day
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([str(int(tick)) for tick in day_ticks])
    minor_ticks = np.arange(1, total_days, 2)
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=6, width=1, direction='out')
    ax.tick_params(axis='x', which='minor', labelbottom=False)  # No labels for minor ticks
    ax.set_xlabel("Day", fontfamily='Arial', fontsize=20)
    
    # Set Arial font for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(18)
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(18)

# Plot centroid_x vs centroid_y as a connected line plot with a color gradient that darkens over time
ax = axes[2]
centroid_x_sampled = clean_df['centroid_x_mm'][::11].values
centroid_y_sampled = clean_df['centroid_y_mm'][::11].values
num_points = len(centroid_x_sampled)
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
points = np.array([centroid_x_sampled, centroid_y_sampled]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
# Use a rainbow colormap for the gradient
lc = LineCollection(segments, cmap='rainbow', norm=plt.Normalize(0, num_points-1))
lc.set_array(np.arange(num_points-1))
lc.set_linewidth(2)
lc.set_alpha(0.4)
ax.add_collection(lc)
# Set axis ranges for centroid plot in millimeters
ax.set_xlim(870 * 0.0018, 940 * 0.0018)
ax.set_ylim(920 * 0.0018, 960 * 0.0018)
ax.set_title("X vs Y Centroid", fontfamily='Arial', fontsize=24)
ax.set_xlabel("X Centroid (mm)", fontfamily='Arial', fontsize=20)
ax.set_ylabel("Y Centroid (mm)", fontfamily='Arial', fontsize=20)
# Remove grid
ax.grid(False)

# Set Arial font for centroid plot tick labels
for label in ax.get_xticklabels():
    label.set_fontfamily('Arial')
    label.set_fontsize(18)
for label in ax.get_yticklabels():
    label.set_fontfamily('Arial')
    label.set_fontsize(18)

# Add a colorbar legend for the rainbow gradient indicating Day 0 and Day 14
cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])  # [left, bottom, width, height]
norm = Normalize(vmin=0, vmax=total_days)
cbar = ColorbarBase(cbar_ax, cmap='rainbow', norm=norm, orientation='vertical')
cbar.set_ticks([0, total_days])
cbar.set_ticklabels(['Day 0', f'Day {total_days}'])
# Set Arial font for colorbar tick labels
for label in cbar.ax.get_yticklabels():
    label.set_fontfamily('Arial')

plt.tight_layout(rect=[0, 0, 0.91, 1])

# ------------------ Save Figure -----------------
# Export the composite figure as a PDF in the same output folder
pdf_filename = "white_organoid_metrics_plots.pdf"
pdf_path = os.path.join(output_folder, pdf_filename)
fig.savefig(pdf_path, format="pdf")
print(f"Figure saved to: {pdf_path}")

plt.show()
plt.close(fig)  # Ensure figure is closed after showing