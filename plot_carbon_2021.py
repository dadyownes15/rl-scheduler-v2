import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the carbon intensity data
df = pd.read_csv('data/DK-DK2_hourly_carbon_intensity_noFeb29.csv')

# Extract 2021 carbon intensity values
carbon_2021 = df['2021'].values

# Create histogram/distribution plot
plt.figure(figsize=(12, 8))

# Plot histogram
plt.subplot(2, 1, 1)
plt.hist(carbon_2021, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Carbon Intensity (gCO₂eq/kWh)')
plt.ylabel('Frequency')
plt.title('Distribution of Carbon Intensity Values for 2021')
plt.grid(True, alpha=0.3)

# Plot time series
plt.subplot(2, 1, 2)
plt.plot(range(len(carbon_2021)), carbon_2021, linewidth=0.8)
plt.xlabel('Hours (from start of year)')
plt.ylabel('Carbon Intensity (gCO₂eq/kWh)')
plt.title('Carbon Intensity Time Series for 2021')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some basic statistics
print(f"2021 Carbon Intensity Statistics:")
print(f"Min: {np.min(carbon_2021):.2f} gCO₂eq/kWh")
print(f"Max: {np.max(carbon_2021):.2f} gCO₂eq/kWh")
print(f"Mean: {np.mean(carbon_2021):.2f} gCO₂eq/kWh")
print(f"Median: {np.median(carbon_2021):.2f} gCO₂eq/kWh")
print(f"Standard Deviation: {np.std(carbon_2021):.2f} gCO₂eq/kWh")
print(f"Total data points: {len(carbon_2021)}") 