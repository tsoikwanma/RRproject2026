from pathlib import Path

PROJECT_ROOT = Path(_file_).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

#Load the data
uber_data = pd.read_csv("uber-raw-data-sep14.csv")
uber_data = uber_data.dropna()
locations = uber_data[['Lat', 'Lon']]

#create columns Date and Hour
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'])
uber_data['date'] = uber_data['Date/Time'].dt.date
uber_data['hour'] = uber_data['Date/Time'].dt.hour
uber_data.head()

# Create a train-test split (95% train, 5% test)
np.random.seed(1)

train_indices = uber_data.sample(frac = 0.95).index

uber_data["train"] = False
uber_data.loc[train_indices, "train"] = True

# Verify the split worked correctly
print("Number of training samples:", len(uber_data[uber_data["train"] == True]), "\n")
print("Number of test samples:", len(uber_data[uber_data["train"] == False]), "\n")

#Visualize the data on the map
# The map of the USA
path = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
map_data = gpd.read_file(path)

# Latitude and longitude limits for New Jersey
lon_min = -74.7
lon_max = -73.0
lat_min = 40.2
lat_max = 41.25

#Creating geo data from uber_data
uber_data_map = gpd.GeoDataFrame(uber_data, geometry=gpd.points_from_xy(uber_data["Lon"], uber_data["Lat"])
)
# Plot the map of the USA
ax = map_data.plot(color = "#e5e5e5")
# Plot the Uber data points within New Jersey
uber_data_map.plot(ax = ax, color = "blue", alpha = 0.01, markersize = 0.5)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_title("Uber Trip Locations in New Jersey")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.savefig(OUTPUT_DIR / "uber_nj_map.png", dpi=300, bbox_inches="tight")