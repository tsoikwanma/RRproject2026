from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist
from yellowbrick.cluster import SilhouetteVisualizer
from gap_statistic import OptimalK

# Data preparation
#Load the data
uber_data = pd.read_csv(DATA_DIR / "uber-raw-data-sep14.csv")
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

ax.figure.savefig(OUTPUT_DIR / "uber_nj_map.png", dpi=300, bbox_inches="tight")



# K-means cluster number calculation
# Reproducibility
np.random.seed(123)

# Randomly sample 100,000 observations from training data
train_data = uber_data.loc[uber_data["train"] == True, ["Lat", "Lon"]]

sample_locations = train_data.sample(n=100000, random_state=123)

# WSS plot
wss_values = []

k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        n_init=75,
        random_state=123
    )
    kmeans.fit(sample_locations)
    wss_values.append(kmeans.inertia_)

# Plot WSS values to visualize the elbow method
plt.figure()
plt.plot(k_values, wss_values, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Total within-clusters sum of squares")
plt.title("Elbow Method")
plt.savefig(OUTPUT_DIR / "uber_nj_elbow.png", dpi=300, bbox_inches="tight")
plt.show()

# Reproducibility
np.random.seed(123)

# Randomly sample 10,000 observations from training data
train_data1 = uber_data.loc[uber_data["train"] == True,["Lat", "Lon"]]

sample_locations1 = train_data1.sample(n=10000, random_state=123)

silhouette_scores = []

for k in range(2, 11):
    kmeans_result = KMeans(
        n_clusters=k,
        n_init=75,
        random_state=123
    )

    cluster_labels = kmeans_result.fit_predict(sample_locations1)

    avg_sil_width = silhouette_score(
        sample_locations1,
        cluster_labels
    )

    silhouette_scores.append(avg_sil_width)

# Plot average silhouette scores
plt.figure()
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Average Silhouette score")
plt.title("Silhouette scores for different K values")
plt.savefig(OUTPUT_DIR / "uber_nj_silhouette.png", dpi=300, bbox_inches="tight")
plt.show()

print(silhouette_scores)


dissimilarity_scores = []

for k in range(2, 11):
    kmeans_result = KMeans(
        n_clusters=k,
        n_init=75,
        random_state=123
    )

    kmeans_result.fit(sample_locations)
    cluster_centers = kmeans_result.cluster_centers_
    distances_between_clusters = pdist(cluster_centers)
    avg_dissimilarity = np.mean(distances_between_clusters)
    dissimilarity_scores.append(avg_dissimilarity)

# Plot dissimilarity scores
plt.figure()
plt.plot(range(2, 11), dissimilarity_scores, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Average Dissimilarity (Distance between centers)")
plt.title("Dissimilarity scores for different k values")
plt.savefig(OUTPUT_DIR / "uber_nj_dissimilarity.png", dpi=300, bbox_inches="tight")
plt.show()

# Perform KMeans clustering with a chosen number of clusters
k = 8
kmeans_result = KMeans(n_clusters = k, n_init = 75, random_state = 42)
kmeans_result.fit(uber_data.loc[uber_data["train"] == True, ["Lon", "Lat"]])

# Assign clusters to the training data
uber_data.loc[uber_data["train"] == True, "cluster"] = kmeans_result.labels_

plt.figure(figsize = (10, 6))
scatter = plt.scatter(uber_data["Lon"], uber_data["Lat"],
                      c = uber_data["cluster"], cmap = "tab10", alpha = 0.6)
plt.colorbar(scatter, label = "Cluster")
plt.title("Uber Trip Clusters by Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(OUTPUT_DIR / "uber_nj_clusters.png", dpi=300, bbox_inches="tight")
plt.show()

# Quality measures
# Silhouette Index
np.random.seed(1)
train_sample = uber_data[uber_data["train"] == True].copy()
train_sample = train_sample.sample(n = 10000, random_state = 1)

silhouette_scores = silhouette_samples(train_sample[["Lon", "Lat"]], kmeans_result.predict(train_sample[["Lon", "Lat"]]))
print(pd.Series(silhouette_scores).describe())
km = KMeans(n_clusters = k, n_init = 75, random_state = 42)
visualizer = SilhouetteVisualizer(km, colors = "tab10")
visualizer.fit(train_sample[["Lon", "Lat"]])
visualizer.show(outpath=OUTPUT_DIR / "uber_nj_silhouette_coefficient.png")

# GAP Statistic
# Calculate GAP statistic for KMeans clustering on the random sample
np.random.seed(123)
optimalK = OptimalK(n_jobs = 1)
n_clusters = optimalK(train_sample[["Lon", "Lat"]].values, cluster_array = range(1, 11), n_refs = 100)
print(optimalK.gap_df)
gap_df = optimalK.gap_df
optimalK.gap_df.to_csv(OUTPUT_DIR / "uber_nj_gap_statistic.csv", index=False)
plt.figure(figsize = (8, 6))
plt.errorbar(gap_df["n_clusters"], gap_df["gap_value"],
             yerr = gap_df["sk"], fmt = 'o-', color='black',
             ecolor = 'red', capsize = 4, markersize = 4)
plt.title("Gap Values by Cluster Count")
plt.xlabel("k")
plt.ylabel("Gap$_k$")
plt.xticks(range(1, 11, 2))
plt.grid(False)
plt.savefig(OUTPUT_DIR / "uber_nj_gap_statistic.png", dpi = 300, bbox_inches = "tight")
plt.show()