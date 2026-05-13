# Spatial and temporal patterns of Uber trip demand around New York City

## Authors
Adrianna Łazuga, Anna Lorenz, Zofia Broszczak, Tsoi Kwan Ma (Group 9)

## Research Question
How is Uber trip demand distributed across space and time in the New York and New Jersey states, and can location-based clusters reveal distinct daily and hourly demand patterns that may also be useful for future forecasting?

## Approach
The original project was developed in R and is available under https://colab.research.google.com/drive/1DWK4m2jPTA3LIZLpBz1AWM3Lb_WZhrJE?usp=sharing#scrollTo=iAa_MnRh0slX.
The project was reimplemented in Python and includes data cleaning, feature extraction (hour and date), and a 95%–5% train–test split. K-means clustering is applied to pickup coordinates, with the number of clusters selected using the elbow method, silhouette scores, and dissimilarity measures (final choice: K = 8). Cluster quality is evaluated using silhouette analysis and the GAP statistic, and temporal patterns are examined through hourly and daily trip aggregations. The trained model is then applied to the test set to assess how well the clustering generalizes.

## Language / Tools
The language used in the project is **Python**.
All required packages and their versions are listed in the requirements.txt file.

## Motivation
Uber trip data provides a real-world example of how urban mobility changes across both space and time. By analyzing these patterns, it is hoped to have a better understanding of travel behavior in a large city and explore how data-driven methods can support transportation analysis and forecasting.

## Requirements
- Python >= 3.10
- Internet connection (required at runtime to fetch the US states GeoJSON map)
- Python libraries (see requirements.txt for exact versions):
  - pandas - data loading and manipulation
  - numpy - numerical operations
  - matplotlib - plotting
  - seaborn - statistical data visualization
  - scikit-learn - KMeans clustering and silhouette scoring
  - scipy - pairwise distance calculations
  - geopandas - geospatial map rendering
  - yellowbrick - silhouette visualizer
  - gap-stat - GAP statistic for optimal K selection

## Setup
Please follow the instructions below to create a virtual environment and install the required Python dependencies.
### macOS / Linux
``` 
git clone https://github.com/tsoikwanma/RRproject2026.git # Clone the repository
cd <repository-folder>

python3 -m venv .venv # Create a virtual environment
source .venv/bin/activate # Activate the virtual environment

pip install -r requirements.txt # Install required dependencies
```
### Windows
```
git clone https://github.com/tsoikwanma/RRproject2026.git # Clone the repository
cd <repository-folder>

python -m venv .venv # Create a virtual environment
.venv\Scripts\activate # Activate the virtual environment

pip install -r requirements.txt # Install required dependencies
```

## How to run
Once the setup is complete, run the command below to start the analysis script:
```
python src/main.py
```

## Expected output
- The first section generates a map of New Jersey state, including the Uber data points in the figure "uber_nj_map.png".
- Figure "uber_nj_elbow.png shows the WSS values using elbow method.
- The next figure, "uber_nj_silhouette.png",  represents the silhouette scores for different numbers of clusters.
- Figure "uber_nj_dissimilarity.png" visualises the dissimilarity between different numbers of clusters.
- Next steps include visualising the clusters on the map in figure "uber_nj_clusters.png" and evaluation of clusters based on the quality measures in figure "uber_nj_silhouette_coefficient.png". Figure "uber_nj_gap_statistic.png" and table "uber_nj_gap_statistic.csv" represent the calculated GAP statistic for clustering.
- Next steps include hourly trip counts for each cluster in figure "uber_nk_hourly_trip_counts.png" and the count of daily trips in figure "uber_nj_daily_trip_counts.png".
- The last figures show plots visualising training and test data in figures "uber_nj_training_clusters.png" and "uber_nj_test_clusters.png".

- The whole execution time is expected at approximately 3-5 minutes.

## Data
- The data comes from the public Kaggle dataset [**Uber Pickups in New York City**](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city?select=uber-raw-data-sep14.csv), published by FiveThirtyEight. 
- The dataset contains Uber pickup records in New York City, including pickup date and time, latitude, longitude, and base information.
- In the repository, the data is present under data\uber-raw-data-sep14.csv

- Link to the original R project that is being reproduced: [(Original Project)](https://colab.research.google.com/drive/1DWK4m2jPTA3LIZLpBz1AWM3Lb_WZhrJE?usp=sharing&fbclid=IwY2xjawRgRgRleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEehWIpuBVHjdC6_yWHwYMFPkC1goSbzCfA840ONaTV0W87GLvcGiZ0-Nm5xdM_aem_KigbOWXRhwQQDLuu0SY_5Q#scrollTo=Lxuj6FnX0Awc)

## Repository structure

RRproject2026

├── README.md

├── data/

├── src/

├── output/

├── requirements.txt

└── .gitignore

- The data folder contains the uber-raw-data-sep14.csv dataset used in the analysis.
- The src folder contains a single script (main.py) because each section depends on variables and data produced in earlier steps, making it easiest to reproduce the full analysis within one continuous workflow.
- The ouptut folder contains the results of the analysis including plots and a dataframe:
  1. EDA map of Uber trips - uber_nj_map.png
  2. Plots for calculating the optimal number of clusters - uber_nj_elbow.png, uber_nj_silhouette.png, uber_nj_dissimilarity.png
  3. Plot of clusters - uber_nj_clusters.png
  4. Quality measures - uber_nj_silhouette_coefficient.png, uber_nj_gap_statistic.png, uber_nj_gap_statistic.csv
  5. Time analysis - uber_nk_hourly_trip_counts.png, uber_nj_daily_trip_counts.png
  6. Forecasting - uber_nj_training_clusters.png, uber_nj_test_clusters.png
