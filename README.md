# Spatial and Temporal Patterns of Uber Trip Demand in New York City

by Adrianna Łazuga, Anna Lorenz, Zofia Broszczak, Tsoi Kwan Ma (Group 9)

## Research Question

How is Uber trip demand distributed across space and time in the New York City area, and can location-based clusters reveal distinct daily and hourly demand patterns that may also be useful for future forecasting?

## Data Source

The data comes from the public Kaggle dataset [**Uber Pickups in New York City**](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city?select=uber-raw-data-sep14.csv), published by FiveThirtyEight. The dataset contains Uber pickup records in New York City, including pickup date and time, latitude, longitude, and base information. [(Original Project)](https://colab.research.google.com/drive/1DWK4m2jPTA3LIZLpBz1AWM3Lb_WZhrJE?usp=sharing&fbclid=IwY2xjawRgRgRleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEehWIpuBVHjdC6_yWHwYMFPkC1goSbzCfA840ONaTV0W87GLvcGiZ0-Nm5xdM_aem_KigbOWXRhwQQDLuu0SY_5Q#scrollTo=Lxuj6FnX0Awc)

## Planned Approach

Python will be translated from R, and used to clean and preprocess the Uber pickup data and extract useful temporal variables such as hour of day, day of week, and date. Spatial distribution of pickups will be explored by clustering methods, such as K-means, to identify areas with similar demand patterns.

After identifying these spatial clusters, trip activity across clusters over time will be compared using visualizations and summary statistics. It is also possible to extend the analysis toward simple forecasting or predictive modeling if the clustering results reveal meaningful patterns.

## Language / Tools

The main language used in the project is **Python**.

It is expected to use the following libraries:
- **pandas** for data cleaning and manipulation
- **numpy** for numerical operations
- **matplotlib** and **seaborn** for visualization
- **scikit-learn** for clustering and model evaluation
- possibly **geopandas**, **folium**, or **plotly** for spatial visualization and mapping

## Motivation

Uber trip data provides a real-world example of how urban mobility changes across both space and time. By analyzing these patterns, it is hoped to have a better understanding of travel behavior in a large city and explore how data-driven methods can support transportation analysis and forecasting.
