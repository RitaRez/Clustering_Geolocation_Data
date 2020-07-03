import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium 
import re
import hdbscan

from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('taxi_data.csv')
data.dropna(inplace=True)
data.drop_duplicates(subset = ['LON', 'LAT'])

x = np.array(data[['LON', 'LAT']], dtype='float64')
plt.scatter(x[:, 0], x[:, 1], alpha=0.3, s=50)

m = folium.Map(location=[data.LAT.mean(), data.LON.mean()], zoom_start=9, tiles='Stamen Toner')
for _, row in data.iterrows():
    folium.CircleMarker(
        location = [row.LAT, row.LON], 
        radius = 5, 
        popup=re.sub(r'[^a-zA-Z ]+', '', row.NAME), 
        color='#1787FE', 
        fill=True,
        fill_color='#1787FE'
    ).add_to(m)

m.save('map.html')
cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080']*10