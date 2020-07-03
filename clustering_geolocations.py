import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium 
import re
import hdbscan

from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080']*10


def data_preprocessing():
    data = pd.read_csv('Data/taxi_data.csv')
    data.dropna(inplace=True)
    data.drop_duplicates(subset = ['LON', 'LAT'])
    x = np.array(data[['LON', 'LAT']], dtype='float64')

    return data

def optimal_number_of_clusters_graphic_visualization(data):
    x = np.array(data[['LON', 'LAT']], dtype='float64')    
    wcss = []
    for i in range (1,30):  
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    plt.scatter(range(1,30), wcss)    
    plt.show()

def optimal_number_of_clusters(data):   
    x = np.array(data[['LON', 'LAT']], dtype='float64')    
    best_silhouette, best_k = -1, 0

    for k in tqdm(range(2, 50)):
        model = KMeans(n_clusters=k, random_state=1).fit(x)
        class_predictions = model.predict(x)
        
        curr_silhouette = silhouette_score(x, class_predictions)
        if curr_silhouette > best_silhouette:
            best_k = k
            best_silhouette = curr_silhouette
            
    print(f'K={best_k}')
    print(f'Silhouette Score: {best_silhouette}')              
    return best_k 

def clustering(data):
    x = np.array(data[['LON', 'LAT']], dtype='float64')
    k = 70
    model = KMeans(n_clusters=k, random_state=17).fit(x)
    class_predictions = model.predict(x)
    data['cluster_column'] = class_predictions

def outliers(data):
    x = np.array(data[['LON', 'LAT']], dtype='float64')
    model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.01)
    class_predictions = model.fit_predict(x)
    data['cluster_hbdscan'] = class_predictions
    
def geographical_visualization(m, data, cluster_column):
    
    for _, row in data.iterrows():
        
        cluster_colour = cols[row.cluster_column]        
        folium.CircleMarker(
            location = [row.LAT, row.LON], 
            radius = 5, 
            popup=row.cluster_column,
            color=cluster_colour, 
            fill=True,
            fill_color='#1787FE'
        ).add_to(m)

    return m


def main():
    data = data_preprocessing()
    #optimal_number_of_clusters_graphic_visualization(data)
    clustering(data)
    outliers(data)
    
    m = folium.Map(location=[data.LAT.mean(), data.LON.mean()], zoom_start=9, tiles='Stamen Toner')
    m = geographical_visualization(m, data, data.cluster_column)
    m = geographical_visualization(m, data, data.cluster_hbdscan)
    m.save('map.html')

main()