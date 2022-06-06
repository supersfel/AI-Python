import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import os
current_path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv("Mall_Customers.csv")
print(df)
data = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k = 3
model = KMeans(n_clusters = k, init = 'k-means++', random_state = 10)

def elbow(X):
    sse = []
    for i in range(1,11):
        km = KMeans(n_clusters=i,init='k-means++',random_state=0)
        km.fit(X)
        sse.append(km.inertia_)

    plt.plot(range(1,11),sse,marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.show()
elbow(data)

df['cluster'] = model.fit_predict(data)

final_centroid = model.cluster_centers_
print(final_centroid)

plt.figure(figsize=(8,8))
for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'Annual Income (k$)'], df.loc[df['cluster'] == i, 'Spending Score (1-100)'], label = 'cluster' + str(i))

plt.scatter(final_centroid[:,0], final_centroid[:,1],s=50,c='violet',marker = 'x', label = 'Centroids')
plt.legend()
plt.title(f'K={k} results',size = 15)
plt.xlabel('Annual Income',size = 12)
plt.ylabel('Spending Score',size = 12)
plt.show()