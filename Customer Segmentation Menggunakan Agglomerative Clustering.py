import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
data = pd.read_csv("D:/IT DEL/SEMESTER 5/CERTAN/Proyek CERTAN/Mall_Customers.csv")
data.head()

# Gantilah 'nama_file.csv' dengan path atau URL dataset Anda
dataset = pd.read_csv("D:/IT DEL/SEMESTER 5/CERTAN/Proyek CERTAN/Mall_Customers.csv")

# Hanya akan menggunakan 2 features agar dapat divisualisasikan
# "Annual Income" dan "Spending Score"
X = dataset.iloc[:, 3:5]
X.head()

# Ukuran data (kolom, baris)
X.shape

# Cek data missing
X.isnull().sum()

# Ringkasan data
X.describe()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

# Memotongnya di garis vertikal yang paling panjang
# Dia tidak berpotongan dengan garis horizontal manapun
# Di horizontalnya, yang memotong paling banyak titik

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
ac.fit(X)