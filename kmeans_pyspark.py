import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import seaborn as sb
import pandas as pd
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit, count, row_number
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window
from collections import Counter
import pylab as pl
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# Create SparkSession
spark = SparkSession \
    .builder \
    .appName("KMeans clustering") \
    .getOrCreate()


# Select file:
Tk().withdraw()
filename = askopenfilename()

# Open CSV
df = spark.read.csv(filename,
                    header='true', inferSchema=True)

df.show(2)

df.select(df['*']).describe().show()
# df.alias("users").select("users.*").describe().show()

df_count = df.groupBy('categoria').count()
df_count.show()

df_without_category = df.drop('categoria').drop('usuario')


histogram_categories = ['ag', 'co', 'ex', 'ne', 'op', 'wordcount']

fig, axes = plt.subplots(nrows=len(histogram_categories), figsize=(10, 8))

for i, col in enumerate(histogram_categories):
    data_values = df.select(col).rdd.flatMap(lambda x: x).collect()
    axes[i].hist(data_values, bins=10, color="blue", alpha=0.7)
    axes[i].set_title(f"Histogram for {col}")


plt.tight_layout()
plt.show()

pandas_df = df.toPandas()

sb_chart = sb.pairplot(pandas_df, hue='categoria', size=4, vars=["op","ex","ag"], kind='scatter')
plt.show()


X = np.array(pandas_df[["op","ex","ag"]])
y = np.array(pandas_df['categoria'])
print(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
asignar=[]
for row in y:
    asignar.append(colores[row])
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)

plt.show()

cost = np.zeros(10)
features_cols = ['op', 'ex', 'ag']
assembler = VectorAssembler(inputCols=features_cols, outputCol="features")
data = assembler.transform(df)

for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(24).setFeaturesCol("features")
    model = kmeans.fit(data)
    cost[k] = model.summary.trainingCost

# Plot the cost
df_cost = pd.DataFrame(cost[2:])
df_cost.columns = ["cost"]
new_col = [2,3,4,5,6,7,8, 9]
df_cost.insert(0, 'cluster', new_col)

pl.plot(df_cost.cluster, df_cost.cost)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


kmeans = KMeans().setK(5).fit(data)
C = np.array(kmeans.clusterCenters())
print(C)

prediction = kmeans.transform(data)
predictions = prediction.select('prediction').rdd.flatMap(lambda x: x).collect()

labels = predictions

colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

ax.set_xlabel('op')
ax.set_ylabel('ex')
ax.set_zlabel('ag')

plt.show()


# Getting the values and plotting it
f1 = df.select('op').rdd.flatMap(lambda x: x).collect()
f2 = df.select('ex').rdd.flatMap(lambda x: x).collect()

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 2], marker='*', c=colores, s=1000)
plt.show()


f1 = df.select('ex').rdd.flatMap(lambda x: x).collect()
f2 = df.select('ag').rdd.flatMap(lambda x: x).collect()

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.show()


f1 = df.select('op').rdd.flatMap(lambda x: x).collect()
f2 = df.select('ag').rdd.flatMap(lambda x: x).collect()

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.show()



df_labels = spark.createDataFrame(labels, IntegerType())

# New DataFrame 'copy' with columns 'usuario', 'categoria' and 'label'
copy = df.select('usuario', 'categoria')

window_spec = Window.orderBy(lit(0))  # Ventana de ordenamiento vacía
df_labels = df_labels.withColumn("id", row_number().over(window_spec))
copy = copy.withColumn("id", row_number().over(window_spec))

copy = copy.join(df_labels, on=["id"], how="inner").drop("id")

cantidadGrupo = spark.createDataFrame(colores, StringType())

cantidadGrupo = copy.groupby('value').agg(count('*').alias('cantidad'))

cantidadGrupo.show()

conteo = Counter(labels)

for elemento, frecuencia in conteo.items():
    print(f"Elemento: {elemento}, Frecuencia: {frecuencia}")