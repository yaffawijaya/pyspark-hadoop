#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import mean
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("TubesITBD").getOrCreate()

file_path = "tubes/Spotify_Dataset_V3.csv" 
df = spark.read.option("header", True).option("delimiter", ";").csv(file_path)

columns_to_drop = ["Title", "Artists", "# of Artist","# of Nationality","Artist (Ind.)","Nationality", "Song URL"]
df1 = df.drop(*columns_to_drop)

window_spec = Window.orderBy(F.monotonically_increasing_id())
df1 = df1.withColumn("songId", F.row_number().over(window_spec))

# Kolom-kolom yang ingin diubah tipe datanya
columns_to_convert = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Valence", "Points (Total)","Points (Ind for each Artist/Nat)"]

# Mengubah tipe data kolom-kolom menjadi Double
for col_name in columns_to_convert:
    df1 = df1.withColumn(col_name, col(col_name).cast("double"))
    
# Kolom-kolom yang ingin digabungkan menjadi vektor
feature_columns = [ "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Valence", "Points (Total)","Points (Ind for each Artist/Nat)" ]

# Membuat VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_cols")

# Menggunakan VectorAssembler untuk membuat vektor fitur
df1 = assembler.transform(df1)

#Mentransformasi data dengan StandardScaler
scaler = StandardScaler(inputCol='features_cols',
                        outputCol='scaled_feature_cols',
                        withStd=True, withMean=True)
scaler = scaler.fit(df1)
df1 = scaler.transform(df1)

## PEMODELAN KLASTERISASI

from pyspark.ml.clustering import KMeans

# Membuat objek KMeans
kmeans = KMeans(k=3, featuresCol="scaled_feature_cols", predictionCol="cluster")

# Melatih model pada data
model = kmeans.fit(df1)

# Membuat klaster pada data
df_kmeans = model.transform(df1)

# # Menampilkan hasil klasterisasi
# df_kmeans.show(truncate=False)

# Membagi data menjadi data yang akan menjadi keluaran
atas, bawah = df_kmeans.randomSplit([0.995, 0.005], seed=1234)

bawah.show()
