from pyspark.sql import SparkSession
from pyspark.sql.functions import when
import pyspark.sql.functions as F
from column_functions import normalize
from classifiers import knn_classifier
from tkinter import Tk
from tkinter.filedialog import askopenfilename


BENIGN = 'Benign'
MALIGNANT = 'Malignant'
use_z_score_standarization = True


# Create SparkSession
spark = SparkSession \
    .builder \
    .appName("KNN Classifier") \
    .getOrCreate()

# Select file:
Tk().withdraw()
filename = askopenfilename()

# Open CSV
df = spark.read.csv(filename,
                    header='true', inferSchema=True)

# Drop ID column
df = df.drop('id')

# Show top two rows of the dataframe
df.show(2)

# Replace B and M for Benign and Malign
df = df.withColumn('diagnosis',
                   when(df['diagnosis'] == 'B', BENIGN)
                   .otherwise(MALIGNANT))

# Count the number of rows in the dataframe
rows_count = df.count()
# Group by diagnosis and get the count for each class
df_count = df.groupBy('diagnosis').count()

# Get percentage of each class
df_count.withColumn('percentual_count',
                    F.round((F.col('count') / rows_count) * 100, 2)) \
    .show()

df.select('radius_mean', 'area_mean', 'smoothness_mean').describe().show()


if use_z_score_standarization:
    for c in df.columns[1:]:
        c_mean = df.select(F.mean(c)).first()[0]
        c_stddev = df.select(F.stddev(c)).first()[0]
        df = df.withColumn(
            c, (F.col(c) - c_mean) / c_stddev)

else:
    for c in df.columns[1:]:
        col_min = df.select(F.min(c)).first()[0]
        col_max = df.select(F.max(c)).first()[0]

        df = df.withColumn(c,
                           normalize(F.col(c), col_min, col_max))


df.select('radius_mean', 'area_mean', 'smoothness_mean').describe().show()

df.show(2)

train_split, test_split = df.randomSplit(weights=[0.80, 0.20], seed=13)

y_train = train_split.select(df.columns[0])
X_train = train_split.select(df.columns[1:])

y_test = test_split.select(df.columns[0])
X_test = test_split.select(df.columns[1:])


benign_benign = 0
bening_malignant = 0
malignant_benign = 0
malignant_malignant = 0

for i, element in enumerate(X_test.collect()):
    target_class = y_test.collect()[i][0]
    predicted_class = knn_classifier(X_train, y_train, element, 21)

    if target_class == BENIGN:
        if predicted_class == BENIGN:
            benign_benign += 1
        else:
            bening_malignant += 1

    elif target_class == MALIGNANT:
        if predicted_class == MALIGNANT:
            malignant_malignant += 1
        else:
            malignant_benign += 1


benign_row_total = benign_benign + bening_malignant
malignant_row_total = malignant_benign + malignant_malignant
benign_col_total = benign_benign + malignant_benign
malignant_col_total = bening_malignant + malignant_malignant

print("{:10} | {:10} | {:10} | {:10}".format(
    'Label', 'Benign', 'Malignant', 'Row total'))
print("{:10} | {:10} | {:10} | {:10}".format(
    'Benign', benign_benign, bening_malignant, benign_row_total))
print("{:10} | {:10} | {:10} | {:10}".format('Malignant',
      malignant_benign, malignant_malignant, malignant_row_total))
print("{:10} | {:10} | {:10} | {:10}".format('Col total', benign_col_total,
      malignant_col_total, benign_col_total + malignant_col_total))

