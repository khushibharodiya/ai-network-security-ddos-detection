import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

df=pd.read_csv("cybersecurity.csv")
print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True)*100)

plt.figure(figsize=(8,6))
sns.countplot(
    data=df,
    x="label"
)
plt.title("label distribution")
plt.xlabel("label")
plt.ylabel("no.of records")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,6))
sns.countplot(
    data=df,
    x="protocol",
    order=df["protocol"].value_counts().index
)
plt.title("protocol distribution")
plt.xlabel("protocol")
plt.ylabel("no.of records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(
    data=df,
    x="protocol",
    hue="label"                       #hue is for split the data into groups according to another variable
)
plt.title("protocol distribution")
plt.xlabel("protocol")
plt.ylabel("no.of records")
plt.legend(
    title="Traffic Type",               #legend is for bins information
    labels=["Benign", "Malicious"]
)

plt.show()


internal_label=pd.crosstab(                #crosstab is combine two column and tells that how many contain 0 and contain 1
    df["protocol"],
    df["label"]
)
print(internal_label)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="is_internal_traffic",
    hue="label"
)

plt.title("Internal Traffic by Traffic Label")
plt.xlabel("Is Internal Traffic")
plt.ylabel("Number of Network Activities")

plt.show()


plt.show()
internal_label=pd.crosstab(
    df["is_internal_traffic"],
    df["label"]
)
print(internal_label)


plt.figure(figsize=(8,6))
sns.histplot(
    data=df,
    x="bytes_sent",
    bins=30,
    kde=True
)
plt.title("distribution of bytes_sent")
plt.xlabel("bytes_types")
plt.ylabel("no.of records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(
    data=df,
    x="bytes_received",
    bins=30,
    kde=True
)
plt.title("distribution of bytes_recieved")
plt.xlabel("bytes_recieved")
plt.ylabel("no.of records")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="src_port",
    bins=30,
    kde=True
)

plt.title("Distribution of Source Ports")
plt.xlabel("Source Port")
plt.ylabel("Number of Records")

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="dst_port",
    bins=30,
    kde=True
)

plt.title("Distribution of Destination Ports")
plt.xlabel("Destination Port")
plt.ylabel("Number of Records")

plt.show()


#box plot of src plot
plt.figure(figsize=(8, 4))

sns.boxplot(x=df["src_port"])

plt.title("Boxplot of src_port")
plt.xlabel("src_port")

plt.show()

#boxplot of dst_port
plt.figure(figsize=(8, 4))

sns.boxplot(x=df["dst_port"])

plt.title("Boxplot of dst_port")
plt.xlabel("dst_port")

plt.show()

#boxplot of bytes_sent
plt.figure(figsize=(8, 4))

sns.boxplot(x=df["bytes_sent"])

plt.title("Boxplot of Bytes Sent")
plt.xlabel("Bytes Sent")

plt.show()
#boxplot of bytes_recieved
plt.figure(figsize=(8, 4))

sns.boxplot(x=df["bytes_received"])

plt.title("Boxplot of Bytes recieved")
plt.xlabel("Bytes recieved")

plt.show()
def detect_outliers(df,column):
    # NOTE: parameter was previously named "columns" while the body
    # used "column" -- it only worked because "column" happened to
    # still be set as a leftover variable from the for-loop below
    # (Python's global/enclosing scope lookup). Renamed the
    # parameter to match so the function is correct on its own,
    # independent of any outer loop state.
    Q1=df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print("\nColumn:", column)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of potential outliers:", len(outliers))
    print(outliers["label"].value_counts())
    print("Percentage of potential outliers:",(len(outliers) / len(df)) * 100)
    return outliers
numeric_columns = [
    "src_port",
    "dst_port",
    "bytes_sent",
    "bytes_received"
]

for column in numeric_columns:
    detect_outliers(df, column)


#boxplot for src_port & label

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="src_port"
)

plt.title("Source Port Distribution by Traffic Label")
plt.xlabel("Traffic Label (0 = Benign, 1 = Malicious)")
plt.ylabel("Source Port")

plt.show()

#boxplot for dst_port vs label

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="dst_port"
)

plt.title("Destination Port Distribution by Traffic Label")
plt.xlabel("Traffic Label (0 = Benign, 1 = Malicious)")
plt.ylabel("Destination Port")

plt.show()

#boxplot for bytes_sent vs label 

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="bytes_sent"
)

plt.title("bytes_sent Distribution by Traffic Label")
plt.xlabel("Traffic Label (0 = Benign, 1 = Malicious)")
plt.ylabel("bytes_sent")

plt.show()

#boxlot for bytes_received

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="bytes_received"
)

plt.title("bytes_received Distribution by Traffic Label")
plt.xlabel("Traffic Label (0 = Benign, 1 = Malicious)")
plt.ylabel("bytes_received")

plt.show()

grouped_stats=df.groupby("label")[[
    "src_port",
    "dst_port",
    "bytes_sent",
    "bytes_received"
]].agg(["mean","median"])
print(grouped_stats.to_string())              #to give whole output not in ..

print(df.groupby("protocol")["bytes_sent"].mean())

corr= df.corr(numeric_only=True)
print(corr)
plt.figure(figsize=(10,7))
sns.heatmap(
    corr,
    annot=True,                     #annot means display the corelation value inside the cell
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("corelation heatmap of numerical features")
plt.show()

