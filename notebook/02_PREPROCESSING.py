import pandas as pd
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df= pd.read_csv("cybersecurity.csv")
print(df.head())
print(df.shape)


##extract hour day month fro timestamp
df["timestamp"]=pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M"
    )

df["hour"]=df["timestamp"].dt.hour
df["day"]=df["timestamp"].dt.day
df["month"]=df["timestamp"].dt.month
df["day_of_week"]=df["timestamp"].dt.dayofweek

##for source_ipadress

print(df["src_ip"].head(10))
print("\nunique source IPs:",df["src_ip"].nunique())
ip= ipaddress.ip_address("192.168.1.10")

print(ip)
print(ip.is_private)
df["src_ip_is_private"]=df["src_ip"].apply(
    lambda x:ipaddress.ip_address(x).is_private             ##lambda is used for Take one value and call that value x because .apply() goes through the column one value at a time
)
print(df[["src_ip", "src_ip_is_private"]].head(10))
print(df["src_ip_is_private"].value_counts())

print(
    pd.crosstab(                            ##crosstab Compare two categorical columns and count how they occur together.                     
        df["src_ip_is_private"],
        df["label"],
        normalize=True,
    )*100
)
print(df["dst_ip"].head(10))
print("Unique destination IPs:", df["dst_ip"].nunique())
df["dst_ip_is_private"] = df["dst_ip"].apply(
    lambda x: ipaddress.ip_address(x).is_private
)
print(
    df[["dst_ip", "dst_ip_is_private"]].head(10)
)
print(df["dst_ip_is_private"].value_counts())

print(
    pd.crosstab(
        df["dst_ip_is_private"],
        df["label"],
        normalize=True
    )*100
)

## for protocol

print(df["protocol"].value_counts())

print(
    pd.crosstab(
        df["protocol"],
           df["label"],
           normalize="index"
    )*100
)



#for is_internal_traffic

print(df["is_internal_traffic"].value_counts())
print(
    pd.crosstab(
        df["is_internal_traffic"],
        df["label"],
        normalize=True
    )*100
)

df["is_internal_traffic"] = df["is_internal_traffic"].astype(int)             #convert the datatypes

print(df["is_internal_traffic"].dtype)

#for user_agent

print(df["user_agent"].nunique())
print(df["user_agent"].value_counts())
print(
    pd.crosstab(
        df["user_agent"],
        df["label"],
        normalize=True
    )*100
)
print(df["user_agent"].unique())

def get_user_agent_type(user_agent):
    if "Chrome" in user_agent or "Firefox" in user_agent or "Safari" in user_agent:
        return "browser"
    elif "curl" in user_agent:
        return "curl"
    elif "python-requests" in user_agent:
        return "python_requests"
    elif "python-urllib" in user_agent:
        return "python_urllib"
    elif "sqlmap" in user_agent:
        return "sqlmap"
    elif "zgrab" in user_agent:
        return "zgrab"
    elif "Googlebot" in user_agent:
        return "googlebot"
    else:
        return "other"

df["user_agent_type"]=df["user_agent"].apply(get_user_agent_type)
print(df[["user_agent","user_agent_type"]].head(10))
print(df["user_agent_type"].value_counts())
print(
    pd.crosstab(
        df["user_agent_type"],
        df["label"],
        normalize="index"
    ) * 100
)


## for url
print("Missing URLs:", df["url"].isna().sum())
print("unique urls:",df["url"].nunique())
print(df["url"].head(10))
print(
    pd.crosstab(
        df["url"].isna(),
        df["label"],
        normalize="index"
    ) * 100
)
df["url_missing"]=df["url"].isna().astype(int)
print(df["url_missing"].value_counts())
df["url_length"]=df["url"].fillna("").str.len()
print(df[["url", "url_length"]].head(10))


df["url_special_chars"] = df["url"].fillna("").str.count(r"[^a-zA-Z0-9]")

print(df[["url", "url_special_chars"]].head(10))

print(df.columns.tolist())

x = df[[
    "src_port",
    "dst_port",
    "bytes_sent",
    "bytes_received",
    "protocol",
    "is_internal_traffic",
    "hour",
    "day",
    "month",
    "day_of_week",
    "src_ip_is_private",
    "dst_ip_is_private",
    "user_agent_type",
    "url_missing",
    "url_length",
    "url_special_chars"
]]

y = df["label"]

print("x shape:",x.shape)
print("y shape:",y.shape)

#traing and testing part
x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("x_train:",x_train.shape)
print("x_test:",x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("categorical columns:")
print(x_train[["protocol","user_agent_type"]].dtypes)

categorical_columns = [
    "protocol",
    "user_agent_type"
]

numerical_columns = [
    "src_port",
    "dst_port",
    "bytes_sent",
    "bytes_received",
    "is_internal_traffic",
    "hour",
    "day",
    "month",
    "day_of_week",
    "src_ip_is_private",
    "dst_ip_is_private",
    "url_missing",
    "url_length",
    "url_special_chars"
]

print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

scaler=StandardScaler()
encoder=OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
preprocessor= ColumnTransformer(
    transformers=[
        ("categotical",encoder,categorical_columns),
        ("numerical",scaler,numerical_columns)
    ],
    remainder="passthrough"
)
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)
print("Processed X_train shape:", x_train_processed.shape)
print("Processed X_test shape:", x_test_processed.shape)
print(y_train.value_counts())
print(y_train.value_counts(normalize=True)*100)

smote=SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(
    x_train_processed,
    y_train
)
smote.fit_resample(
    x_train_processed,
    y_train
)
print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_balanced.value_counts())

print("\nBalanced x_train shape:", x_train_balanced.shape)
print("Balanced y_train shape:", y_train_balanced.shape)

print(" FINAL PREPROCESSING CHECK ")

# 1. Check shapes
print("\nShapes:")
print("x_train_balanced:", x_train_balanced.shape)
print("y_train_balanced:", y_train_balanced.shape)
print("X_test_processed:", x_test_processed.shape)
print("y_test:", y_test.shape)

# 2. Check missing values
print("\nMissing values:")
print("X_train_balanced:", pd.DataFrame(x_train_balanced).isna().sum().sum())
print("X_test_processed:", pd.DataFrame(x_test_processed).isna().sum().sum())

# 3. Check class distribution
print("\nTraining class distribution:")
print(y_train_balanced.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())

# 4. Check data types
print("\nTraining data type:")
print(type(x_train_balanced))

print("\nTest data type:")
print(type(x_test_processed))