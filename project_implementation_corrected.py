# ==========================================
# STEP 1: LOAD, UNDERSTAND AND CLEAN DATA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, ConfusionMatrixDisplay



from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load dataset
df = pd.read_csv("export.csv", low_memory=False)

# Display first five rows
print("First Five Rows:")
print(df.head())

# Display last five rows
print("\nLast Five Rows:")
print(df.tail())

# Check number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDataset Information:")
df.info()

# Statistical summary of numerical columns
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Check unique values in important columns
print("\nUnique Years:")
print(df["YearStart"].unique())

print("\nUnique Locations:")
print(df["LocationDesc"].nunique())

print("\nUnique Topics:")
print(df["Topic"].nunique())

# ------------------------------------------
# DATA CLEANING
# ------------------------------------------

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert target and numeric measurement columns
numeric_columns = [
    "Data_Value",
    "Data_Value_Alt",
    "Low_Confidence_Limit",
    "High_Confidence_Limit",
    "Sample_Size"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows where the target value is missing
df = df.dropna(subset=["Data_Value"])

# Remove rows containing infinite target values
df = df[np.isfinite(df["Data_Value"])]

# Reset index
df = df.reset_index(drop=True)

# Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

print("\nShape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())







# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Select relevant input features
features = [
    "YearStart",
    "YearEnd",
    "LocationAbbr",
    "LocationDesc",
    "Datasource",
    "Class",
    "Topic",
    "Question",
    "Data_Value_Type",
    "LocationID",
    "StratificationCategory1",
    "Stratification1"
]

# Keep only features that exist in the dataset
features = [col for col in features if col in df.columns]

# Create input variables
X = df[features].copy()

# Create target variable
y = df["Data_Value"].copy()

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Identify numerical columns
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Identify categorical columns without the warning
categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]
print("Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

# Numerical preprocessing
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the fitted preprocessing
X_test_processed = preprocessor.transform(X_test)

# Display results
print("\nOriginal Training Data Shape:", X_train.shape)
print("Original Testing Data Shape:", X_test.shape)

print("\nProcessed Training Data Shape:", X_train_processed.shape)
print("Processed Testing Data Shape:", X_test_processed.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nData Preprocessing Completed Successfully!")

# ==========================================
# VISUALIZATION 1: DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Data_Value",
    bins=40,
    kde=True
)

plt.title("Distribution of Data Values")
plt.xlabel("Data Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ==========================================
# VISUALIZATION 2: YEARLY TREND
# ==========================================

yearly_data = (
    df.groupby("YearStart", as_index=False)["Data_Value"]
    .mean()
    .sort_values("YearStart")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=yearly_data,
    x="YearStart",
    y="Data_Value",
    marker="o"
)

plt.title("Average Data Value by Year")
plt.xlabel("Year")
plt.ylabel("Average Data Value")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# ==========================================
# VISUALIZATION 3: LOCATION COMPARISON
# ==========================================

location_data = (
    df.groupby("LocationDesc")["Data_Value"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=location_data,
    x="Data_Value",
    y="LocationDesc"
)

plt.title("Top 10 Locations by Average Data Value")
plt.xlabel("Average Data Value")
plt.ylabel("Location")

plt.tight_layout()
plt.show()


# ==========================================
# VISUALIZATION 4: TOPIC COMPARISON
# ==========================================

topic_data = (
    df.groupby("Topic")["Data_Value"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=topic_data,
    x="Data_Value",
    y="Topic"
)

plt.title("Average Data Value by Topic")
plt.xlabel("Average Data Value")
plt.ylabel("Topic")

plt.tight_layout()
plt.show()



# ==========================================
# VISUALIZATION 5: CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Variables")

plt.tight_layout()
plt.show()





# STEP 4: SIMPLE LINEAR REGRESSION
# ============================================================

print("\n======================================")
print("1. SIMPLE LINEAR REGRESSION")
print("======================================")

# X contains one feature.
# y contains the numerical target.

X = df[["YearStart"]]
y = df["Data_Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

simple_model = LinearRegression()

simple_model.fit(X_train, y_train)

simple_predictions = simple_model.predict(X_test)

simple_mae = mean_absolute_error(
    y_test, simple_predictions
)

simple_mse = mean_squared_error(
    y_test, simple_predictions
)

simple_rmse = np.sqrt(simple_mse)

simple_r2 = r2_score(
    y_test, simple_predictions
)

print("Feature: YearStart")
print("Target: Data_Value")
print("MAE:", round(simple_mae, 4))
print("MSE:", round(simple_mse, 4))
print("RMSE:", round(simple_rmse, 4))
print("R2 Score:", round(simple_r2, 4))

print("Intercept:", simple_model.intercept_)
print("Coefficient:", simple_model.coef_[0])


# Plot actual and predicted values

simple_plot = pd.DataFrame({
    "Year": X_test["YearStart"].values,
    "Actual": y_test.values,
    "Predicted": simple_predictions
})

simple_plot = simple_plot.sort_values("Year")

plt.figure(figsize=(9, 5))

plt.scatter(
    simple_plot["Year"],
    simple_plot["Actual"],
    alpha=0.5,
    label="Actual Values"
)

plt.plot(
    simple_plot["Year"],
    simple_plot["Predicted"],
    linewidth=2,
    label="Predicted Values"
)

plt.title("Simple Linear Regression")
plt.xlabel("Year Start")
plt.ylabel("Data Value")
plt.legend()
plt.tight_layout()
plt.show()


# Example prediction for a new year

new_year = pd.DataFrame({
    "YearStart": [2024]
})

prediction = simple_model.predict(new_year)

print(
    "\nPredicted Data Value for 2024:",
    round(prediction[0], 4)
)


# ============================================================
# STEP 5: MULTIPLE LINEAR REGRESSION
# ============================================================

print("\n======================================")
print("2. MULTIPLE LINEAR REGRESSION")
print("======================================")

# Select numeric features.
# Remove target, IDs, and columns that directly duplicate
# or reveal the target.

excluded_columns = [
    "Data_Value",
    "Data_Value_Alt",
    "Low_Confidence_Limit",
    "High_Confidence_Limit",
    "LocationID",
    "StratificationID1",
    "StratificationID2",
    "ClassID",
    "TopicID",
    "QuestionID",
    "DataValueTypeID"
]

numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

features = [
    col for col in numeric_columns
    if col not in excluded_columns
]

if "YearStart" not in features:
    features.append("YearStart")

features = list(dict.fromkeys(features))

X = df[features]
y = df["Data_Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

imputer = SimpleImputer(strategy="median")

X_train_clean = imputer.fit_transform(X_train)
X_test_clean = imputer.transform(X_test)

multiple_model = LinearRegression()

multiple_model.fit(
    X_train_clean,
    y_train
)

multiple_predictions = multiple_model.predict(
    X_test_clean
)

multiple_mae = mean_absolute_error(
    y_test, multiple_predictions
)

multiple_mse = mean_squared_error(
    y_test, multiple_predictions
)

multiple_rmse = np.sqrt(multiple_mse)

multiple_r2 = r2_score(
    y_test, multiple_predictions
)

print("Features:", features)
print("Target: Data_Value")
print("MAE:", round(multiple_mae, 4))
print("MSE:", round(multiple_mse, 4))
print("RMSE:", round(multiple_rmse, 4))
print("R2 Score:", round(multiple_r2, 4))


# Actual versus predicted graph

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    multiple_predictions,
    alpha=0.5
)

plt.title("Multiple Linear Regression")
plt.xlabel("Actual Data Value")
plt.ylabel("Predicted Data Value")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 6: PREPARE DATA FOR CLASSIFICATION
# ============================================================

print("\n======================================")
print("PREPARING CLASSIFICATION DATA")
print("======================================")

# Create High and Low categories using the median.
# This is a project-defined classification target.

median_value = df["Data_Value"].median()

df["Value_Category"] = np.where(
    df["Data_Value"] >= median_value,
    "High",
    "Low"
)

print("Median threshold:", median_value)

print("\nCategory counts:")
print(df["Value_Category"].value_counts())


# Select numeric features.
# Data_Value must not be included because Value_Category
# was created from Data_Value.

classification_excluded = excluded_columns + [
    "Value_Category",
    "YearEnd"
]

classification_features = [
    col
    for col in df.select_dtypes(include=["number"]).columns
    if col not in classification_excluded
]

if not classification_features:
    classification_features = ["YearStart"]

X = df[classification_features]
y = df["Value_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nClassification features:")
print(classification_features)


# ============================================================
# PREPARE MODEL VARIABLES FOR THE RATIO COMPARISON
# ============================================================

# Simple Linear Regression: one input feature
X_simple = df[["YearStart"]].copy()
y_simple = df["Data_Value"].copy()

# Multiple Linear Regression: numeric input features only.
# Exclude the target, target-derived columns, and identifier columns.
excluded_columns = [
    "Data_Value",
    "Data_Value_Alt",
    "Low_Confidence_Limit",
    "High_Confidence_Limit",
    "LocationID",
    "StratificationID1",
    "StratificationID2",
    "ClassID",
    "TopicID",
    "QuestionID",
    "DataValueTypeID"
]

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
multiple_features = [
    col for col in numeric_columns
    if col not in excluded_columns
]

if "YearStart" not in multiple_features:
    multiple_features.append("YearStart")

X_multiple = df[multiple_features].copy()
y_reg = df["Data_Value"].copy()

# Classification target: high or low Data_Value, based on the median.
# Do not include Data_Value or columns derived from it as predictors.
median_value = df["Data_Value"].median()
df["Value_Category"] = np.where(
    df["Data_Value"] >= median_value,
    "High",
    "Low"
)

classification_excluded = excluded_columns + [
    "Value_Category",
    "YearEnd"
]

classification_features = [
    col for col in df.select_dtypes(include=["number"]).columns
    if col not in classification_excluded
]

if not classification_features:
    classification_features = ["YearStart"]

X_class = df[classification_features].copy()
y_class = df["Value_Category"].copy()

# Train/test split ratios: test_size is the proportion reserved for testing.
ratios = {
    "70:30": 0.30,
    "80:20": 0.20,
    "75:25": 0.25
}

results = []


# ============================================================
# STEP 3: RUN ALL FIVE ALGORITHMS AT EACH RATIO
# ============================================================

for ratio_name, test_size in ratios.items():

    print("\n" + "=" * 65)
    print("TRAIN/TEST RATIO:", ratio_name)
    print("=" * 65)

    # --------------------------------------------------------
    # 1. SIMPLE LINEAR REGRESSION
    # --------------------------------------------------------

   
    X_train, X_test, y_train, y_test = train_test_split(
        X_simple,
        y_simple,
        test_size=test_size,
        random_state=42
    )

    simple_model = LinearRegression()
    simple_model.fit(X_train, y_train)

    predictions = simple_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n1. SIMPLE LINEAR REGRESSION")
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    results.append({
        "Ratio": ratio_name,
        "Algorithm": "Simple Linear Regression",
        "Metric": "R2",
        "Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    # --------------------------------------------------------
    # 2. MULTIPLE LINEAR REGRESSION
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_multiple,
        y_reg,
        test_size=test_size,
        random_state=42
    )

    multiple_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])

    multiple_model.fit(X_train, y_train)
    predictions = multiple_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n2. MULTIPLE LINEAR REGRESSION")
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    results.append({
        "Ratio": ratio_name,
        "Algorithm": "Multiple Linear Regression",
        "Metric": "R2",
        "Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    # Use the same classification split for all three
    # classification algorithms at this ratio.

    X_train, X_test, y_train, y_test = train_test_split(
        X_class,
        y_class,
        test_size=test_size,
        random_state=42,
        stratify=y_class
    )

    # --------------------------------------------------------
    # 3. LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    logistic_model.fit(X_train, y_train)
    predictions = logistic_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n3. LOGISTIC REGRESSION")
    print("Accuracy:", round(accuracy, 4))

    results.append({
        "Ratio": ratio_name,
        "Algorithm": "Logistic Regression",
        "Metric": "Accuracy",
        "Score": accuracy,
        "MAE": np.nan,
        "MSE": np.nan,
        "RMSE": np.nan
    })

    # --------------------------------------------------------
    # 4. DECISION TREE
    # --------------------------------------------------------

    tree_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ])

    tree_model.fit(X_train, y_train)
    predictions = tree_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n4. DECISION TREE")
    print("Accuracy:", round(accuracy, 4))

    results.append({
        "Ratio": ratio_name,
        "Algorithm": "Decision Tree",
        "Metric": "Accuracy",
        "Score": accuracy,
        "MAE": np.nan,
        "MSE": np.nan,
        "RMSE": np.nan
    })

    # --------------------------------------------------------
    # 5. K-NEAREST NEIGHBORS (KNN)
    # --------------------------------------------------------

    knn_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])

    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n5. K-NEAREST NEIGHBORS (KNN)")
    print("Accuracy:", round(accuracy, 4))

    results.append({
        "Ratio": ratio_name,
        "Algorithm": "KNN",
        "Metric": "Accuracy",
        "Score": accuracy,
        "MAE": np.nan,
        "MSE": np.nan,
        "RMSE": np.nan
    })


# ============================================================
# STEP 4: FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(results)

print("\n" + "=" * 75)
print("FINAL COMPARISON OF ALL 15 MODEL EVALUATIONS")
print("=" * 75)

print(comparison.round(4).to_string(index=False))


# ============================================================
# STEP 5: SAVE RESULTS TO CSV
# ============================================================

comparison.to_csv(
    "algorithm_ratio_comparison.csv",
    index=False
)

print("\nComparison saved to algorithm_ratio_comparison.csv")


# ============================================================
# STEP 6: COMPARISON GRAPHS
# ============================================================

for ratio_name in ratios:

    data = comparison[
        comparison["Ratio"] == ratio_name
    ]

    plt.figure(figsize=(10, 5))

    plt.bar(
        data["Algorithm"],
        data["Score"]
    )

    plt.title(
        "Algorithm Comparison - " + ratio_name
    )

    plt.xlabel("Algorithm")
    plt.ylabel("R2 Score / Accuracy")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()



print("\nPROJECT COMPLETED SUCCESsfully")






