import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#https://censusindia.gov.in/census.website/data/census-tables
# Load the data from the specified file path
df = pd.read_csv(r'D:\Desktop\PYTHONDATASET.csv')
print(df.head());
#print(df.desicribe());
# --- Data Cleaning ---
# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.rstrip(".")
print("Cleaned column names:", df.columns.tolist())

# Strip whitespace from Name column
df["Name"] = df["Name"].str.strip()

# Convert numeric columns
comma_cols = ["No_of_villages_Inhabited", "No_of_villages_Uninhabited", "Number_of_households", 
              "Population_Persons", "Population_Males", "Population_Females"]
# for col in comma_cols:
#     df[col] = df[col].astype(str).str.replace(",", "").astype(int)

# Convert Area and Population_per_sq_km to float, handling potential non-numeric values
df["Area"] = pd.to_numeric(df["Area"], errors='coerce')  # Converts invalid values to NaN
df["Population_per_sq_km"] = pd.to_numeric(df["Population_per_sq_km"], errors='coerce')

# Check for missing values in the full dataset
print("Missing values in full dataset:\n", df.isnull().sum())

# Consistency check for India
india_total = df[(df["Region"] == "INDIA") & (df["Area_type"] == "Total")]["Population_Persons"].iloc[0]
india_rural = df[(df["Region"] == "INDIA") & (df["Area_type"] == "Rural")]["Population_Persons"].iloc[0]
india_urban = df[(df["Region"] == "INDIA") & (df["Area_type"] == "Urban")]["Population_Persons"].iloc[0]
print(f"Consistency check: Total={india_total}, Rural+Urban={india_rural + india_urban}")

# --- Visualizations ---

# 1. Population Distribution Across States
states_total = df[(df["Region"] == "STATE") & (df["Area_type"] == "Total")]
states_total = states_total.sort_values("Population_Persons", ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(states_total["Name"], states_total["Population_Persons"])
plt.xlabel("Population")
plt.ylabel("State")
plt.title("Population Distribution Across States")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# 2. Urban vs Rural Population for Each State
states_rural_urban = df[(df["Region"] == "STATE") & (df["Area_type"].isin(["Rural", "Urban"]))]
pivot = states_rural_urban.pivot(index="Name", columns="Area_type", values="Population_Persons")
pivot.plot(kind="bar", stacked=True, figsize=(12, 8))
plt.xlabel("State")
plt.ylabel("Population")
plt.title("Urban vs Rural Population for Each State")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 3. Population Density Across Districts (Top 10)
districts_total = df[(df["Region"] == "DISTRICT") & (df["Area_type"] == "Total")]
top10_density = districts_total.nlargest(10, "Population_per_sq_km")
plt.figure(figsize=(10, 6))
plt.bar(top10_density["Name"], top10_density["Population_per_sq_km"])
plt.xlabel("District")
plt.ylabel("Population Density (per sq km)")
plt.title("Top 10 Districts by Population Density")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 4. Gender Ratio Across States
states_total = df[(df["Region"] == "STATE") & (df["Area_type"] == "Total")].copy()
states_total["Gender_Ratio"] = (states_total["Population_Females"] / states_total["Population_Males"]) * 1000
plt.figure(figsize=(12, 8))
plt.bar(states_total["Name"], states_total["Gender_Ratio"])
plt.axhline(y=1000, color="r", linestyle="--", label="Equal Ratio")
plt.xlabel("State")
plt.ylabel("Gender Ratio (Females per 1000 Males)")
plt.title("Gender Ratio Across States")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# 5. Relationship Between Households and Population for Sub-Districts
sub_districts = df[df["Region"] == "SUB-DISTRICT"]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sub_districts, x="Number_of_households", y="Population_Persons", hue="Area_type")
plt.xlabel("Number_of_households")
plt.ylabel("Population")
plt.title("Households vs Population in Sub-Districts")
plt.tight_layout()
plt.show()


# 6. Pair Plot of Key Numerical Variables for Sub-Districts
sub_districts_subset = sub_districts[["Population_Persons", "Number_of_households", "Area", "Population_per_sq_km"]]
plt.figure(figsize=(10, 10))
sns.pairplot(sub_districts_subset)
plt.suptitle("Pair Plot of Population, Households, Area, and Density in Sub-Districts", y=1.02)
plt.tight_layout()
plt.show()


# 7. Proportion of Urban vs Rural Population Across India
india_rural_urban = df[(df["Region"] == "INDIA") & (df["Area_type"].isin(["Rural", "Urban"]))][["Area_type", "Population_Persons"]]
plt.figure(figsize=(8, 8))
plt.pie(india_rural_urban["Population_Persons"], labels=india_rural_urban["Area_type"], autopct="%1.1f%%", startangle=90)
plt.title("Urban vs Rural Population in India")
plt.axis("equal")
plt.tight_layout()
plt.show()


# 8. Linear Regression Model for Population Prediction
# Prepare data for regression (sub-districts, Total area type)
sub_districts_total = sub_districts[sub_districts["Area_type"] == "Total"]
X = sub_districts_total[["Number_of_households", "Area", "Population_per_sq_km"]]
y = sub_districts_total["Population_Persons"]

# Check for NaN values in the regression data
print("Missing values in regression data (X):\n", X.isnull().sum())
print("Missing values in regression target (y):\n", y.isnull().sum())

# Drop rows with NaN values
X = X.dropna()
y = y.loc[X.index]  # Align y with X after dropping NaNs
print("Shape of X after dropping NaNs:", X.shape)
print("Shape of y after dropping NaNs:", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Regression Model R² Score: {r2:.4f}")

# Visualize actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")
plt.title(f"Actual vs Predicted Population (R² = {r2:.4f})")
plt.tight_layout()
plt.show()
