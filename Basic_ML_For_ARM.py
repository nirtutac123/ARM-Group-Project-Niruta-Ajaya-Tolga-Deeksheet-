import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

print("Step 1: Loading and preparing data...")

# Load your cleaned data
df = pd.read_csv('EDA_SCMS_Delivery_History_CLEANED.csv', encoding='latin-1')

# Check the columns available
print(f"Dataset shape: {df.shape}")
print(f"Columns available: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(3))

print("\nStep 2: Checking date columns and calculating delivery delay...")

# Check date columns
date_columns = [col for col in df.columns if 'Date' in col or 'date' in col]
print(f"Date columns found: {date_columns}")

# Check if we have scheduled vs actual delivery dates
print("\nDate column samples:")
for date_col in ['Scheduled_Delivery_Date', 'Delivered_to_Client_Date']:
    if date_col in df.columns:
        print(f"{date_col}: {df[date_col].head(3).tolist()}")
    else:
        print(f"{date_col}: NOT FOUND")

# Calculate delivery delay if we have the dates
if 'Scheduled_Delivery_Date' in df.columns and 'Delivered_to_Client_Date' in df.columns:
    print("\nCalculating delivery delay...")
    
    # Convert to datetime
    df['Scheduled_Delivery_Date'] = pd.to_datetime(df['Scheduled_Delivery_Date'], errors='coerce')
    df['Delivered_to_Client_Date'] = pd.to_datetime(df['Delivered_to_Client_Date'], errors='coerce')
    
    # Calculate delay in days
    df['Delivery_Delay_Days'] = (df['Delivered_to_Client_Date'] - df['Scheduled_Delivery_Date']).dt.days
    
    # Check results
    print(f"Delivery delay statistics:")
    print(f"Mean delay: {df['Delivery_Delay_Days'].mean():.2f} days")
    print(f"Std delay: {df['Delivery_Delay_Days'].std():.2f} days")
    print(f"Min delay: {df['Delivery_Delay_Days'].min()} days")
    print(f"Max delay: {df['Delivery_Delay_Days'].max()} days")
    
    # Check for missing values
    print(f"Missing values in delay: {df['Delivery_Delay_Days'].isna().sum()} / {len(df)}")
else:
    print("\n Cannot calculate delivery delay - missing date columns")
    
# Show categorical column distributions
print("\n\nStep 2.1: Checking key categorical columns...")
for col in ['Country', 'Product_Group', 'Vendor', 'Shipment_Mode']:
    if col in df.columns:
        print(f"\n{col} unique values ({df[col].nunique()} total):")
        print(df[col].value_counts().head(10))
    else:
        print(f"\n{col}: NOT FOUND")

print("\n Step 2 complete. ")

print("\nStep 3: Creating simplified features for ML model...")

# 1. Handle missing delivery delays
if 'Delivery_Delay_Days' not in df.columns:
    print("❌ ERROR: Delivery_Delay_Days column not found. Cannot proceed.")
    exit()

# Remove rows with missing delay values
initial_count = len(df)
df_clean = df.dropna(subset=['Delivery_Delay_Days']).copy()
print(f"Removed {initial_count - len(df_clean)} rows with missing delivery delay")
print(f"Remaining rows: {len(df_clean)}")

# 2. Create month and quarter from available date
date_col_to_use = None
for date_col in ['PO_Sent_to_Vendor_Date', 'Scheduled_Delivery_Date', 'Delivered_to_Client_Date']:
    if date_col in df_clean.columns:
        date_col_to_use = date_col
        break

if date_col_to_use:
    df_clean[date_col_to_use] = pd.to_datetime(df_clean[date_col_to_use], errors='coerce')
    df_clean['month'] = df_clean[date_col_to_use].dt.month.fillna(7)  # July as default
    df_clean['quarter'] = ((df_clean['month'] - 1) // 3) + 1
    print(f"Using '{date_col_to_use}' for month/quarter extraction")
else:
    print("⚠️ No date column found for month extraction, using defaults")
    df_clean['month'] = 7
    df_clean['quarter'] = 3

# 3. Prepare simplified features
features_data = []

# Numerical features (use defaults if missing)
features_data.append(('weight', df_clean['Weight_(Kilograms)'].fillna(10.0).values))
features_data.append(('quantity', df_clean['Line_Item_Quantity'].fillna(100).values))
features_data.append(('freight_cost', df_clean['Freight_Cost_(USD)'].fillna(1000).values))

# Temporal features
features_data.append(('month', df_clean['month'].values))
features_data.append(('quarter', df_clean['quarter'].values))

# Categorical features will be encoded
country_encoder = LabelEncoder()
product_encoder = LabelEncoder()

# Encode countries
df_clean['country_encoded'] = country_encoder.fit_transform(df_clean['Country'].fillna('Unknown'))
features_data.append(('country_encoded', df_clean['country_encoded'].values))

# Encode product groups  
df_clean['product_encoded'] = product_encoder.fit_transform(df_clean['Product_Group'].fillna('Unknown'))
features_data.append(('product_encoded', df_clean['product_encoded'].values))

# Create final features DataFrame
feature_names = [name for name, _ in features_data]
X = pd.DataFrame({name: values for name, values in features_data}, columns=feature_names)

# Target variable
y = df_clean['Delivery_Delay_Days'].values

print(f"\n✅ Simplified features created:")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {feature_names}")
print(f"\nSample features (first 3 rows):")
print(X.head(3))
print(f"\nCorresponding delays: {y[:3]}")

# Save mappings for later use
mappings = {
    'country_mapping': dict(zip(country_encoder.classes_, range(len(country_encoder.classes_)))),
    'product_mapping': dict(zip(product_encoder.classes_, range(len(product_encoder.classes_)))),
    'feature_names': feature_names
}

print(f"\n📊 Country mapping created ({len(mappings['country_mapping'])} countries)")
print(f"📊 Product mapping created ({len(mappings['product_mapping'])} product groups)")

print("\nStep 4 (Fixed): Cleaning weight data and training model...")

# First, fix the weight column - convert non-numeric to NaN then fill
print(f"Unique values in Weight_(Kilograms): {df_clean['Weight_(Kilograms)'].unique()[:20]}")

# Convert weight column properly
def clean_weight(value):
    try:
        return float(value)
    except:
        return np.nan

df_clean['Weight_(Kilograms)_clean'] = df_clean['Weight_(Kilograms)'].apply(clean_weight)

# Check weight statistics
print(f"\nWeight statistics after cleaning:")
print(f"Missing values: {df_clean['Weight_(Kilograms)_clean'].isna().sum()}")
print(f"Mean weight: {df_clean['Weight_(Kilograms)_clean'].mean():.2f}")
print(f"Median weight: {df_clean['Weight_(Kilograms)_clean'].median():.2f}")

# Update features with cleaned weight
X['weight'] = df_clean['Weight_(Kilograms)_clean'].fillna(10.0).values

# Verify all features are numeric
print(f"\nFeature data types:")
print(X.dtypes)

print(f"\nSample features after cleaning:")
print(X.head(3))
print(f"\nCorresponding delays: {y[:3]}")

print("\nStep 5 (Fixed): Cleaning freight_cost data...")

# Check what's in freight_cost column
print(f"Unique values in freight_cost: {df_clean['Freight_Cost_(USD)'].unique()[:10]}")

# Clean freight_cost column
def clean_freight(value):
    try:
        return float(value)
    except:
        return np.nan

df_clean['Freight_Cost_(USD)_clean'] = df_clean['Freight_Cost_(USD)'].apply(clean_freight)

print(f"\nFreight cost statistics after cleaning:")
print(f"Missing values: {df_clean['Freight_Cost_(USD)_clean'].isna().sum()}")
print(f"Mean freight: ${df_clean['Freight_Cost_(USD)_clean'].mean():.2f}")
print(f"Median freight: ${df_clean['Freight_Cost_(USD)_clean'].median():.2f}")

# Update the features DataFrame with cleaned freight cost
X['freight_cost'] = df_clean['Freight_Cost_(USD)_clean'].fillna(1000.0).values

# Verify all features are numeric
print(f"\nFeature data types after cleaning:")
print(X.dtypes)

print(f"\nSample features after all cleaning:")
print(X.head(5))

# Check for any remaining non-numeric values
print(f"\nChecking for non-numeric values in each column:")
for col in X.columns:
    non_numeric = X[col].apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating))).sum()
    print(f"{col}: {non_numeric} non-numeric values")

# Check feature ranges
print(f"\nFeature ranges:")
for col in X.columns:
    print(f"{col}: min={X[col].min():.2f}, max={X[col].max():.2f}, mean={X[col].mean():.2f}")

print("\nStep 6: Training Random Forest model with cleaned data...")

# Split the cleaned data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✅ Data split successfully:")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")

# Initialize and train the model
print("\n🔧 Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,      # 100 trees
    max_depth=10,          # Limit tree depth to prevent overfitting
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples in a leaf
    random_state=42,       # For reproducibility
    n_jobs=-1,             # Use all CPU cores
    verbose=1              # Show progress
)

model.fit(X_train, y_train)
print("✅ Model training completed!")

# Evaluate the model
print("\n📊 Model Evaluation:")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"R² Score (Training): {train_score:.4f}")
print(f"R² Score (Test): {test_score:.4f}")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📈 Prediction Errors:")
print(f"Mean Absolute Error (MAE): {mae:.2f} days")
print(f"Mean Squared Error (MSE): {mse:.2f} days²")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} days")

# Check actual vs predicted
print(f"\n📋 Sample Predictions (first 5 test samples):")
for i in range(min(5, len(y_test))):
    print(f"  Actual: {y_test[i]:6.1f} days | Predicted: {y_pred[i]:6.1f} days | Diff: {abs(y_test[i]-y_pred[i]):.1f} days")


print("\nStep 7: Saving model and creating integration files...")

# Save the trained model
joblib.dump(model, 'simplified_delay_model.pkl')
print("✅ Saved: simplified_delay_model.pkl")

# Save the mappings (country, product encodings)
mappings = {
    'country_mapping': dict(zip(country_encoder.classes_, range(len(country_encoder.classes_)))),
    'product_mapping': dict(zip(product_encoder.classes_, range(len(product_encoder.classes_)))),
    'feature_names': list(X.columns),
    'feature_statistics': {
        'weight_mean': float(df_clean['Weight_(Kilograms)_clean'].mean()),
        'weight_median': float(df_clean['Weight_(Kilograms)_clean'].median()),
        'quantity_mean': float(df_clean['Line_Item_Quantity'].mean()),
        'freight_mean': float(df_clean['Freight_Cost_(USD)_clean'].mean()),
    }
}

joblib.dump(mappings, 'simplified_model_mappings.pkl')
print("✅ Saved: simplified_model_mappings.pkl")

# Create a simple scaler (for compatibility with tariff simulator)
class DummyScaler:
    def fit(self, X): return self
    def transform(self, X): return X
    def fit_transform(self, X): return X

scaler = DummyScaler()
joblib.dump(scaler, 'simplified_scaler.pkl')
print("✅ Saved: simplified_scaler.pkl")

# Create a test function to verify model works with tariff simulator format
print("\n🔍 Testing model with tariff simulator format...")

def predict_delay_for_simulator(shipment_data):
    """
    Predict delay for a shipment in tariff simulator format
    shipment_data: dict with keys: Country, Product_Group, optional: Weight_(Kilograms), Line_Item_Quantity, Freight_Cost_(USD)
    """
    # Get mappings
    country_code = mappings['country_mapping'].get(shipment_data.get('Country', 'Unknown'), 0)
    product_code = mappings['product_mapping'].get(shipment_data.get('Product_Group', 'Unknown'), 0)
    
    # Use provided values or defaults from statistics
    weight = shipment_data.get('Weight_(Kilograms)', mappings['feature_statistics']['weight_median'])
    quantity = shipment_data.get('Line_Item_Quantity', mappings['feature_statistics']['quantity_mean'])
    freight = shipment_data.get('Freight_Cost_(USD)', mappings['feature_statistics']['freight_mean'])
    
    # Current month/quarter (for recency)
    current_month = datetime.now().month
    current_quarter = (current_month - 1) // 3 + 1
    
    # Create feature array in correct order
    features = np.array([[
        float(weight),
        float(quantity),
        float(freight),
        float(current_month),
        float(current_quarter),
        float(country_code),
        float(product_code)
    ]])
    
    # Make prediction
    delay = model.predict(features)[0]
    return delay

# Test with sample shipments
print("\n📊 Test predictions for tariff simulator:")
test_shipments = [
    {'Country': 'South Africa', 'Product_Group': 'ARV'},
    {'Country': 'India', 'Product_Group': 'HRDT'},
    {'Country': 'China', 'Product_Group': 'ACT'},
    {'Country': 'Vietnam', 'Product_Group': 'ANTM'},
]

for shipment in test_shipments:
    delay = predict_delay_for_simulator(shipment)
    print(f"  {shipment['Country']:15} - {shipment['Product_Group']:6}: {delay:6.1f} days")

# Create a README file for integration
print("\n📝 Creating integration instructions...")
integration_info = """
TARIFF SIMULATOR INTEGRATION GUIDE
==================================

Files created:
1. simplified_delay_model.pkl - Your trained ML model
2. simplified_model_mappings.pkl - Country/product encodings
3. simplified_scaler.pkl - Dummy scaler for compatibility

To integrate with tariff_simulator.py:

1. Update load_resources() function:
   - Change to load simplified_delay_model.pkl
   - Use feature_names from mappings

2. Update prepare_features() method:
   - Use predict_delay_for_simulator() logic
   - Map Country/Product_Group to encoded values

3. Test integration with 25% tariff scenario

Expected baseline delays by country (approximate):
- India: 8-12 days
- China: 7-11 days  
- South Africa: 5-9 days
- Vietnam: 6-10 days
"""

print(integration_info)
print("✅ Step 7 complete! Model is ready for tariff simulator integration.")
print("\nNext: Update tariff_simulator.py to use your new simplified ML model.")