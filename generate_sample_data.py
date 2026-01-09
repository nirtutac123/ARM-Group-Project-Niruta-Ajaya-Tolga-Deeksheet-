import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating sample supply chain data for demonstration...")

# Set random seed for reproducibility
np.random.seed(42)

# Define sample data parameters
n_samples = 5000

# Sample categories
countries = ['Nigeria', 'South Africa', 'Uganda', 'Zimbabwe', 'Côte d\'Ivoire', 'Zambia', 'Haiti', 'Vietnam']
product_groups = ['ARV', 'HRDT', 'ANTIMALARIA', 'HIV test', 'ANTIMALARIALS', 'Emergency', 'Safety']
vendors = ['Aurobindo Unit III, India', 'Hetero Unit III India', 'Cipla Ltd.', 'Matrix Laboratories', 'Ranbaxy Fine Chemicals Limited']
managed_by = ['PMO - US', 'PMO - SA', 'PMO - NG']
fulfill_via = ['Direct Drop', 'From RDC']
shipment_modes = ['Air', 'Ocean', 'Air Charter', 'Truck']
manufacturing_sites = ['Aurobindo Unit III', 'Hetero Unit III', 'Cipla', 'Matrix', 'Ranbaxy']

# Generate base dataframe
print(f"Creating {n_samples} sample records...")

data = {
    'ID': range(1, n_samples + 1),
    'Project_Code': [f'PC-{np.random.randint(10000, 99999)}' for _ in range(n_samples)],
    'PQ': np.random.choice(['PQ', 'PQS', 'Non-PQ'], n_samples, p=[0.7, 0.2, 0.1]),
    'PO_SO': [f'PO{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
    'ASN_DN': [f'ASN{np.random.randint(10000, 99999)}' for _ in range(n_samples)],
    'Country': np.random.choice(countries, n_samples),
    'Managed_By': np.random.choice(managed_by, n_samples),
    'Fulfill_Via': np.random.choice(fulfill_via, n_samples),
    'Vendor_INCO_Term': np.random.choice(['EXW', 'FCA', 'CIF', 'DDP'], n_samples),
    'Shipment_Mode': np.random.choice(shipment_modes, n_samples, p=[0.25, 0.45, 0.15, 0.15]),
    'Product_Group': np.random.choice(product_groups, n_samples),
    'Sub_Classification': [f'Sub-{i % 20 + 1}' for i in range(n_samples)],
    'Vendor': np.random.choice(vendors, n_samples),
    'Item_Description': [f'Medicine {chr(65 + i % 26)} 100mg' for i in range(n_samples)],
    'Molecule_Test_Type': [f'Type-{i % 15 + 1}' for i in range(n_samples)],
    'Brand': [f'Brand-{chr(65 + i % 10)}' for i in range(n_samples)],
    'Dosage': np.random.choice(['100mg', '200mg', '50mg', '500mg'], n_samples),
    'Dosage_Form': np.random.choice(['Tablet', 'Capsule', 'Solution'], n_samples),
    'Unit_of_Measure_Per_Pack': np.random.choice([30, 60, 90, 100, 120], n_samples),
    'Line_Item_Quantity': np.random.randint(100, 10000, n_samples),
    'Line_Item_Value': np.random.uniform(1000, 100000, n_samples).round(2),
    'Pack_Price': np.random.uniform(5, 500, n_samples).round(2),
    'Unit_Price': np.random.uniform(0.1, 10, n_samples).round(2),
    'Manufacturing_Site': np.random.choice(manufacturing_sites, n_samples),
    'First_Line_Designation': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
}

# Generate dates
base_date = datetime(2015, 1, 1)

po_dates = [base_date + timedelta(days=int(np.random.uniform(0, 270))) for _ in range(n_samples)]
scheduled_delivery_dates = [po_date + timedelta(days=int(np.random.uniform(30, 90))) for po_date in po_dates]

# Create actual delivery dates with realistic delays
actual_delivery_dates = []
for sched_date in scheduled_delivery_dates:
    delay = np.random.choice(
        [-5, -3, -1, 0, 1, 3, 5, 7, 14, 21, 30, 45, 60],
        p=[0.05, 0.05, 0.1, 0.2, 0.15, 0.15, 0.1, 0.08, 0.05, 0.04, 0.02, 0.005, 0.005]
    )
    actual_delivery_dates.append(sched_date + timedelta(days=int(delay)))

delivery_recorded_dates = [act_date + timedelta(days=int(np.random.uniform(0, 5))) for act_date in actual_delivery_dates]

data['PO_Sent_to_Vendor_Date'] = po_dates
data['Scheduled_Delivery_Date'] = scheduled_delivery_dates
data['Delivered_to_Client_Date'] = actual_delivery_dates
data['Delivery_Recorded_Date'] = delivery_recorded_dates

# Calculate delivery delay
delivery_delays = [(actual - scheduled).days for actual, scheduled in zip(actual_delivery_dates, scheduled_delivery_dates)]
data['Delivery_Delay_Days'] = delivery_delays

# Add weight and freight cost
weights = []
for i in range(n_samples):
    base_weight = data['Line_Item_Quantity'][i] * data['Unit_of_Measure_Per_Pack'][i] * 0.01  # rough estimation
    weight = max(1, base_weight + np.random.normal(0, base_weight * 0.1))
    weights.append(round(weight, 2))

data['Weight_Kilograms'] = weights

# Freight cost based on weight and shipment mode
freight_costs = []
for i in range(n_samples):
    if data['Shipment_Mode'][i] == 'Air':
        rate = np.random.uniform(2, 5)
    elif data['Shipment_Mode'][i] == 'Air Charter':
        rate = np.random.uniform(4, 8)
    elif data['Shipment_Mode'][i] == 'Ocean':
        rate = np.random.uniform(0.5, 1.5)
    else:  # Truck
        rate = np.random.uniform(0.3, 1)
    
    cost = data['Weight_Kilograms'][i] * rate
    freight_costs.append(round(cost, 2))

data['Freight_Cost_USD'] = freight_costs

# Add line item insurance
data['Line_Item_Insurance_USD'] = (data['Line_Item_Value'] * 0.01).round(2)

# Create DataFrame
df = pd.DataFrame(data)

# Add some temporal features
df['Year'] = df['Delivered_to_Client_Date'].dt.year
df['Month'] = df['Delivered_to_Client_Date'].dt.month
df['Quarter'] = df['Delivered_to_Client_Date'].dt.quarter
df['Month_Name'] = df['Delivered_to_Client_Date'].dt.month_name()

# Add a few random nulls to be realistic (but keep most data clean)
null_percentage = 0.02  # 2% nulls
for col in ['Vendor', 'Manufacturing_Site', 'Brand']:
    null_mask = np.random.random(n_samples) < null_percentage
    df.loc[null_mask, col] = np.nan

print(f"Generated dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"\nSample statistics:")
print(f"  Date range: {df['Delivered_to_Client_Date'].min()} to {df['Delivered_to_Client_Date'].max()}")
print(f"  Average delivery delay: {df['Delivery_Delay_Days'].mean():.1f} days")
print(f"  Total shipment value: ${df['Line_Item_Value'].sum():,.2f}")
print(f"  Total freight cost: ${df['Freight_Cost_USD'].sum():,.2f}")

# Save to the expected location
output_path = 'EDA/EDA_SCMS_Delivery_History_CLEANED.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ Data saved successfully to: {output_path}")
print(f"File size: {len(df)} rows × {len(df.columns)} columns")

# Show preview
print("\nFirst few rows:")
print(df.head(3))
print("\nColumn list:")
print(df.columns.tolist())
