import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load the cleaned supply chain data"""
    # Try multiple file locations
    possible_paths = [
        'EDA/EDA_SCMS_Delivery_History_CLEANED.csv',
        'SCMS_Delivery_History_CLEANED.csv',
        'EDA_SCMS_Delivery_History_CLEANED.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path, encoding='latin-1')
            if len(df) > 0:  # Check if dataframe has data
                break
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
    
    if df is None or len(df) == 0:
        return None
    
    # Convert date columns
    date_columns = ['PO_Sent_to_Vendor_Date', 'Scheduled_Delivery_Date', 
                   'Delivered_to_Client_Date', 'Delivery_Recorded_Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate delivery delay if not present
    if 'Delivery_Delay_Days' not in df.columns:
        if 'Scheduled_Delivery_Date' in df.columns and 'Delivered_to_Client_Date' in df.columns:
            df['Delivery_Delay_Days'] = (df['Delivered_to_Client_Date'] - 
                                         df['Scheduled_Delivery_Date']).dt.days
    
    # Extract temporal features
    if 'Delivered_to_Client_Date' in df.columns:
        df['Year'] = df['Delivered_to_Client_Date'].dt.year
        df['Month'] = df['Delivered_to_Client_Date'].dt.month
        df['Quarter'] = df['Delivered_to_Client_Date'].dt.quarter
        df['Month_Name'] = df['Delivered_to_Client_Date'].dt.month_name()
    
    return df

# Load ML model
@st.cache_resource
def load_ml_model():
    """Load the trained ML model"""
    try:
        model = joblib.load('simplified_delay_model.pkl')
        mappings = joblib.load('simplified_model_mappings.pkl')
        return model, mappings
    except FileNotFoundError:
        return None, None

# Generate sample data if no data file exists
def generate_sample_data(n_samples=1000):
    """Generate sample supply chain data for demo purposes"""
    np.random.seed(42)
    
    countries = ['India', 'China', 'Vietnam', 'Germany', 'France', 'South Africa', 'USA', 'Kenya', 'Uganda']
    products = ['ARV', 'HRDT', 'ACT', 'ANTM', 'MRDT']
    shipment_modes = ['Air', 'Ocean', 'Truck', 'Air Charter']
    
    data = {
        'Country': np.random.choice(countries, n_samples),
        'Product_Group': np.random.choice(products, n_samples),
        'Shipment_Mode': np.random.choice(shipment_modes, n_samples),
        'Line_Item_Quantity': np.random.randint(10, 10000, n_samples),
        'Weight_(Kilograms)': np.random.uniform(1, 5000, n_samples),
        'Freight_Cost_(USD)': np.random.uniform(100, 50000, n_samples),
        'Line_Item_Value': np.random.uniform(500, 500000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate dates
    start_date = pd.Timestamp('2013-01-01')
    df['Scheduled_Delivery_Date'] = pd.to_datetime([start_date + pd.Timedelta(days=np.random.randint(0, 1000)) 
                                                     for _ in range(n_samples)])
    
    # Generate actual delivery with some delays
    delays = np.random.normal(5, 10, n_samples)  # mean 5 days, std 10 days
    df['Delivered_to_Client_Date'] = df['Scheduled_Delivery_Date'] + pd.to_timedelta(delays, unit='D')
    df['Delivery_Delay_Days'] = delays
    
    # Extract temporal features
    df['Year'] = df['Delivered_to_Client_Date'].dt.year
    df['Month'] = df['Delivered_to_Client_Date'].dt.month
    df['Quarter'] = df['Delivered_to_Client_Date'].dt.quarter
    df['Month_Name'] = df['Delivered_to_Client_Date'].dt.month_name()
    
    return df

# Main app
def main():
    # Title
    st.title("📦 Supply Chain Management Analytics Dashboard")
    st.markdown("### Advanced Research Methodologies - Group Project")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.warning("⚠️ Data file not found or is empty!")
        st.info("📁 **Expected file location:** `EDA/EDA_SCMS_Delivery_History_CLEANED.csv`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Option 1: Upload CSV File")
            uploaded_file = st.file_uploader("Upload your cleaned supply chain CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    
                    # Process the uploaded data
                    date_columns = ['PO_Sent_to_Vendor_Date', 'Scheduled_Delivery_Date', 
                                   'Delivered_to_Client_Date', 'Delivery_Recorded_Date']
                    for col in date_columns:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    if 'Delivery_Delay_Days' not in df.columns:
                        if 'Scheduled_Delivery_Date' in df.columns and 'Delivered_to_Client_Date' in df.columns:
                            df['Delivery_Delay_Days'] = (df['Delivered_to_Client_Date'] - 
                                                         df['Scheduled_Delivery_Date']).dt.days
                    
                    if 'Delivered_to_Client_Date' in df.columns:
                        df['Year'] = df['Delivered_to_Client_Date'].dt.year
                        df['Month'] = df['Delivered_to_Client_Date'].dt.month
                        df['Quarter'] = df['Delivered_to_Client_Date'].dt.quarter
                        df['Month_Name'] = df['Delivered_to_Client_Date'].dt.month_name()
                    
                    st.success(f"✅ File uploaded successfully! ({len(df)} records)")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    df = None
        
        with col2:
            st.markdown("### Option 2: Use Sample Data")
            if st.button("📊 Generate Sample Data", type="primary"):
                df = generate_sample_data(1000)
                st.success("✅ Sample data generated! (1000 records)")
                st.info("💡 This is demonstration data for testing the dashboard")
        
        if df is None:
            st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Analysis",
        ["📊 Overview", "🌍 Geographic Analysis", "📦 Product Analysis", 
         "⏱️ Delivery Performance", "💰 Cost Analysis", "🤖 ML Predictions",
         "🎯 Tariff Impact Simulator"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Dataset Info")
    st.sidebar.metric("Total Shipments", f"{len(df):,}")
    st.sidebar.metric("Countries", df['Country'].nunique() if 'Country' in df.columns else "N/A")
    st.sidebar.metric("Products", df['Product_Group'].nunique() if 'Product_Group' in df.columns else "N/A")
    
    # Page routing
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🌍 Geographic Analysis":
        show_geographic_analysis(df)
    elif page == "📦 Product Analysis":
        show_product_analysis(df)
    elif page == "⏱️ Delivery Performance":
        show_delivery_performance(df)
    elif page == "💰 Cost Analysis":
        show_cost_analysis(df)
    elif page == "🤖 ML Predictions":
        show_ml_predictions(df)
    elif page == "🎯 Tariff Impact Simulator":
        show_tariff_simulator(df)

def show_overview(df):
    """Display overview dashboard"""
    st.header("📊 Supply Chain Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_shipments = len(df)
        st.metric("Total Shipments", f"{total_shipments:,}")
    
    with col2:
        if 'Delivery_Delay_Days' in df.columns:
            avg_delay = df['Delivery_Delay_Days'].mean()
            st.metric("Avg Delay", f"{avg_delay:.1f} days")
        else:
            st.metric("Avg Delay", "N/A")
    
    with col3:
        if 'Line_Item_Value' in df.columns:
            total_value = df['Line_Item_Value'].sum()
            st.metric("Total Value", f"${total_value/1e6:.1f}M")
        else:
            st.metric("Total Value", "N/A")
    
    with col4:
        if 'Freight_Cost_(USD)' in df.columns:
            total_freight = df['Freight_Cost_(USD)'].sum()
            st.metric("Total Freight", f"${total_freight/1e6:.1f}M")
        else:
            st.metric("Total Freight", "N/A")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Shipments by country
        if 'Country' in df.columns:
            st.subheader("Top 10 Countries by Shipments")
            country_counts = df['Country'].value_counts().head(10)
            fig = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                labels={'x': 'Number of Shipments', 'y': 'Country'},
                color=country_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Shipments by product group
        if 'Product_Group' in df.columns:
            st.subheader("Shipments by Product Group")
            product_counts = df['Product_Group'].value_counts()
            fig = px.pie(
                values=product_counts.values,
                names=product_counts.index,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    if 'Delivered_to_Client_Date' in df.columns:
        st.subheader("📈 Shipments Over Time")
        df_time = df.groupby(df['Delivered_to_Client_Date'].dt.to_period('M')).size().reset_index()
        df_time.columns = ['Month', 'Shipments']
        df_time['Month'] = df_time['Month'].dt.to_timestamp()
        
        fig = px.line(df_time, x='Month', y='Shipments', markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Shipment mode distribution
    if 'Shipment_Mode' in df.columns:
        st.subheader("🚚 Shipment Mode Distribution")
        mode_counts = df['Shipment_Mode'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=mode_counts.index,
                y=mode_counts.values,
                labels={'x': 'Shipment Mode', 'y': 'Count'},
                color=mode_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create metrics for each mode
            for mode, count in mode_counts.items():
                percentage = (count / len(df)) * 100
                st.metric(mode, f"{count:,}", f"{percentage:.1f}%")

def show_geographic_analysis(df):
    """Display geographic analysis"""
    st.header("🌍 Geographic Analysis")
    
    if 'Country' not in df.columns:
        st.warning("Country information not available in the dataset.")
        return
    
    # Country selector
    countries = ['All'] + sorted(df['Country'].unique().tolist())
    selected_country = st.selectbox("Select Country", countries)
    
    if selected_country != 'All':
        df_filtered = df[df['Country'] == selected_country].copy()
    else:
        df_filtered = df.copy()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Shipments", f"{len(df_filtered):,}")
    
    with col2:
        if 'Delivery_Delay_Days' in df_filtered.columns:
            avg_delay = df_filtered['Delivery_Delay_Days'].mean()
            st.metric("Avg Delay", f"{avg_delay:.1f} days")
    
    with col3:
        if 'Line_Item_Value' in df_filtered.columns:
            total_value = df_filtered['Line_Item_Value'].sum()
            st.metric("Total Value", f"${total_value/1e6:.2f}M")
    
    with col4:
        if 'Freight_Cost_(USD)' in df_filtered.columns:
            avg_freight = df_filtered['Freight_Cost_(USD)'].mean()
            st.metric("Avg Freight", f"${avg_freight:.2f}")
    
    st.markdown("---")
    
    # Country comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Delay by Country")
        if 'Delivery_Delay_Days' in df.columns:
            country_delay = df.groupby('Country')['Delivery_Delay_Days'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(
                x=country_delay.values,
                y=country_delay.index,
                orientation='h',
                labels={'x': 'Average Delay (days)', 'y': 'Country'},
                color=country_delay.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Total Value by Country")
        if 'Line_Item_Value' in df.columns:
            country_value = df.groupby('Country')['Line_Item_Value'].sum().sort_values(ascending=False).head(15)
            country_value_m = country_value / 1e6
            fig = px.bar(
                x=country_value_m.values,
                y=country_value_m.index,
                orientation='h',
                labels={'x': 'Total Value ($M)', 'y': 'Country'},
                color=country_value_m.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Shipment mode by country
    if selected_country != 'All' and 'Shipment_Mode' in df_filtered.columns:
        st.subheader(f"Shipment Modes in {selected_country}")
        mode_counts = df_filtered['Shipment_Mode'].value_counts()
        fig = px.pie(values=mode_counts.values, names=mode_counts.index, hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

def show_product_analysis(df):
    """Display product analysis"""
    st.header("📦 Product Analysis")
    
    if 'Product_Group' not in df.columns:
        st.warning("Product information not available in the dataset.")
        return
    
    # Product selector
    products = ['All'] + sorted(df['Product_Group'].unique().tolist())
    selected_product = st.selectbox("Select Product Group", products)
    
    if selected_product != 'All':
        df_filtered = df[df['Product_Group'] == selected_product].copy()
    else:
        df_filtered = df.copy()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Shipments", f"{len(df_filtered):,}")
    
    with col2:
        if 'Line_Item_Quantity' in df_filtered.columns:
            total_qty = df_filtered['Line_Item_Quantity'].sum()
            st.metric("Total Quantity", f"{total_qty:,.0f}")
    
    with col3:
        if 'Weight_(Kilograms)' in df_filtered.columns:
            total_weight = df_filtered['Weight_(Kilograms)'].sum()
            st.metric("Total Weight", f"{total_weight:,.0f} kg")
    
    with col4:
        if 'Line_Item_Value' in df_filtered.columns:
            total_value = df_filtered['Line_Item_Value'].sum()
            st.metric("Total Value", f"${total_value/1e6:.2f}M")
    
    st.markdown("---")
    
    # Product comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Delay by Product")
        if 'Delivery_Delay_Days' in df.columns:
            product_delay = df.groupby('Product_Group')['Delivery_Delay_Days'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=product_delay.index,
                y=product_delay.values,
                labels={'x': 'Product Group', 'y': 'Average Delay (days)'},
                color=product_delay.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Shipment Count by Product")
        product_counts = df['Product_Group'].value_counts()
        fig = px.bar(
            x=product_counts.index,
            y=product_counts.values,
            labels={'x': 'Product Group', 'y': 'Number of Shipments'},
            color=product_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Value analysis
    if 'Line_Item_Value' in df.columns:
        st.subheader("💰 Value Distribution by Product")
        product_value = df.groupby('Product_Group')['Line_Item_Value'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        product_value['sum_millions'] = product_value['sum'] / 1e6
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=product_value.index,
            y=product_value['sum_millions'],
            name='Total Value ($M)',
            marker_color='lightblue'
        ))
        fig.update_layout(
            xaxis_title='Product Group',
            yaxis_title='Total Value ($M)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_delivery_performance(df):
    """Display delivery performance analysis"""
    st.header("⏱️ Delivery Performance Analysis")
    
    if 'Delivery_Delay_Days' not in df.columns:
        st.warning("Delivery delay information not available. Please ensure date columns are present.")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_delay = df['Delivery_Delay_Days'].mean()
        st.metric("Avg Delay", f"{avg_delay:.1f} days")
    
    with col2:
        median_delay = df['Delivery_Delay_Days'].median()
        st.metric("Median Delay", f"{median_delay:.1f} days")
    
    with col3:
        on_time = (df['Delivery_Delay_Days'] <= 0).sum()
        on_time_pct = (on_time / len(df)) * 100
        st.metric("On-Time %", f"{on_time_pct:.1f}%")
    
    with col4:
        late = (df['Delivery_Delay_Days'] > 0).sum()
        late_pct = (late / len(df)) * 100
        st.metric("Late %", f"{late_pct:.1f}%")
    
    st.markdown("---")
    
    # Delay distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Delay Distribution")
        fig = px.histogram(
            df,
            x='Delivery_Delay_Days',
            nbins=50,
            labels={'Delivery_Delay_Days': 'Delay (days)', 'count': 'Frequency'},
            color_discrete_sequence=['steelblue']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Delay Box Plot by Shipment Mode")
        if 'Shipment_Mode' in df.columns:
            fig = px.box(
                df,
                x='Shipment_Mode',
                y='Delivery_Delay_Days',
                labels={'Delivery_Delay_Days': 'Delay (days)', 'Shipment_Mode': 'Mode'},
                color='Shipment_Mode'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    if 'Month_Name' in df.columns:
        st.subheader("📅 Average Delay by Month")
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_delay = df.groupby('Month_Name')['Delivery_Delay_Days'].mean()
        monthly_delay = monthly_delay.reindex([m for m in month_order if m in monthly_delay.index])
        
        fig = px.line(
            x=monthly_delay.index,
            y=monthly_delay.values,
            markers=True,
            labels={'x': 'Month', 'y': 'Average Delay (days)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Country and product delay comparison
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Country' in df.columns:
            st.subheader("Top 10 Countries by Delay")
            country_delay = df.groupby('Country')['Delivery_Delay_Days'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=country_delay.values,
                y=country_delay.index,
                orientation='h',
                labels={'x': 'Average Delay (days)', 'y': 'Country'},
                color=country_delay.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Product_Group' in df.columns:
            st.subheader("Delay by Product Group")
            product_delay = df.groupby('Product_Group')['Delivery_Delay_Days'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=product_delay.values,
                y=product_delay.index,
                orientation='h',
                labels={'x': 'Average Delay (days)', 'y': 'Product'},
                color=product_delay.values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_cost_analysis(df):
    """Display cost analysis"""
    st.header("💰 Cost Analysis")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Freight_Cost_(USD)' in df.columns:
            total_freight = df['Freight_Cost_(USD)'].sum()
            st.metric("Total Freight", f"${total_freight/1e6:.2f}M")
    
    with col2:
        if 'Freight_Cost_(USD)' in df.columns:
            avg_freight = df['Freight_Cost_(USD)'].mean()
            st.metric("Avg Freight", f"${avg_freight:.2f}")
    
    with col3:
        if 'Line_Item_Value' in df.columns:
            total_value = df['Line_Item_Value'].sum()
            st.metric("Total Value", f"${total_value/1e6:.2f}M")
    
    with col4:
        if 'Line_Item_Value' in df.columns and 'Freight_Cost_(USD)' in df.columns:
            freight_ratio = (df['Freight_Cost_(USD)'].sum() / df['Line_Item_Value'].sum()) * 100
            st.metric("Freight/Value %", f"{freight_ratio:.2f}%")
    
    st.markdown("---")
    
    # Cost distributions
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Freight_Cost_(USD)' in df.columns:
            st.subheader("Freight Cost Distribution")
            fig = px.histogram(
                df,
                x='Freight_Cost_(USD)',
                nbins=50,
                labels={'Freight_Cost_(USD)': 'Freight Cost ($)'},
                color_discrete_sequence=['green']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Line_Item_Value' in df.columns:
            st.subheader("Line Item Value Distribution")
            fig = px.histogram(
                df,
                x='Line_Item_Value',
                nbins=50,
                labels={'Line_Item_Value': 'Line Item Value ($)'},
                color_discrete_sequence=['blue']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Country cost analysis
    if 'Country' in df.columns and 'Freight_Cost_(USD)' in df.columns:
        st.subheader("Average Freight Cost by Country (Top 15)")
        country_freight = df.groupby('Country')['Freight_Cost_(USD)'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=country_freight.values,
            y=country_freight.index,
            orientation='h',
            labels={'x': 'Average Freight Cost ($)', 'y': 'Country'},
            color=country_freight.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Shipment mode cost comparison
    if 'Shipment_Mode' in df.columns and 'Freight_Cost_(USD)' in df.columns:
        st.subheader("Cost Comparison by Shipment Mode")
        mode_stats = df.groupby('Shipment_Mode').agg({
            'Freight_Cost_(USD)': ['mean', 'median', 'sum', 'count']
        }).round(2)
        
        fig = px.box(
            df,
            x='Shipment_Mode',
            y='Freight_Cost_(USD)',
            labels={'Freight_Cost_(USD)': 'Freight Cost ($)', 'Shipment_Mode': 'Mode'},
            color='Shipment_Mode'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_ml_predictions(df):
    """Display ML model predictions"""
    st.header("🤖 Machine Learning Predictions")
    
    model, mappings = load_ml_model()
    
    if model is None:
        st.warning("ML model not found. Please train the model first using Basic_ML_For_ARM.py")
        return
    
    st.success("✅ Model loaded successfully!")
    
    st.markdown("### Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Countries", len(mappings['country_mapping']))
    with col2:
        st.metric("Products", len(mappings['product_mapping']))
    with col3:
        st.metric("Features", len(mappings['feature_names']))
    
    st.markdown("---")
    
    st.markdown("### 🎯 Make a Prediction")
    st.markdown("Enter shipment details to predict delivery delay:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical inputs
        country = st.selectbox("Country", sorted(mappings['country_mapping'].keys()))
        product = st.selectbox("Product Group", sorted(mappings['product_mapping'].keys()))
        
        # Numerical inputs
        weight = st.number_input("Weight (kg)", min_value=0.1, value=10.0, step=0.1)
        quantity = st.number_input("Quantity", min_value=1, value=100, step=1)
    
    with col2:
        freight = st.number_input("Freight Cost (USD)", min_value=0.0, value=1000.0, step=10.0)
        month = st.slider("Month", min_value=1, max_value=12, value=datetime.now().month)
        quarter = (month - 1) // 3 + 1
        st.info(f"Quarter: Q{quarter}")
    
    if st.button("🔮 Predict Delivery Delay", type="primary"):
        # Prepare features
        country_code = mappings['country_mapping'].get(country, 0)
        product_code = mappings['product_mapping'].get(product, 0)
        
        features = np.array([[
            float(weight),
            float(quantity),
            float(freight),
            float(month),
            float(quarter),
            float(country_code),
            float(product_code)
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Display result
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Delay", f"{prediction:.1f} days")
        
        with col2:
            if prediction <= 0:
                st.success("✅ On Time")
            elif prediction <= 7:
                st.warning("⚠️ Slight Delay")
            else:
                st.error("❌ Significant Delay")
        
        with col3:
            confidence = "High" if abs(prediction) < 30 else "Medium"
            st.info(f"Confidence: {confidence}")
        
        # Additional insights
        st.markdown("### 💡 Insights")
        if prediction > 0:
            st.write(f"- Expected delivery: **{prediction:.1f} days late**")
            st.write(f"- Consider expedited shipping for time-sensitive orders")
        else:
            st.write(f"- Expected delivery: **{abs(prediction):.1f} days early**")
            st.write(f"- Good performance expected for this route and product")

def show_tariff_simulator(df):
    """Display tariff impact simulator"""
    st.header("🎯 Tariff Impact Simulator")
    
    st.markdown("""
    This tool simulates the impact of U.S. tariffs on delivery times and costs based on:
    - Country vulnerability factors
    - Product category sensitivity
    - Historical trade policy analysis
    """)
    
    st.markdown("---")
    
    # Load model for baseline prediction
    model, mappings = load_ml_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Shipment Details")
        
        if model is not None:
            country = st.selectbox("Origin Country", sorted(mappings['country_mapping'].keys()), key="tariff_country")
            product = st.selectbox("Product Group", sorted(mappings['product_mapping'].keys()), key="tariff_product")
        else:
            country = st.text_input("Origin Country", "India")
            product = st.text_input("Product Group", "ARV")
        
        weight = st.number_input("Weight (kg)", min_value=0.1, value=10.0, step=0.1, key="tariff_weight")
        quantity = st.number_input("Quantity", min_value=1, value=100, step=1, key="tariff_quantity")
        freight = st.number_input("Freight Cost (USD)", min_value=0.0, value=1000.0, step=10.0, key="tariff_freight")
    
    with col2:
        st.markdown("### 📊 Tariff Scenario")
        
        tariff_pct = st.slider("Tariff Increase (%)", min_value=0, max_value=50, value=25, step=5)
        
        st.markdown("### 🌍 Vulnerability Factors")
        
        # Predefined vulnerability scores
        country_vuln_map = {
            'India': 0.85, 'China': 0.75, 'Vietnam': 0.45,
            'Germany': 0.35, 'France': 0.35, 'South Africa': 0.25,
            'USA': 0.10
        }
        
        product_vuln_map = {
            'ARV': 0.80, 'HRDT': 0.65, 'ACT': 0.40,
            'ANTM': 0.40, 'MRDT': 0.50
        }
        
        country_vuln = country_vuln_map.get(country, 0.5)
        product_vuln = product_vuln_map.get(product, 0.5)
        
        st.info(f"Country Vulnerability: {country_vuln:.2f}")
        st.info(f"Product Vulnerability: {product_vuln:.2f}")
    
    if st.button("🚀 Simulate Impact", type="primary"):
        st.markdown("---")
        st.markdown("### 📈 Simulation Results")
        
        # Calculate baseline delay
        baseline_delay = 0
        if model is not None:
            country_code = mappings['country_mapping'].get(country, 0)
            product_code = mappings['product_mapping'].get(product, 0)
            month = datetime.now().month
            quarter = (month - 1) // 3 + 1
            
            features = np.array([[
                float(weight), float(quantity), float(freight),
                float(month), float(quarter),
                float(country_code), float(product_code)
            ]])
            
            baseline_delay = model.predict(features)[0]
        else:
            baseline_delay = 5.0  # Default estimate
        
        # Calculate tariff impact
        base_impact_per_10pct = 3.5  # days
        cost_impact_per_10pct = 0.08  # 8%
        
        additional_delay = (base_impact_per_10pct * (tariff_pct / 10) * 
                          country_vuln * product_vuln)
        
        cost_increase_pct = (cost_impact_per_10pct * (tariff_pct / 10) * 
                            country_vuln * product_vuln) * 100
        
        total_delay = baseline_delay + additional_delay
        new_freight_cost = freight * (1 + cost_increase_pct / 100)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Baseline Delay",
                f"{baseline_delay:.1f} days",
                help="Predicted delay without tariffs"
            )
        
        with col2:
            st.metric(
                "Total Delay",
                f"{total_delay:.1f} days",
                delta=f"+{additional_delay:.1f} days",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "New Freight Cost",
                f"${new_freight_cost:.2f}",
                delta=f"+{cost_increase_pct:.1f}%",
                delta_color="inverse"
            )
        
        # Visualization
        st.markdown("### 📊 Impact Visualization")
        
        # Delay comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=['Delivery Delay'],
            y=[baseline_delay],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='With Tariff',
            x=['Delivery Delay'],
            y=[total_delay],
            marker_color='coral'
        ))
        
        fig.update_layout(
            title=f"Impact of {tariff_pct}% Tariff on Delivery Time",
            yaxis_title="Days",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if additional_delay > 7:
            st.error("⚠️ **High Impact Alert**")
            st.write("- Consider alternative sourcing countries with lower vulnerability")
            st.write("- Build safety stock to account for extended delays")
            st.write("- Negotiate expedited shipping options")
        elif additional_delay > 3:
            st.warning("⚠️ **Moderate Impact**")
            st.write("- Monitor supply chain closely")
            st.write("- Consider partial diversification of suppliers")
        else:
            st.success("✅ **Low Impact**")
            st.write("- Current supply chain resilient to tariff changes")
            st.write("- Minor adjustments may be sufficient")

# Run the app
if __name__ == "__main__":
    main()
