import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnergyDashboard:
    def __init__(self):
        self.data = None
        self.filtered_data = None
    
    @st.cache_data
    def load_data(_self, uploaded_file):
        """
        Load and preprocess data from uploaded file
        """
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Read Excel file
                df = pd.read_excel(uploaded_file, skiprows=2)  # Skip title row and empty row
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Remove rows with missing essential data
            df = df.dropna(subset=['Date', 'Price', 'Grade'])
            
            # Extract year for filtering
            df['Year'] = df['Date'].dt.year
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def get_filter_options(self, df):
        """
        Get unique values for filter dropdowns
        """
        grades = sorted(df['Grade'].dropna().unique())
        del_periods = sorted(df['Del. Period'].dropna().unique())
        years = sorted(df['Year'].dropna().unique())
        
        return grades, del_periods, years
    
    def filter_data(self, df, grade, del_period, year):
        """
        Filter data based on user selections
        """
        filtered_df = df.copy()
        
        if grade and grade != "All":
            filtered_df = filtered_df[filtered_df['Grade'] == grade]
        
        if del_period and del_period != "All":
            filtered_df = filtered_df[filtered_df['Del. Period'] == del_period]
        
        if year and year != "All":
            filtered_df = filtered_df[filtered_df['Year'] == year]
        
        return filtered_df
    
    def prepare_daily_data(self, df):
        """
        Prepare daily aggregated data for charting
        """
        if df.empty:
            return pd.DataFrame()
        
        # Group by date and calculate daily aggregates
        daily_data = df.groupby(df['Date'].dt.date).agg({
            'Price': ['mean', 'min', 'max', 'std'],
            'Volume': ['sum', 'mean', 'count'],
            'Grade': 'count'
        }).reset_index()
        
        # Flatten column names
        daily_data.columns = ['Date', 'Avg_Price', 'Min_Price', 'Max_Price', 'Price_Std',
                             'Total_Volume', 'Avg_Volume', 'Volume_Count', 'Deal_Count']
        
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        daily_data = daily_data.sort_values('Date')
        
        return daily_data
    
    def create_price_trend_chart(self, daily_data):
        """
        Create interactive price trend chart
        """
        if daily_data.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Avg_Price'],
            mode='lines+markers',
            name='Average Price',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> £%{y:.2f}/MWh<extra></extra>'
        ))
        
        # Add price range (min/max) as filled area
        if 'Min_Price' in daily_data.columns and 'Max_Price' in daily_data.columns:
            fig.add_trace(go.Scatter(
                x=daily_data['Date'],
                y=daily_data['Max_Price'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_data['Date'],
                y=daily_data['Min_Price'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Price Range',
                fillcolor='rgba(46, 134, 171, 0.2)',
                hovertemplate='<b>Min:</b> £%{y:.2f}/MWh<extra></extra>'
            ))
        
        # Add trend line
        if len(daily_data) > 1:
            x_numeric = pd.to_numeric(daily_data['Date'])
            z = np.polyfit(x_numeric, daily_data['Avg_Price'], 1)
            trend_line = np.poly1d(z)(x_numeric)
            
            fig.add_trace(go.Scatter(
                x=daily_data['Date'],
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Trend:</b> £%{y:.2f}/MWh<extra></extra>'
            ))
        
        fig.update_layout(
            title='Average Price Over Time',
            xaxis_title='Date',
            yaxis_title='Price (£/MWh)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_volume_chart(self, daily_data):
        """
        Create interactive volume chart
        """
        if daily_data.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=daily_data['Date'],
            y=daily_data['Total_Volume'],
            name='Daily Volume',
            marker_color='#A23B72',
            opacity=0.8,
            hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:.0f} MW<extra></extra>'
        ))
        
        # Add volume trend line
        if len(daily_data) > 1:
            x_numeric = pd.to_numeric(daily_data['Date'])
            z = np.polyfit(x_numeric, daily_data['Total_Volume'], 1)
            trend_line = np.poly1d(z)(x_numeric)
            
            fig.add_trace(go.Scatter(
                x=daily_data['Date'],
                y=trend_line,
                mode='lines',
                name='Volume Trend',
                line=dict(color='orange', width=3, dash='dash'),
                hovertemplate='<b>Trend:</b> %{y:.0f} MW<extra></extra>'
            ))
        
        fig.update_layout(
            title='Daily Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume (MW)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_price_distribution_chart(self, df):
        """
        Create price distribution histogram
        """
        if df.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df['Price'],
            nbinsx=20,
            name='Price Distribution',
            marker_color='#F18F01',
            opacity=0.8,
            hovertemplate='<b>Price Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_price = df['Price'].mean()
        fig.add_vline(
            x=mean_price, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: £{mean_price:.2f}/MWh"
        )
        
        fig.update_layout(
            title='Price Distribution',
            xaxis_title='Price (£/MWh)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_summary_metrics(self, df, daily_data):
        """
        Create summary metrics cards
        """
        if df.empty or daily_data.empty:
            return {
                'total_deals': 0,
                'avg_price': 0,
                'total_volume': 0,
                'price_volatility': 0,
                'date_range': 'No data'
            }
        
        metrics = {
            'total_deals': len(df),
            'avg_price': df['Price'].mean(),
            'total_volume': df['Volume'].sum(),
            'price_volatility': df['Price'].std(),
            'date_range': f"{df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')}"
        }
        
        return metrics

def main():
    st.title("⚡ Energy Trading Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = EnergyDashboard()
    
    # Sidebar for file upload and filters
    st.sidebar.header("📊 Data & Filters")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your energy trading data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with energy trading data"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            df = dashboard.load_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            
            # Get filter options
            grades, del_periods, years = dashboard.get_filter_options(df)
            
            # Filters
            st.sidebar.subheader("🔍 Filters")
            
            selected_grade = st.sidebar.selectbox(
                "Select Contract/Grade:",
                ["All"] + grades,
                index=1 if "UK OTC base load" in grades else 0,
                help="Choose the energy contract type"
            )
            
            selected_del_period = st.sidebar.selectbox(
                "Select Delivery Period:",
                ["All"] + del_periods,
                index=1 if "winter Gregorian" in del_periods else 0,
                help="Choose the delivery period"
            )
            
            selected_year = st.sidebar.selectbox(
                "Select Year:",
                ["All"] + [str(year) for year in years],
                help="Choose the year for analysis"
            )
            
            # Filter data
            filtered_df = dashboard.filter_data(df, selected_grade, selected_del_period, selected_year)
            daily_data = dashboard.prepare_daily_data(filtered_df)
            
            # Display current selection
            st.sidebar.markdown("### Current Selection:")
            st.sidebar.info(f"""
            **Contract:** {selected_grade}  
            **Delivery Period:** {selected_del_period}  
            **Year:** {selected_year}  
            **Records:** {len(filtered_df)}
            """)
            
            # Main dashboard
            if not filtered_df.empty:
                # Summary metrics
                metrics = dashboard.create_summary_metrics(filtered_df, daily_data)
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Deals", f"{metrics['total_deals']:,}")
                
                with col2:
                    st.metric("Average Price", f"£{metrics['avg_price']:.2f}/MWh")
                
                with col3:
                    st.metric("Total Volume", f"{metrics['total_volume']:,.0f} MW")
                
                with col4:
                    st.metric("Price Volatility", f"£{metrics['price_volatility']:.2f}")
                
                st.markdown(f"**Analysis Period:** {metrics['date_range']}")
                st.markdown("---")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price trend chart
                    price_chart = dashboard.create_price_trend_chart(daily_data)
                    st.plotly_chart(price_chart, use_container_width=True)
                
                with col2:
                    # Volume chart
                    volume_chart = dashboard.create_volume_chart(daily_data)
                    st.plotly_chart(volume_chart, use_container_width=True)
                
                # Price distribution (full width)
                price_dist_chart = dashboard.create_price_distribution_chart(filtered_df)
                st.plotly_chart(price_dist_chart, use_container_width=True)
                
                # Data table (expandable)
                with st.expander("📋 View Raw Data"):
                    st.dataframe(
                        filtered_df[['Date', 'Grade', 'Del. Period', 'Price', 'Volume', 'Year']].head(100),
                        use_container_width=True
                    )
                
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv,
                    file_name=f'filtered_energy_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
                
            else:
                st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        
        else:
            st.error("❌ Could not load the uploaded file. Please check the file format.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload your energy trading data file using the sidebar to get started.")
        
        st.markdown("""
        ### 📋 Expected Data Format
        
        Your file should contain the following columns:
        - **Date**: Trading date
        - **Grade**: Contract type (e.g., 'UK OTC base load')
        - **Del. Period**: Delivery period (e.g., 'winter Gregorian')
        - **Price**: Price in £/MWh
        - **Volume**: Volume in MW
        - **Year**: Year of the trade
        
        ### 🎯 Features
        
        - **Interactive Charts**: Zoom, pan, and hover for details
        - **Dynamic Filtering**: Filter by contract, delivery period, and year
        - **Real-time Updates**: Charts update automatically when filters change
        - **Data Export**: Download filtered data as CSV
        - **Responsive Design**: Works on desktop and mobile
        """)

if __name__ == "__main__":
    main()
