"""
Streamlit Frontend for Real Estate Price Prediction
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint - default to localhost
DEFAULT_API_URL = "http://localhost:8000"
API_URL = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_API_URL,
    help="Backend API URL (default: http://localhost:8000)"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def predict_price(property_data):
    """Make prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json=property_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_historical_predictions():
    """Get historical predictions from session state"""
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    return st.session_state.predictions_history

def add_to_history(prediction_result, input_data):
    """Add prediction to history"""
    history = get_historical_predictions()
    history.append({
        'timestamp': datetime.now().isoformat(),
        'input': input_data,
        'prediction': prediction_result
    })
    # Keep only last 50 predictions
    if len(history) > 50:
        history = history[-50:]
    st.session_state.predictions_history = history

# Main app
def main():
    st.markdown('<h1 class="main-header">🏠 Real Estate Price Predictor</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not available. Please ensure the backend is running.")
        st.info(f"Expected API URL: {API_URL}")
        return
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 System Status")
        if health_data:
            status_emoji = "✅" if health_data.get("status") == "healthy" else "⚠️"
            st.metric("Status", f"{status_emoji} {health_data.get('status', 'unknown').upper()}")
            st.metric("Model", health_data.get("model_status", "unknown"))
            st.metric("Preprocessor", health_data.get("preprocessor_status", "unknown"))
        
        st.header("🔗 Quick Links")
        st.markdown(f"[API Documentation]({API_URL}/docs)")
        st.markdown(f"[Health Check]({API_URL}/api/v1/health)")
        st.markdown(f"[Metrics]({API_URL}/api/v1/metrics)")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📈 Analytics Dashboard", "📋 Prediction History"])
    
    with tab1:
        st.header("Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area_sqft = st.number_input("Area (sqft)", min_value=100.0, max_value=10000.0, value=1200.0, step=100.0)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
            floor = st.number_input("Floor", min_value=1, max_value=50, value=3, step=1)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10, step=1)
            property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5, step=1)
        
        with col2:
            location = st.selectbox("Location", ["City Center", "Suburb", "Rural"])
            city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi", "Pune", "Hyderabad"])
            property_type = st.selectbox("Property Type", ["Apartment", "Villa", "House"])
            facing = st.selectbox("Facing", ["East", "West", "North", "South"])
            furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
            parking = st.number_input("Parking Slots", min_value=0, max_value=5, value=1, step=1)
            amenities_score = st.slider("Amenities Score", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
            distance_city_center_km = st.number_input("Distance from City Center (km)", min_value=0.0, max_value=100.0, value=12.4, step=0.1)
        
        # Predict button
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner("Predicting..."):
                property_data = {
                    "area_sqft": area_sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "floor": floor,
                    "total_floors": total_floors,
                    "property_age": property_age,
                    "location": location,
                    "city": city,
                    "property_type": property_type,
                    "facing": facing,
                    "furnishing": furnishing,
                    "parking": parking,
                    "amenities_score": amenities_score,
                    "distance_city_center_km": distance_city_center_km
                }
                
                result = predict_price(property_data)
                
                if result:
                    # Add to history
                    add_to_history(result, property_data)
                    
                    # Display prediction
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Price",
                            f"₹{result['predicted_price']:,.2f}",
                            help="Predicted property price in INR"
                        )
                    
                    with col2:
                        st.metric(
                            "Lower Bound",
                            f"₹{result['confidence_interval']['lower_bound']:,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Upper Bound",
                            f"₹{result['confidence_interval']['upper_bound']:,.2f}"
                        )
                    
                    # Confidence interval visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Lower Bound', 'Predicted', 'Upper Bound'],
                        y=[
                            result['confidence_interval']['lower_bound'],
                            result['predicted_price'],
                            result['confidence_interval']['upper_bound']
                        ],
                        marker_color=['#ff7f0e', '#1f77b4', '#2ca02c'],
                        text=[
                            f"₹{result['confidence_interval']['lower_bound']:,.0f}",
                            f"₹{result['predicted_price']:,.0f}",
                            f"₹{result['confidence_interval']['upper_bound']:,.0f}"
                        ],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Price Prediction with Confidence Interval",
                        yaxis_title="Price (INR)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metadata
                    with st.expander("📋 Prediction Details"):
                        st.json(result)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Analytics Dashboard")
        
        history = get_historical_predictions()
        
        if len(history) == 0:
            st.info("No predictions yet. Make some predictions to see analytics!")
        else:
            # Prepare data
            df = pd.DataFrame([
                {
                    'timestamp': h['timestamp'],
                    'predicted_price': h['prediction']['predicted_price'],
                    'area_sqft': h['input']['area_sqft'],
                    'location': h['input']['location'],
                    'property_type': h['input']['property_type'],
                    'city': h['input']['city']
                }
                for h in history
            ])
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(df))
            with col2:
                st.metric("Avg Price", f"₹{df['predicted_price'].mean():,.2f}")
            with col3:
                st.metric("Min Price", f"₹{df['predicted_price'].min():,.2f}")
            with col4:
                st.metric("Max Price", f"₹{df['predicted_price'].max():,.2f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution
                fig = px.histogram(df, x='predicted_price', nbins=20, title="Price Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price by location
                fig = px.box(df, x='location', y='predicted_price', title="Price by Location")
                st.plotly_chart(fig, use_container_width=True)
            
            # Price by property type
            fig = px.bar(
                df.groupby('property_type')['predicted_price'].mean().reset_index(),
                x='property_type',
                y='predicted_price',
                title="Average Price by Property Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time series
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_sorted = df.sort_values('timestamp')
            fig = px.line(df_sorted, x='timestamp', y='predicted_price', title="Price Predictions Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Prediction History")
        
        history = get_historical_predictions()
        
        if len(history) == 0:
            st.info("No prediction history yet.")
        else:
            # Display as table
            history_df = pd.DataFrame([
                {
                    'Timestamp': h['timestamp'],
                    'Area (sqft)': h['input']['area_sqft'],
                    'Location': h['input']['location'],
                    'Property Type': h['input']['property_type'],
                    'Predicted Price': f"₹{h['prediction']['predicted_price']:,.2f}",
                    'Lower Bound': f"₹{h['prediction']['confidence_interval']['lower_bound']:,.2f}",
                    'Upper Bound': f"₹{h['prediction']['confidence_interval']['upper_bound']:,.2f}"
                }
                for h in history
            ])
            
            st.dataframe(history_df, use_container_width=True)
            
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.predictions_history = []
                st.success("History cleared!")
                st.rerun()

if __name__ == "__main__":
    main()

