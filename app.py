# app.py
# streamlit run app.py
import streamlit as st
from inference import GPUPredictor
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

# Set page config
st.set_page_config(
    page_title="GPU Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "GPU Performance Predictor - Predict GPU performance using machine learning models"
    }
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
    /* Super-sized metric styling */
    div.big-metric div[data-testid="stMetricValue"] {
        font-size: 6rem !important;
        font-weight: 900 !important;
        color: #0f5fff !important;
    }
    div.big-metric div[data-testid="stMetricDelta"] {
        font-size: 2.5rem !important;
    }
    div.big-metric div[data-testid="stMetricLabel"] {
        font-size: 3rem !important;
        font-weight: bold !important;
    }
    /* Additional emphasis for the container */
    div.big-metric.super-big-metric {
        padding: 2rem !important;
        margin: 2rem 0 !important;
    }
    div.big-metric.super-big-metric div[data-testid="stMetricValue"] {
        font-size: 7rem !important;
        line-height: 1.2 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom numeric input using st.text_input to hide increment/decrement buttons
def custom_number_input(label, value, min_value=None, max_value=None, help=None):
    input_val = st.text_input(label, value=str(value), help=help)
    try:
        # Convert the input to a number and enforce min/max constraints
        input_val = float(input_val)
        if min_value is not None and input_val < min_value:
            input_val = min_value
        if max_value is not None and input_val > max_value:
            input_val = max_value
        return input_val
    except ValueError:
        st.warning(f"Please enter a valid number for {label}")
        return value

def main():
    st.title("GPU Performance Predictor")
    st.markdown("Predict GPU performance using hybrid ML/DL models")
    
    try:
        # Initialize predictor
        predictor = GPUPredictor()
        categories = predictor.get_categories()
        
        # Create tabs for input and about
        tab1, tab2 = st.tabs(["Predictor", "About"])
        
        with tab1:
            # Create input form
            with st.expander("GPU Specifications", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    manufacturer_default = "NVIDIA"  # Replace with your desired default value
                    gpu_chip_default = "AD106"   # Replace with your desired default value
                    mem_type_default = "GDDR6"      # Replace with your desired default value
                    bus_default = "PCIe 4.0 x16"        # Replace with your desired default value

                    manufacturer = st.selectbox(
                        "Manufacturer", 
                        options=categories['manufacturer'],
                        index=categories['manufacturer'].index(manufacturer_default) if manufacturer_default in categories['manufacturer'] else 0,
                        help="Select the GPU manufacturer"
                    )

                    gpu_chip = st.selectbox(
                        "GPU Chip", 
                        options=categories['gpuChip'],
                        index=categories['gpuChip'].index(gpu_chip_default) if gpu_chip_default in categories['gpuChip'] else 0,
                        help="Select the GPU chip model"
                    )

                    mem_type = st.selectbox(
                        "Memory Type", 
                        options=categories['memType'],
                        index=categories['memType'].index(mem_type_default) if mem_type_default in categories['memType'] else 0,
                        help="Select the memory type (e.g., GDDR6)"
                    )

                    bus = st.selectbox(
                        "Bus Interface", 
                        options=categories['bus'],
                        index=categories['bus'].index(bus_default) if bus_default in categories['bus'] else 0,
                        help="Select the PCIe interface version"
                    )
                
                with col2:
                    mem_size = custom_number_input(
                        "Memory Size (GB)", value=8, min_value=1, max_value=48,
                        help="Total video memory in gigabytes"
                    )

                    mem_bus_width = custom_number_input(
                        "Memory Bus Width (bits)", value=128, min_value=32, max_value=768,
                        help="Memory interface width in bits"
                    )

                    gpu_clock = custom_number_input(
                        "GPU Clock (MHz)", value=1925, min_value=100, max_value=4000,
                        help="Base GPU clock speed in MHz"
                    )

                    mem_clock = custom_number_input(
                        "Memory Clock (MHz)", value=2250, min_value=100, max_value=4000,
                        help="Memory clock speed in MHz"
                    )
                
                with col3:
                    unified_shader = custom_number_input(
                        "Unified Shaders", value=3840, min_value=1, max_value=20000,
                        help="Number of shader processing units"
                    )

                    tmu = custom_number_input(
                        "Texture Mapping Units (TMUs)", value=120, min_value=1, max_value=1000,
                        help="Number of texture mapping units"
                    )

                    rop = custom_number_input(
                        "Render Output Units (ROPs)", value=48, min_value=1, max_value=500,
                        help="Number of render output units"
                    )

                    release_year = custom_number_input(
                        "Release Year", value=2024, min_value=2014, max_value=2025,
                        help="GPU release year"
                    )
            
            if st.button("🚀 Predict Performance"):
                with st.spinner("Making predictions..."):
                    input_data = {
                        'manufacturer': manufacturer,
                        'gpuChip': gpu_chip,
                        'memType': mem_type,
                        'bus': bus,
                        'releaseYear': release_year,
                        'memSize': mem_size,
                        'memBusWidth': mem_bus_width,
                        'gpuClock': gpu_clock,
                        'memClock': mem_clock,
                        'unifiedShader': unified_shader,
                        'tmu': tmu,
                        'rop': rop
                    }
                    
                    predictions, original_values = predictor.predict(input_data)
                    
                    # Calculate average prediction
                    avg_prediction = np.mean(list(predictions.values()))
                    
                    # Display predictions with improved visualization
                    st.header("📊 Model Predictions")
                    
                    # Display individual predictions
                    cols = st.columns(len(predictions))
                    for col, (model_name, prediction) in zip(cols, predictions.items()):
                        with col:
                            delta_color = "normal" if prediction >= 50 else "off"
                            st.metric(
                                label=model_name.replace('_', ' ').title(),
                                value=f"{int(prediction)}", 
                                delta="Performance Score",
                                delta_color=delta_color
                            )
                    
                    # Display average with bigger styling and integer value
                    st.markdown("### 🎯 Overall Performance Score")
                    avg_delta_color = "normal" if avg_prediction >= 50 else "off"
                    
                    # Wrap the metric in a div with enhanced custom CSS classes
                    st.markdown('<div class="super-big-metric">', unsafe_allow_html=True)
                    st.metric(
                        label="Average Performance Score",
                        value=f"{int(avg_prediction)}",  
                        delta="Combined Model Prediction",
                        delta_color=avg_delta_color
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display specifications in an organized manner
                    with st.expander("Selected Specifications", expanded=True):
                        specs_col1, specs_col2 = st.columns(2)
                        
                        with specs_col1:
                            st.markdown("### 📝 Hardware Details")
                            st.info(f"""
                            - **Manufacturer:** {original_values['manufacturer']}
                            - **GPU Chip:** {original_values['gpuChip']}
                            - **Memory Type:** {original_values['memType']}
                            - **Bus Interface:** {original_values['bus']}
                            """)
                        
                        with specs_col2:
                            st.markdown("### ⚙️ Technical Specifications")
                            st.info(f"""
                            - **Memory:** {mem_size}GB @ {mem_clock}MHz
                            - **Bus Width:** {mem_bus_width} bits
                            - **GPU Clock:** {gpu_clock}MHz
                            - **Processing Units:** {unified_shader} shaders, {tmu} TMUs, {rop} ROPs
                            """)
        
        with tab2:
            st.markdown("""
                ### About GPU Performance Predictor
                                
                This application uses machine learning and deep learning models to predict GPU performance based on hardware specifications. 
                The prediction system combines multiple models:
                
                - **XGBoost + LSTM**: Gradient boosting combined with long short-term memory networks
                - **LightGBM + LSTM**: Light gradient boosting with LSTM
                - **XGBoost + CNN**: Gradient boosting with convolutional neural networks
                - **LightGBM + CNN**: Light gradient boosting with CNN
                
                #### How to Use
                1. Enter your GPU specifications in the Predictor tab
                2. Click "Predict Performance"
                3. View the predicted performance scores from different models
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()