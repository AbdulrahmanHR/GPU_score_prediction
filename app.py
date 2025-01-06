# app.py
# Run command: streamlit run app.py
import streamlit as st
from inference import GPUPredictor
import numpy as np
import logging
import joblib

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Basic page configuration
st.set_page_config(
    page_title="GPU Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Essential CSS styling - removed decorative elements, keeping only functional styles
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
    </style>
    """, unsafe_allow_html=True)

def custom_number_input(label, value, min_value=None, max_value=None, help=None):
    """
    Custom numeric input function that provides better input validation
    than standard Streamlit number input
    """
    input_val = st.text_input(label, value=str(value), help=help)
    try:
        input_val = float(input_val)
        # Apply min/max constraints if specified
        if min_value is not None:
            input_val = max(min_value, input_val)
        if max_value is not None:
            input_val = min(max_value, input_val)
        return input_val
    except ValueError:
        st.warning(f"Please enter a valid number for {label}")
        return value

def main():
    """Main application function"""
    st.title("GPU Performance Predictor")
    st.markdown("Predict GPU performance using hybrid ML/DL models")
    
    try:
        # Initialize the GPU predictor and get available categories
        predictor = GPUPredictor()
        categories = predictor.get_categories()
        
        # Create tabs for predictor and about section
        tab1, tab2 = st.tabs(["Predictor", "About"])
        
        with tab1:
            # GPU specifications input form
            with st.expander("GPU Specifications", expanded=True):
                # Divide form into three columns for better organization
                col1, col2, col3 = st.columns(3)
                
                # Column 1: Categorical inputs
                with col1:
                    # Set default values for dropdown menus
                    defaults = {
                        'manufacturer': "NVIDIA",
                        'gpu_chip': "AD106",
                        'mem_type': "GDDR6",
                        'bus': "PCIe 4.0 x16"
                    }
                    
                    # Create dropdown menus with default values
                    manufacturer = st.selectbox(
                        "Manufacturer", 
                        options=categories['manufacturer'],
                        index=categories['manufacturer'].index(defaults['manufacturer']),
                        help="Select the GPU manufacturer"
                    )
                    
                    gpu_chip = st.selectbox(
                        "GPU Chip", 
                        options=categories['gpuChip'],
                        index=categories['gpuChip'].index(defaults['gpu_chip']),
                        help="Select the GPU chip model"
                    )
                    
                    mem_type = st.selectbox(
                        "Memory Type", 
                        options=categories['memType'],
                        index=categories['memType'].index(defaults['mem_type']),
                        help="Select the memory type"
                    )
                    
                    bus = st.selectbox(
                        "Bus Interface", 
                        options=categories['bus'],
                        index=categories['bus'].index(defaults['bus']),
                        help="Select the PCIe interface version"
                    )
                
                # Column 2: Memory and clock specifications
                with col2:
                    mem_size = custom_number_input(
                        "Memory Size (GB)", 8, 1, 48,
                        help="Total video memory in gigabytes"
                    )
                    mem_bus_width = custom_number_input(
                        "Memory Bus Width (bits)", 128, 32, 768,
                        help="Memory interface width in bits"
                    )
                    gpu_clock = custom_number_input(
                        "GPU Clock (MHz)", 1925, 100, 4000,
                        help="Base GPU clock speed in MHz"
                    )
                    mem_clock = custom_number_input(
                        "Memory Clock (MHz)", 2250, 100, 4000,
                        help="Memory clock speed in MHz"
                    )
                
                # Column 3: Processing units and release year
                with col3:
                    unified_shader = custom_number_input(
                        "Unified Shaders", 3840, 1, 20000,
                        help="Number of shader processing units"
                    )
                    tmu = custom_number_input(
                        "Texture Mapping Units (TMUs)", 120, 1, 1000,
                        help="Number of texture mapping units"
                    )
                    rop = custom_number_input(
                        "Render Output Units (ROPs)", 48, 1, 500,
                        help="Number of render output units"
                    )
                    release_year = custom_number_input(
                        "Release Year", 2024, 2014, 2025,
                        help="GPU release year"
                    )
            
            # Prediction button and results
            if st.button("🚀 Predict Performance"):
                with st.spinner("Making predictions..."):
                    # Prepare input data dictionary
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
                    
                    # Get predictions and original values
                    predictions, original_values = predictor.predict(input_data)
                    avg_prediction = np.mean(list(predictions.values()))
                    
                    # Display individual model predictions
                    st.header("📊 Model Predictions")
                    cols = st.columns(len(predictions))
                    for col, (model_name, prediction) in zip(cols, predictions.items()):
                        with col:
                            st.metric(
                                label=model_name.replace('_', ' ').title(),
                                value=f"{int(prediction)}", 
                                delta="Performance Score"
                            )
                    
                    # Display average prediction
                    st.markdown("### 🎯 Overall Performance Score")
                    st.metric(
                        label="Average Performance Score",
                        value=f"{int(avg_prediction)}",
                        delta="Combined Model Prediction"
                    )
                    
                    # Display input specifications summary
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
        
        # About tab content
        with tab2:
            st.markdown("""
                ### About GPU Performance Predictor
                
                This application uses machine learning and deep learning models to predict GPU performance based on hardware specifications. 
                The prediction system combines multiple models:
                
                - XGBoost + LSTM: Gradient boosting combined with long short-term memory networks
                - LightGBM + LSTM: Light gradient boosting with LSTM
                - XGBoost + CNN: Gradient boosting with convolutional neural networks
                - LightGBM + CNN: Light gradient boosting with CNN
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()