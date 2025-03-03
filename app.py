# app.py
# Run command: streamlit run app.py
import streamlit as st
from inference import GPUPredictor
import numpy as np
import logging
import pandas as pd

# GPU Performance Tiers
GPU_TIERS = {
    'Entry-level': (1000, 4000, "Basic computing, media playback, lightweight office work"),
    'Budget': (4000, 8000, "Casual gaming at 1080p, basic content creation"),
    'Mid-range': (8000, 16500, "1080p/1440p gaming, content creation"),
    'Performance': (16500, 20000, "High refresh 1440p gaming, professional workloads"),
    'High-end': (20000, 25000, "4K gaming, professional content creation"),
    'Premium': (25000, 35000, "4K high refresh gaming, AI/ML development"),
    'Enthusiast': (35000, float('inf'), "8K gaming, professional rendering, advanced AI/ML workloads")
}

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'categories' not in st.session_state:
        st.session_state.categories = None
    if 'last_input' not in st.session_state:
        st.session_state.last_input = None

def get_performance_tier(score):
    """Determine the performance tier based on G3D Mark score"""
    for tier, (min_score, max_score, description) in GPU_TIERS.items():
        if min_score <= score < max_score:
            return tier, description
    return "Enthusiast", GPU_TIERS['Enthusiast'][2]

def custom_number_input(label, key, value, min_value=None, max_value=None, help=None):
    """
    Custom numeric input function with state management
    """
    input_val = st.text_input(
        label, 
        value=str(value), 
        key=key,
        help=help,
        disabled=st.session_state.processing
    )
    try:
        input_val = float(input_val)
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
    st.set_page_config(
        page_title="GPU Performance Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Essential CSS styling
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
        [data-testid="stMetricDelta"] > div {
            color: #44c429 !important; 
        }
        /* This removes the default up/down arrow if (none) */
        [data-testid="stMetricDelta"] svg {
            display: up;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()
    
    st.title("GPU Performance Predictor")
    st.markdown("Predict GPU performance using hybrid ML/DL models")
    
    try:
        # Initialize predictor only once
        if st.session_state.predictor is None:
            with st.spinner("Loading models..."):
                st.session_state.predictor = GPUPredictor()
                st.session_state.categories = st.session_state.predictor.get_categories()
        
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
                        'gpuChip': "AD106",
                        'memType': "GDDR6",
                        'bus': "PCIe 4.0 x16"
                    }
                    
                    # Create dropdown menus with default values
                    manufacturer = st.selectbox(
                        "Manufacturer", 
                        options=st.session_state.categories['manufacturer'],
                        index=st.session_state.categories['manufacturer'].index(defaults['manufacturer']) 
                        if defaults['manufacturer'] in st.session_state.categories['manufacturer'] else 0,
                        help="Select the GPU manufacturer",
                        key="manufacturer",
                        disabled=st.session_state.processing
                    )
                    
                    gpu_chip = st.selectbox(
                        "GPU Chip", 
                        options=st.session_state.categories['gpuChip'],
                        index=st.session_state.categories['gpuChip'].index(defaults['gpuChip']) 
                        if defaults['gpuChip'] in st.session_state.categories['gpuChip'] else 0,
                        help="Select the GPU chip model",
                        key="gpu_chip",
                        disabled=st.session_state.processing
                    )
                    
                    mem_type = st.selectbox(
                        "Memory Type", 
                        options=st.session_state.categories['memType'],
                        index=st.session_state.categories['memType'].index(defaults['memType']) 
                        if defaults['memType'] in st.session_state.categories['memType'] else 0,
                        help="Select the memory type",
                        key="mem_type",
                        disabled=st.session_state.processing
                    )
                    
                    bus = st.selectbox(
                        "Bus Interface", 
                        options=st.session_state.categories['bus'],
                        index=st.session_state.categories['bus'].index(defaults['bus']) 
                        if defaults['bus'] in st.session_state.categories['bus'] else 0,
                        help="Select the PCIe interface version",
                        key="bus",
                        disabled=st.session_state.processing
                    )
                          
                # Column 2: Memory and clock specifications
                with col2:
                    mem_size = custom_number_input(
                        "Memory Size (GB)", "mem_size", 8, 1, 48,
                        help="Total video memory in gigabytes"
                    )
                    mem_bus_width = custom_number_input(
                        "Memory Bus Width (bits)", "mem_bus_width", 128, 32, 768,
                        help="Memory interface width in bits"
                    )
                    gpu_clock = custom_number_input(
                        "GPU Clock (MHz)", "gpu_clock", 1925, 100, 4000,
                        help="Base GPU clock speed in MHz"
                    )
                    mem_clock = custom_number_input(
                        "Memory Clock (MHz)", "mem_clock", 2250, 100, 4000,
                        help="Memory clock speed in MHz"
                    )
                
                # Column 3: Processing units and release year
                with col3:
                    unified_shader = custom_number_input(
                        "Unified Shaders", "unified_shader", 3840, 1, 20000,
                        help="Number of shader processing units"
                    )
                    tmu = custom_number_input(
                        "Texture Mapping Units (TMUs)", "tmu", 120, 1, 1000,
                        help="Number of texture mapping units"
                    )
                    rop = custom_number_input(
                        "Render Output Units (ROPs)", "rop", 48, 1, 500,
                        help="Number of render output units"
                    )
                    release_year = custom_number_input(
                        "Release Year", "release_year", 2024, 2014, 2025,
                        help="GPU release year"
                    )
                    
            # Create input data dictionary
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

            # Only show prediction button if input has changed
            if st.session_state.last_input != str(input_data):
                if st.button("🚀 Predict Performance", disabled=st.session_state.processing):
                    st.session_state.processing = True
                    st.session_state.last_input = str(input_data)
                    
                    with st.spinner("Making predictions..."):
                        try:
                            predictions, original_values = st.session_state.predictor.predict(input_data)
                            
                            if predictions:  # Check if we got valid predictions
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
                                                
                                # Display average prediction with performance tier
                                st.markdown("### 🎯 Overall Performance Score")
                                avg_score = int(avg_prediction)
                                tier, tier_description = get_performance_tier(avg_score)
                                st.metric(
                                    label="Average Performance Score",
                                    value=f"{avg_score}",
                                    delta=f"{tier} Tier"
                                )
                                st.info(f"This GPU falls in the **{tier}** category: {tier_description}")
                                
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
                            else:
                                st.error("No valid predictions were generated. Please check your input values.")
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                    
                    st.session_state.processing = False
                    
        # About tab content
        with tab2:
            st.markdown("### About GPU Performance Predictor")
            st.markdown("""
                This application uses machine learning and deep learning models to predict GPU performance based on hardware specifications. 
                The prediction system combines multiple models:
                
                - XGBoost + LSTM: Gradient boosting combined with long short-term memory networks
                - LightGBM + LSTM: Light gradient boosting with LSTM
                - XGBoost + CNN: Gradient boosting with convolutional neural networks
                - LightGBM + CNN: Light gradient boosting with CNN
            """)
            
            st.markdown("### Performance Tiers")
            for tier, (min_score, max_score, description) in GPU_TIERS.items():
                st.info(f"""
                **{tier}** (Score: {min_score:,} - {max_score if max_score != float('inf') else '∞'})
                {description}
                """)
            
            st.markdown("""
                ### Use Cases
                - Compare different GPU configurations
                - Evaluate potential upgrades
                - Assess performance implications of specification changes
                - Benchmark custom GPU configurations
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.processing = False
        logging.exception("Application error")

if __name__ == "__main__":
    main()