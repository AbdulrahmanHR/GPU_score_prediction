# app.py
import streamlit as st
from inference import GPUPredictor
import os

# Set page config
st.set_page_config(
    page_title="GPU Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "GPU Performance Predictor - Predict GPU performance using machine learning models"
    }
)

# Custom CSS
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
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🎮 GPU Performance Predictor")
    st.markdown("Predict GPU performance using advanced machine learning models")
    
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
                    manufacturer = st.selectbox("Manufacturer", options=categories['manufacturer'])
                    gpu_chip = st.selectbox("GPU Chip", options=categories['gpuChip'])
                    mem_type = st.selectbox("Memory Type", options=categories['memType'])
                    bus = st.selectbox("Bus Interface", options=categories['bus'])
                
                with col2:
                    mem_size = st.number_input(
                        "Memory Size (GB)",
                        min_value=1, 
                        max_value=48, 
                        value=8
                    )
                    
                    mem_bus_width = st.number_input(
                        "Memory Bus Width (bits)",
                        min_value=32,
                        max_value=768,
                        value=128
                    )
                    
                    gpu_clock = st.number_input(
                        "GPU Clock (MHz)",
                        min_value=100,
                        max_value=4000,
                        value=1925
                    )
                    
                    mem_clock = st.number_input(
                        "Memory Clock (MHz)",
                        min_value=100,
                        max_value=4000,
                        value=2250
                    )
                
                with col3:
                    unified_shader = st.number_input(
                        "Unified Shaders", 
                        min_value=1, 
                        max_value=20000, 
                        value=3840
                    )
                    
                    tmu = st.number_input(
                        "Texture Mapping Units (TMUs)",
                        min_value=1,
                        max_value=1000,
                        value=120
                    )
                    
                    rop = st.number_input(
                        "Render Output Units (ROPs)",
                        min_value=1,
                        max_value=500,
                        value=48
                    )
                    
                    release_year = st.number_input(
                        "Release Year",
                        min_value=2014,
                        max_value=2025,
                        value=2024
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
                    
                    # Display predictions with improved visualization
                    st.header("📊 Predicted Performance Scores")
                    
                    # Create metrics with color coding
                    cols = st.columns(len(predictions))
                    for col, (model_name, prediction) in zip(cols, predictions.items()):
                        with col:
                            delta_color = "normal" if prediction >= 50 else "off"
                            st.metric(
                                label=model_name.replace('_', ' ').title(),
                                value=f"{prediction:.2f}",
                                delta="Performance Score",
                                delta_color=delta_color
                            )
                    
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
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()