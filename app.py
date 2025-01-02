import streamlit as st
import pandas as pd
import joblib
import json
from inference import InferencePipeline

class GPUPredictorApp:
    def __init__(self):
        self.pipeline = InferencePipeline()
        self.gpu_chips = []
        self.load_models()
        
    def load_models(self):
        """Load the saved model paths and initialize the pipeline"""
        try:
            model_paths = joblib.load('models/model_paths.pkl')
            self.pipeline.load_models(model_paths)
            
            # Load known categories for GPU chips
            with open(model_paths['data_processing']['known_categories'], 'r') as f:
                known_categories = json.load(f)
                self.gpu_chips = sorted(known_categories['gpuChip'])
                
        except FileNotFoundError:
            st.error("Model files not found. Please ensure models are trained and saved first.")
            st.stop()
            
    def create_input_dataframe(self, inputs):
        """Create a pandas DataFrame from the input values"""
        # Create initial DataFrame
        df = pd.DataFrame([inputs])
        
        # Ensure columns are in the correct order
        expected_columns = ['gpuChip', 'releaseYear', 'memSize', 'memType', 'bus',
                          'memBusWidth', 'gpuClock', 'memClock', 'unifiedShader',
                          'tmu', 'rop', 'manufacturer']
        
        # Verify all columns exist
        for col in expected_columns:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return None
                
        return df[expected_columns]
    
    def run(self):
        st.title("GPU Performance Predictor")
        st.write("Enter GPU specifications to predict its performance score")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                manufacturer = st.selectbox(
                    "Manufacturer",
                    options=["AMD", "Intel", "NVIDIA"]
                )
                
                gpu_chip = st.selectbox(
                    "GPU Chip Model",
                    options=self.gpu_chips
                )
                
                release_year = st.number_input(
                    "Release Year",
                    min_value=2000,
                    max_value=2025,
                    value=2023
                )
                
                mem_size = st.number_input(
                    "Memory Size (GB)",
                    min_value=0,
                    max_value=48,
                    value=8
                )
                
                mem_type = st.selectbox(
                    "Memory Type",
                    options=["GDDR6X", "GDDR6", "GDDR5X", "GDDR5", "HBM2", "HBM3"]
                )
                
                bus = st.selectbox(
                    "Bus Type",
                    options=["PCIe 4.0 x16", "PCIe 3.0 x16", "PCIe 5.0 x16"]
                )
            
            with col2:
                mem_bus_width = st.number_input(
                    "Memory Bus Width",
                    min_value=64,
                    max_value=768,
                    value=256
                )
                
                gpu_clock = st.number_input(
                    "GPU Clock (MHz)",
                    min_value=100,
                    max_value=3000,
                    value=1500
                )
                
                mem_clock = st.number_input(
                    "Memory Clock (MHz)",
                    min_value=100,
                    max_value=3000,
                    value=1000
                )
                
                unified_shader = st.number_input(
                    "Unified Shaders",
                    min_value=0,
                    max_value=20000,
                    value=3000
                )
                
                tmu = st.number_input(
                    "TMUs",
                    min_value=0,
                    max_value=1000,
                    value=96
                )
                
                rop = st.number_input(
                    "ROPs",
                    min_value=0,
                    max_value=500,
                    value=64
                )
            
            submitted = st.form_submit_button("Predict Performance")
            
            if submitted:
                input_data = {
                    'manufacturer': manufacturer,
                    'gpuChip': gpu_chip,
                    'releaseYear': release_year,
                    'memSize': mem_size,
                    'memType': mem_type,
                    'bus': bus,
                    'memBusWidth': mem_bus_width,
                    'gpuClock': gpu_clock,
                    'memClock': mem_clock,
                    'unifiedShader': unified_shader,
                    'tmu': tmu,
                    'rop': rop
                }
                
                input_df = self.create_input_dataframe(input_data)
                
                if input_df is not None:
                    try:
                        predictions = self.pipeline.predict_all(input_df)
                        
                        st.subheader("Predicted Performance Scores")
                        
                        # Create a DataFrame for the results
                        results = pd.DataFrame({
                            'Model': list(predictions.keys()),
                            'Predicted Score': [pred[0] for pred in predictions.values()]
                        })
                        
                        # Calculate average prediction
                        avg_prediction = results['Predicted Score'].mean()
                        
                        # Display results
                        st.dataframe(results.style.format({'Predicted Score': '{:.2f}'}))
                        
                        st.metric(
                            label="Average Predicted Score",
                            value=f"{avg_prediction:.2f}"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app = GPUPredictorApp()
    app.run()