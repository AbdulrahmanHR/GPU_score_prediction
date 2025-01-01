# predict.py
import pandas as pd
import json
from inference import InferencePipeline
import joblib

def load_known_categories():
    """Load known categories from saved file"""
    try:
        with open('models/data_processing/known_categories.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: Known categories file not found.")
        return {}

def get_user_input(known_categories):
    """Get GPU specifications from user input with category guidance"""
    gpu_specs = {}
    
    # Get manufacturer
    print("\nAvailable manufacturers: AMD, NVIDIA")
    manufacturer = input("Enter GPU manufacturer (AMD/NVIDIA): ").strip().upper()
    while manufacturer not in ['AMD', 'NVIDIA']:
        print("Invalid manufacturer. Please enter either AMD or NVIDIA.")
        manufacturer = input("Enter GPU manufacturer (AMD/NVIDIA): ").strip().upper()
    gpu_specs['manufacturer'] = manufacturer

    # Get categorical inputs with guidance
    if known_categories:
        print("\nKnown GPU chips:", ", ".join(known_categories.get('gpuChip', [])))
        print("Known bus types:", ", ".join(known_categories.get('bus', [])))
        print("Known memory types:", ", ".join(known_categories.get('memType', [])))
    
    # Get other specifications with validation
    while True:
        try:
            gpu_specs['gpuChip'] = input("\nEnter GPU chip model (e.g., GA102): ").strip()
            gpu_specs['releaseYear'] = float(input("Enter release year (e.g., 2020): ").strip())
            gpu_specs['bus'] = input("Enter bus type (e.g., PCIe 4.0 x16): ").strip()
            gpu_specs['memSize'] = float(input("Enter memory size in GB: ").strip())
            gpu_specs['memType'] = input("Enter memory type (e.g., GDDR6X): ").strip()
            gpu_specs['memBusWidth'] = float(input("Enter memory bus width in bits: ").strip())
            gpu_specs['gpuClock'] = float(input("Enter GPU clock speed in MHz: ").strip())
            gpu_specs['memClock'] = float(input("Enter memory clock speed in MHz: ").strip())
            gpu_specs['unifiedShader'] = float(input("Enter number of unified shaders: ").strip())
            gpu_specs['tmu'] = float(input("Enter number of TMUs: ").strip())
            gpu_specs['rop'] = float(input("Enter number of ROPs: ").strip())
            break
        except ValueError:
            print("\nError: Please enter valid numerical values where required.")
            print("Let's try again from the beginning.\n")

    return gpu_specs

def main():
    print("GPU Performance Prediction Tool")
    print("===============================")
    
    try:
        # Load model paths and known categories
        model_paths = joblib.load('models/model_paths.pkl')
        known_categories = load_known_categories()
        
        # Initialize inference pipeline
        pipeline = InferencePipeline()
        pipeline.load_models(model_paths)
        
        while True:
            try:
                # Get user input
                gpu_specs = get_user_input(known_categories)
                
                # Convert to DataFrame
                input_df = pd.DataFrame([gpu_specs])
                
                # Available models
                models = ['xgboost_lstm', 'lightgbm_lstm', 'xgboost_cnn', 'lightgbm_cnn']
                print("\nAvailable models:", ", ".join(models))
                
                # Get model selection
                model_name = input("Enter the model name to use for prediction: ").strip().lower()
                while model_name not in models:
                    print("Invalid model name. Please choose from:", ", ".join(models))
                    model_name = input("Enter the model name to use for prediction: ").strip().lower()
                
                # Make prediction
                try:
                    prediction = pipeline.predict(input_df, model_name)
                    print(f"\nPredicted GPU Performance Score: {prediction[0]:.2f}")
                    print("\nNote: If you entered any unknown categories, they were mapped to the most similar known categories.")
                except Exception as e:
                    print(f"Error making prediction: {str(e)}")
                
                # Ask if user wants to make another prediction
                again = input("\nWould you like to make another prediction? (yes/no): ").strip().lower()
                if again != 'yes':
                    break
                    
            except Exception as e:
                print(f"\nAn error occurred while processing input: {str(e)}")
                again = input("Would you like to try again? (yes/no): ").strip().lower()
                if again != 'yes':
                    break
        
        print("\nThank you for using the GPU Performance Prediction Tool!")
        
    except FileNotFoundError:
        print("Error: Model files not found. Please ensure the models are saved in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")