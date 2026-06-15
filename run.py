import argparse
import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n======================================")
    print(f"Executing: {description}")
    print(f"======================================\n")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during execution of: {description}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Multi-Sector Energy Consumption Forecasting Project CLI Manager")
    parser.add_argument(
        "action",
        choices=["download", "generate", "preprocess", "features", "train", "evaluate", "serve", "pipeline"],
        help="Action to perform. 'pipeline' runs everything end-to-end and starts the server."
    )
    
    args = parser.parse_args()
    
    # Ensure correct python executable
    python_exec = sys.executable
    
    if args.action == "download":
        run_command(f'"{python_exec}" src/data/download_real_data.py', "Real Data Ingestion")
        
    elif args.action == "generate":
        run_command(f'"{python_exec}" src/data/generate_synthetic_data.py', "Raw Data Generation")
        
    elif args.action == "preprocess":
        run_command(f'"{python_exec}" src/data/preprocess_residential.py', "Residential Preprocessing")
        run_command(f'"{python_exec}" src/data/preprocess_commercial.py', "Commercial Preprocessing")
        run_command(f'"{python_exec}" src/data/preprocess_industrial.py', "Industrial Preprocessing")
        
    elif args.action == "features":
        run_command(f'"{python_exec}" src/data/feature_engineering.py', "Feature Engineering")
        
    elif args.action == "train":
        run_command(f'"{python_exec}" -m src.models.train_models', "Model Training (Seasonal, LSTM, XGBoost, Hybrid)")
        
    elif args.action == "evaluate":
        run_command(f'"{python_exec}" -m src.models.evaluate_model', "Model Evaluation & Metrics Export")
        
    elif args.action == "serve":
        run_command('uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload', "FastAPI Serve & Dashboard")
        
    elif args.action == "pipeline":
        print("\nRunning complete pipeline end-to-end...")
        run_command(f'"{python_exec}" src/data/download_real_data.py', "1. Real Data Ingestion")
        run_command(f'"{python_exec}" src/data/preprocess_residential.py', "2. Residential Preprocessing")
        run_command(f'"{python_exec}" src/data/preprocess_commercial.py', "2. Commercial Preprocessing")
        run_command(f'"{python_exec}" src/data/preprocess_industrial.py', "2. Industrial Preprocessing")
        run_command(f'"{python_exec}" src/data/feature_engineering.py', "3. Feature Engineering")
        run_command(f'"{python_exec}" -m src.models.train_models', "4. Model Training")
        run_command(f'"{python_exec}" -m src.models.evaluate_model', "5. Model Evaluation")
        run_command('uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload', "6. Launching FastAPI Server & Dashboard")

if __name__ == "__main__":
    main()
