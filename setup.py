#!/usr/bin/env python
"""
AQI Analysis Setup & Execution Script
Runs the complete pipeline for data processing, ML training, and visualization
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

def run_command(cmd, description):
    """Run a shell command with description"""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}\n")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='AQI Analysis Pipeline')
    parser.add_argument('--step', type=int, default=0,
                       help='Start from specific step (0=all, 1=data, 2=preprocess, etc.)')
    parser.add_argument('--skip-models', action='store_true',
                       help='Skip ML model training')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    os.chdir(base_dir)

    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║     AQI ANALYSIS PROJECT - TAMIL NADU (2020-2025)         ║
    ║          Complete Pipeline Execution                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Load Data
    if args.step <= 1:
        if not run_command(
            "python src/data_loader.py",
            "Step 1: Loading AQI Data from CPCB"
        ):
            print("⚠️  Data loading failed. Check the error above.")
            return

    # Step 2: Preprocess Data
    if args.step <= 2:
        if not run_command(
            "python src/data_preprocessing.py",
            "Step 2: Preprocessing & Cleaning Data"
        ):
            print("⚠️  Preprocessing failed. Check the error above.")
            return

    # Step 3: Feature Engineering
    if args.step <= 3:
        if not run_command(
            "python src/features.py",
            "Step 3: Feature Engineering"
        ):
            print("⚠️  Feature engineering failed. Check the error above.")
            return

    # Step 4: Train ML Models
    if args.step <= 4 and not args.skip_models:
        if not run_command(
            "python src/models.py",
            "Step 4: Training Machine Learning Models"
        ):
            print("⚠️  Model training failed. Check the error above.")
            return

    # Step 5: Create Visualizations
    if args.step <= 5:
        if not run_command(
            "python src/visualization.py",
            "Step 5: Creating Visualizations (12+ Charts)"
        ):
            print("⚠️  Visualization creation failed. Check the error above.")
            return

    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║              ✓ PIPELINE EXECUTION COMPLETE                ║
    ╚════════════════════════════════════════════════════════════╝

    📊 DATA PIPELINE:
       ✓ Raw data loaded and validated
       ✓ Data cleaned and preprocessed
       ✓ Features engineered and normalized
       ✓ ML models trained and evaluated
       ✓ Visualizations created (12+ charts)

    📁 OUTPUT FILES:
       • Raw data: aqi_data/raw_data/tamil_nadu_aqi_raw.csv
       • Processed: aqi_data/processed_data/tamil_nadu_aqi_processed.csv
       • Features: aqi_data/processed_data/tamil_nadu_aqi_features.csv
       • Models: aqi_data/models/
       • Charts: dashboard/assets/

    📊 NEXT STEPS:

    1️⃣  Launch Jupyter Notebook for detailed analysis:
        jupyter notebook notebooks/AQI_Analysis.ipynb

    2️⃣  Start Interactive Dashboard:
        streamlit run dashboard/app.py
        (Open browser to: http://localhost:8501)

    3️⃣  Review Documentation:
        cat README.md

    ╔════════════════════════════════════════════════════════════╗
    ║  For support, check README.md or review Jupyter notebook  ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Ask what to do next
    print("\nWould you like to:")
    print("  1. Run Jupyter Notebook")
    print("  2. Start Streamlit Dashboard")
    print("  3. Exit")

    try:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            print("\nLaunching Jupyter Notebook...")
            os.system("jupyter notebook notebooks/AQI_Analysis.ipynb")

        elif choice == '2':
            print("\nLaunching Streamlit Dashboard...")
            print("Opening http://localhost:8501...")
            os.system("streamlit run dashboard/app.py")

        else:
            print("\n✓ Pipeline complete. Thank you!")

    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
