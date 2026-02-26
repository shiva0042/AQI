"""
QUICK START GUIDE - AQI Analysis Project
Run this to get started immediately
"""

import os
import sys
from pathlib import Path

def print_header():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌍 AQI ANALYSIS PROJECT - TAMIL NADU (2020-2025) 🌍           ║
║                    Quick Start Guide & Instructions                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

def print_installation_steps():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  STEP 1: INSTALLATION & SETUP                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
""")
    print("\n1️⃣  Navigate to project directory:")
    print("   cd GroceryStoreDataset")

    print("\n2️⃣  Create Python virtual environment:")
    print("   python -m venv venv")

    print("\n3️⃣  Activate virtual environment:")
    print("   # Windows:")
    print("   venv\\Scripts\\activate")
    print("   # Linux/Mac:")
    print("   source venv/bin/activate")

    print("\n4️⃣  Install dependencies:")
    print("   pip install -r requirements.txt")

    print("\n" + "="*75 + "\n")

def print_pipeline_steps():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  STEP 2: RUN DATA COLLECTION & PROCESSING PIPELINE                       ║
╠═══════════════════════════════════════════════════════════════════════════╣

Option A: RUN COMPLETE PIPELINE (Recommended)
─────────────────────────────────────────────

python setup.py

This will automatically:
  ✓ Load AQI data from CPCB
  ✓ Clean and preprocess data
  ✓ Engineer advanced features
  ✓ Train ML models
  ✓ Generate 12+ visualizations

Then it will prompt you to launch Jupyter or Dashboard.


Option B: RUN INDIVIDUAL STEPS MANUALLY
───────────────────────────────────────

python src/data_loader.py           # Load data
python src/data_preprocessing.py    # Clean & preprocess
python src/features.py              # Engineer features
python src/models.py                # Train ML models
python src/visualization.py         # Create charts

""")
    print("="*75 + "\n")

def print_dashboard_steps():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  STEP 3: LAUNCH INTERACTIVE DASHBOARD                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣

streamlit run dashboard/app.py

Dashboard features:
  📊 Overview: Key metrics and latest AQI readings
  📈 Charts: Interactive visualizations (12+ charts)
  🗺️  Map: Geographic visualization of Tamil Nadu
  🤖 ML: Machine learning predictions and insights
  📋 About: Project information and metrics

The dashboard will open at: http://localhost:8501

""")
    print("="*75 + "\n")

def print_notebook_steps():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  STEP 4: DETAILED ANALYSIS - JUPYTER NOTEBOOK                            ║
╠═══════════════════════════════════════════════════════════════════════════╣

jupyter notebook notebooks/AQI_Analysis.ipynb

The notebook includes:
  1. Data Loading & Overview
  2. Exploratory Data Analysis (5+ visualizations)
  3. Statistical Analysis
  4. Machine Learning Models
  5. Results & Insights
  6. Recommendations

Run all cells (Kernel → Run All) for complete analysis.

""")
    print("="*75 + "\n")

def print_project_structure():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  PROJECT STRUCTURE & FILES                                               ║
╠═══════════════════════════════════════════════════════════════════════════╣

GroceryStoreDataset/
├── 📂 aqi_data/
│   ├── raw_data/              ← Raw CSV files from API
│   ├── processed_data/        ← Cleaned data & features
│   └── models/                ← Trained ML models
│
├── 📂 notebooks/
│   ├── AQI_Analysis.ipynb     ← Main analysis (10+ visualizations)
│   └── data_exploration.ipynb ← EDA notebook
│
├── 📂 src/
│   ├── data_loader.py         ← CPCB API data fetching
│   ├── data_preprocessing.py  ← Data cleaning
│   ├── features.py            ← Feature engineering
│   ├── models.py              ← ML models (ARIMA, LSTM, etc)
│   └── visualization.py       ← Chart generation (12 charts)
│
├── 📂 dashboard/
│   ├── app.py                 ← Main Streamlit app
│   └── assets/                ← Generated charts & maps
│
├── requirements.txt           ← Python dependencies
├── setup.py                   ← Automated pipeline script
├── README.md                  ← Full documentation
└── QUICKSTART.md              ← This file

""")
    print("="*75 + "\n")

def print_features_summary():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  PROJECT FEATURES & CAPABILITIES                                         ║
╠═══════════════════════════════════════════════════════════════════════════╣

📊 DATA:
  • 6+ years of AQI data (2020-2025)
  • 10+ Tamil Nadu monitoring stations
  • 6 pollutants tracked: AQI, PM2.5, PM10, NO₂, SO₂, CO
  • 50,000+ data points

🔄 PROCESSING:
  • Data validation & cleaning
  • Missing value imputation
  • Outlier detection & capping
  • Temporal feature extraction
  • Statistical feature engineering
  • Standardization & normalization

🤖 MACHINE LEARNING (4 Model Types):
  1. Time Series Forecasting
     • ARIMA for trend analysis
     • LSTM for pattern prediction
  2. Classification
     • Random Forest for AQI level prediction
  3. Clustering
     • K-Means for pattern identification
     • DBSCAN for anomaly detection
  4. Anomaly Detection
     • Isolation Forest
     • Z-Score method

📈 VISUALIZATIONS (12+ Charts):
  1. AQI Trend by Year
  2. AQI by Month
  3. Seasonal Patterns
  4. AQI by Station
  5. Station Performance Heatmap
  6. Top Polluted Stations
  7. AQI Distribution
  8. Pollutant Distribution
  9. Correlation Heatmap
  10. Moving Averages
  11. Year-on-Year Comparison
  12. Anomaly Detection

🌐 DASHBOARD:
  • 5 interactive pages
  • Real-time filtering
  • Geographic mapping
  • ML predictions
  • Responsive design

📓 JUPYTER NOTEBOOK:
  • 10+ visualizations
  • Detailed explanations
  • Statistical analysis
  • ML model training
  • Actionable insights

""")
    print("="*75 + "\n")

def print_troubleshooting():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  TROUBLESHOOTING & COMMON ISSUES                                         ║
╠═══════════════════════════════════════════════════════════════════════════╣

❌ "ModuleNotFoundError: No module named 'streamlit'"
   → Solution: pip install -r requirements.txt

❌ "Port 8501 is already in use"
   → Solution: streamlit run dashboard/app.py --server.port 8502

❌ "No such file or directory: aqi_data/processed_data..."
   → Solution: Run data_loader.py first, then data_preprocessing.py

❌ "TensorFlow/LSTM errors"
   → Solution: These are optional. Models gracefully degrade if unavailable

❌ Memory issues with large datasets
   → Solution: Process data in batches or use --profile-memory flag

📖 For more help, see README.md or check the Jupyter notebook

""")
    print("="*75 + "\n")

def print_tips():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  TIPS & BEST PRACTICES                                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣

💡 Tips:
  • Start with Dashboard for quick insights
  • Use Jupyter for detailed analysis
  • Check data in aqi_data/processed_data/ folder
  • ML models are optional - dashboard works without them
  • Use filters in dashboard to focus on specific areas/dates

⚡ Performance:
  • First run may take 5-10 minutes (data processing)
  • Subsequent runs are faster (cached data)
  • Dashboard loads in seconds once data is ready
  • LSTM training is CPU/GPU intensive

📚 Learning Resources:
  • README.md: Full documentation
  • AQI_Analysis.ipynb: Detailed analysis walkthrough
  • Source code comments: Implementation details

🔄 Data Updates:
  • To refresh data: Delete aqi_data/raw_data/ files
  • Then run: python src/data_loader.py

""")
    print("="*75 + "\n")

def print_next_steps():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║  NEXT STEPS                                                              ║
╠═══════════════════════════════════════════════════════════════════════════╣

🎯 IMMEDIATE (Next 5 minutes):
  1. Install dependencies: pip install -r requirements.txt
  2. Run pipeline: python setup.py
  3. Launch dashboard: streamlit run dashboard/app.py

📊 SHORT TERM (Next hour):
  1. Explore dashboard pages
  2. Check different time periods
  3. Review generated charts
  4. Open Jupyter notebook for deep dive

🔬 LONG TERM (Next day+):
  1. Analyze trends for your specific area
  2. Train custom models with your parameters
  3. Share findings with stakeholders
  4. Use insights for decision-making

❓ Questions?
  • Check README.md for detailed documentation
  • Review Jupyter notebook for examples
  • Check source code comments
  • See GitHub issues/discussions

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                    🚀 READY TO GET STARTED? 🚀                          ║
║                                                                           ║
║              Run: python setup.py                                         ║
║              Or: streamlit run dashboard/app.py                           ║
║              Or: jupyter notebook notebooks/AQI_Analysis.ipynb           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

def main():
    print_header()
    input("Press Enter to continue...")

    print_installation_steps()
    input("Press Enter to continue...")

    print_pipeline_steps()
    input("Press Enter to continue...")

    print_dashboard_steps()
    input("Press Enter to continue...")

    print_notebook_steps()
    input("Press Enter to continue...")

    print_project_structure()
    input("Press Enter to continue...")

    print_features_summary()
    input("Press Enter to continue...")

    print_troubleshooting()
    input("Press Enter to continue...")

    print_tips()
    input("Press Enter to continue...")

    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGuide closed. Happy analyzing! 🌍📊")
