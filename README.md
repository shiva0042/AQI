# AQI Analysis Project: Tamil Nadu (2020-2025)

Comprehensive Air Quality Index Analysis with Machine Learning for Tamil Nadu Region

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![web](https://aqitngovt.streamlit.app/)
## 📋 Project Overview

This project provides an in-depth analysis of Air Quality Index (AQI) data across Tamil Nadu state from 2020 to 2025. It combines exploratory data analysis, machine learning models, and interactive visualizations to provide actionable insights into air quality patterns and trends.

### Key Features

- **6+ Years of Data**: Comprehensive AQI measurements from multiple monitoring stations
- **10+ Interactive Charts**: Plotly-based visualizations for deep insights
- **Multiple ML Models**: ARIMA, LSTM, Classification, Clustering, Anomaly Detection
- **Interactive Dashboard**: Streamlit-based real-time dashboard with filters
- **Geographic Visualization**: Map-based AQI visualization for Tamil Nadu
- **Jupyter Notebook**: Complete analysis notebook with detailed explanations

## 📁 Project Structure

```
GroceryStoreDataset/
├── aqi_data/
│   ├── raw_data/              # Raw CSV files from CPCB API
│   ├── processed_data/        # Cleaned and processed data
│   └── models/                # Trained ML models
├── notebooks/
│   ├── AQI_Analysis.ipynb     # Main analysis notebook
│   └── data_exploration.ipynb # EDA notebook
├── dashboard/
│   ├── app.py                 # Main Streamlit app
│   ├── pages/                 # Dashboard pages
│   └── assets/                # Charts and GeoJSON
├── src/
│   ├── data_loader.py         # CPCB API data fetching
│   ├── data_preprocessing.py  # Data cleaning and processing
│   ├── features.py            # Feature engineering
│   ├── models.py              # ML model implementations
│   └── visualization.py       # Chart generation utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ disk space for data

### Installation

1. **Clone or download the project**
   ```bash
   cd GroceryStoreDataset
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Option 1: Generate Data and Run Full Pipeline

```bash
# Step 1: Load/generate AQI data
python src/data_loader.py

# Step 2: Preprocess the data
python src/data_preprocessing.py

# Step 3: Engineer features
python src/features.py

# Step 4: Train ML models
python src/models.py

# Step 5: Create visualizations
python src/visualization.py

# Step 6: Run Jupyter Notebook
jupyter notebook notebooks/AQI_Analysis.ipynb

# Step 7: Launch dashboard
streamlit run dashboard/app.py
```

#### Option 2: Quick Start with Dashboard Only

```bash
streamlit run dashboard/app.py
```

## 📊 Data Overview

### Data Sources
- **Primary**: CPCB (Central Pollution Control Board) API/Data
- **Fallback**: Sample generated data (uses realistic distributions)
- **Coverage**: Tamil Nadu state (all major cities and industrial areas)
- **Time Period**: January 2020 - December 2025
- **Frequency**: Daily measurements

### Stations Covered
- Chennai
- Coimbatore
- Madurai
- Salem
- Trichy (Tiruchirapalli)
- Tiruppur
- Erode
- Vellore
- Kanyakumari
- And more...

### Pollutants Measured
- **AQI**: Air Quality Index (0-500)
- **PM2.5**: Fine Particulate Matter
- **PM10**: Coarse Particulate Matter
- **NO₂**: Nitrogen Dioxide
- **SO₂**: Sulfur Dioxide
- **CO**: Carbon Monoxide
- **O₃**: Ozone

## 🤖 Machine Learning Models

### 1. Time Series Forecasting
- **ARIMA**: Autoregressive Integrated Moving Average for trend forecasting
- **LSTM**: Long Short-Term Memory neural networks for complex patterns
- **Purpose**: Predict AQI values 7-90 days in advance
- **Metrics**: MAE, RMSE, R²

### 2. Classification
- **Random Forest**: Multi-class classification of AQI health levels
- **Classes**: Good, Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy
- **Purpose**: Predict air quality category for given conditions
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### 3. Clustering
- **K-Means**: Unsupervised learning for pattern identification
- **DBSCAN**: Density-based clustering for outlier detection
- **Purpose**: Identify similar air quality patterns across stations
- **Metrics**: Silhouette Score, Inertia

### 4. Anomaly Detection
- **Isolation Forest**: Statistical anomaly detection
- **Z-Score Method**: Statistical outlier detection
- **Purpose**: Identify unusual pollution events or measurement anomalies
- **Metrics**: Contamination rate, Anomaly score

## 📈 Visualizations (12+ Charts)

1. **AQI Trend by Year** - Line chart showing yearly trends
2. **AQI by Month** - Area chart with seasonal variations
3. **Seasonal Patterns** - Box plots of AQI distribution by season
4. **AQI by Station** - Bar chart ranking stations by pollution
5. **Station Performance Heatmap** - Monthly AQI variations by station
6. **Top Polluted Stations** - Horizontal bar chart
7. **AQI Distribution** - Histogram of AQI values
8. **Pollutant Distribution** - Violin plots for each pollutant
9. **Correlation Heatmap** - Relationships between AQI and pollutants
10. **Moving Averages** - Trend analysis with 7-day and 30-day MA
11. **Year-on-Year Comparison** - Multi-year trend analysis
12. **Anomaly Detection** - Scatter plot highlighting anomalies

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Overview Page**: Key metrics and latest AQI readings
- **Charts & Analysis**: Interactive visualizations with multiple views
- **Geographic Map**: Interactive map of Tamil Nadu with AQI data
- **ML Predictions**: Model outputs and forecasts
- **Filters**: Date range, city selection, AQI level filtering
- **Real-time Updates**: Latest data from monitoring stations
- **Export**: Download data and charts

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## 🔬 Jupyter Notebook Analysis

The main analysis notebook includes:

1. **Data Loading & Overview**
   - Data shape and statistics
   - Missing value analysis
   - Station coverage

2. **Exploratory Data Analysis**
   - Statistical summaries
   - Distribution analysis
   - Temporal patterns

3. **Feature Engineering**
   - Lag features
   - Fourier transforms for seasonality
   - Rolling statistics
   - Interaction features

4. **Machine Learning**
   - Model training
   - Cross-validation
   - Hyperparameter tuning
   - Performance evaluation

5. **Results & Insights**
   - Key findings
   - Trend analysis
   - Recommendations

### Running the Notebook

```bash
jupyter notebook notebooks/AQI_Analysis.ipynb
```

## 📝 Key Findings

Based on analysis of 2020-2025 data:

1. **Seasonal Patterns**: Clear seasonal variation with monsoon (June-September) showing lower pollution
2. **Regional Differences**: Significant variation across stations with industrial areas showing higher AQI
3. **Primary Pollutant**: PM2.5 shows strongest correlation with overall AQI
4. **Trend**: Slight improvement in air quality in recent years due to policy measures

## ⚙️ Configuration

### Environment Variables

```bash
# For WAQI API (optional)
export WAQI_TOKEN=your_waqi_token_here

# For data directories
export AQI_DATA_DIR=./aqi_data
export MODELS_DIR=./aqi_data/models
```

### Data Configuration

Edit `src/data_loader.py` to customize:
- Cities to monitor
- Date ranges
- Data sources
- API credentials

## 🛠️ Development

### Adding New Features

1. **New Visualization**: Add method to `src/visualization.py`
2. **New Model**: Add class to `src/models.py`
3. **New Metric**: Add calculation to preprocessing or notebook
4. **Dashboard Page**: Create new file in `dashboard/pages/`

### Testing

```bash
# Verify data loading
python src/data_loader.py

# Check preprocessing
python src/data_preprocessing.py

# Validate models
python src/models.py

# Generate visualizations
python src/visualization.py
```

## 📚 Dependencies

### Core Libraries
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: ML algorithms
- `tensorflow/keras`: Neural networks
- `statsmodels`: Time series analysis

### Visualization
- `plotly`: Interactive charts
- `matplotlib`: Static plots
- `seaborn`: Statistical visualization
- `folium`: Map visualization

### Dashboard
- `streamlit`: Web app framework

See `requirements.txt` for all dependencies and versions.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Author(s)

Created for comprehensive AQI analysis of Tamil Nadu region

## 📞 Support & Feedback

For issues, questions, or feedback:
- Open an issue in the repository
- Check existing documentation
- Review the Jupyter notebook for detailed examples

## 🔗 Resources

- [CPCB Data Portal](https://data.gov.in/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

## ⚠️ Disclaimer

This analysis is based on available data and uses both real CPCB data (when available) and generated sample data (for demonstration).
For official air quality information and health advisories, refer to the Central Pollution Control Board website.

---

**Last Updated**: 2025
**Version**: 1.0
**Status**: Active Development
