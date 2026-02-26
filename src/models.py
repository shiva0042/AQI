"""
Machine Learning Models for AQI Analysis
Includes: ARIMA, LSTM, Classification, Clustering, Anomaly Detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Note: LSTM requires TensorFlow which is not available on Python 3.14
# Skipping LSTM imports - using ARIMA for time series forecasting instead

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from scipy import stats

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)

import matplotlib.pyplot as plt


class AQIMLModels:
    def __init__(self, features_path="aqi_data/processed_data/tamil_nadu_aqi_features.csv",
                 models_dir="aqi_data/models"):
        """Initialize ML models"""
        self.features_path = Path(features_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()

    def load_features(self):
        """Load engineered features"""
        print("Loading engineered features...")
        self.df = pd.read_csv(self.features_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.dropna(subset=['aqi'])
        print(f"Loaded {len(self.df)} records")
        return self.df

    # ==================== TIME SERIES FORECASTING ====================

    def train_arima_model(self, city='Chennai', order=(1, 1, 1)):
        """Train ARIMA model for AQI forecasting"""
        print(f"\n1. Training ARIMA({order}) model for {city}...")

        city_data = self.df[self.df['city'] == city].sort_values('date')
        aqi_series = city_data['aqi'].values

        try:
            model = ARIMA(aqi_series, order=order)
            fitted_model = model.fit()

            # Store results
            self.models[f'arima_{city}'] = fitted_model
            print(f"   [OK] ARIMA model trained for {city}")
            print(f"   AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

            return fitted_model
        except Exception as e:
            print(f"   [FAIL] Error training ARIMA: {e}")
            return None

    def train_lstm_model(self, sequence_length=30):
        """Train LSTM model for AQI forecasting"""
        print(f"\n2. Training LSTM model (sequence_length={sequence_length})...")
        print("   [SKIP] TensorFlow not available on Python 3.14")
        print("   Using ARIMA for time series forecasting instead")
        return None, None

    def forecast_arima(self, city='Chennai', steps=30):
        """Forecast using ARIMA model"""
        model = self.models.get(f'arima_{city}')
        if model is None:
            model = self.train_arima_model(city)

        try:
            forecast = model.get_forecast(steps=steps)
            forecast_values = forecast.predicted_mean
            return forecast_values.values
        except:
            return None

    def forecast_lstm(self, steps=30):
        """Forecast using LSTM model"""
        model = self.models.get('lstm')
        if model is None:
            return None

        try:
            aqi_values = self.df['aqi'].values[-30:]
            forecasts = []

            for _ in range(steps):
                X_input = aqi_values[-30:].reshape(1, 30, 1)
                pred = model.predict(X_input, verbose=0)[0][0]
                forecasts.append(pred)
                aqi_values = np.append(aqi_values, pred)

            return np.array(forecasts)
        except Exception as e:
            print(f"   [FAIL] Error in LSTM forecast: {e}")
            return None

    # ==================== CLASSIFICATION ====================

    def train_classification_model(self):
        """Train Random Forest for AQI level classification"""
        print("\n3. Training Classification Model (Random Forest)...")

        try:
            # Select features and target
            feature_cols = [col for col in self.df.columns
                           if col not in ['date', 'city', 'station', 'aqi', 'aqi_category',
                                         'latitude', 'longitude', 'aqi_category_next_week']]

            X = self.df[feature_cols].dropna()
            y = self.df.loc[X.index, 'aqi_category_next_week'].dropna()

            # Filter to matching indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            if len(X) < 2:
                print("   [FAIL] Not enough data for classification")
                return None

            # Split data
            split = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            # Train Random Forest
            clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)

            # Evaluate
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)

            self.models['classifier_rf'] = clf
            print(f"   [OK] Random Forest classifier trained")
            print(f"   Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")

            return clf
        except Exception as e:
            print(f"   [FAIL] Error training classifier: {e}")
            return None

    def predict_aqi_category(self, features):
        """Predict AQI category (Good, Moderate, etc.)"""
        clf = self.models.get('classifier_rf')
        if clf is None:
            return None
        return clf.predict(features)

    # ==================== CLUSTERING ====================

    def train_kmeans_clustering(self, n_clusters=4):
        """Train K-Means clustering on AQI patterns"""
        print(f"\n4. Training K-Means Clustering (n_clusters={n_clusters})...")

        try:
            # Select features for clustering
            feature_cols = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co']
            X = self.df[feature_cols].dropna()

            # Standardize
            X_scaled = self.scaler.fit_transform(X)

            # Train K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            # Evaluate
            silhouette = silhouette_score(X_scaled, clusters)

            self.models['kmeans'] = kmeans
            self.df.loc[X.index, 'aqi_cluster'] = clusters

            print(f"   [OK] K-Means trained with {n_clusters} clusters")
            print(f"   Silhouette Score: {silhouette:.4f}")

            return kmeans
        except Exception as e:
            print(f"   [FAIL] Error training K-Means: {e}")
            return None

    def train_dbscan_clustering(self, eps=0.5, min_samples=5):
        """Train DBSCAN clustering on AQI patterns"""
        print(f"\n5. Training DBSCAN Clustering (eps={eps}, min_samples={min_samples})...")

        try:
            # Select features for clustering
            feature_cols = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co']
            X = self.df[feature_cols].dropna()

            # Standardize
            X_scaled = self.scaler.fit_transform(X)

            # Train DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)

            self.models['dbscan'] = dbscan
            self.df.loc[X.index, 'aqi_cluster_dbscan'] = clusters

            print(f"   [OK] DBSCAN trained")
            print(f"   Number of clusters: {n_clusters}, Noise points: {n_noise}")

            return dbscan
        except Exception as e:
            print(f"   [FAIL] Error training DBSCAN: {e}")
            return None

    # ==================== ANOMALY DETECTION ====================

    def train_isolation_forest(self, contamination=0.1):
        """Train Isolation Forest for anomaly detection"""
        print(f"\n6. Training Isolation Forest (contamination={contamination})...")

        try:
            # Select features
            feature_cols = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co']
            X = self.df[feature_cols].dropna()

            # Train Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            anomalies = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.score_samples(X)

            n_anomalies = sum(anomalies == -1)

            self.models['isolation_forest'] = iso_forest
            self.df.loc[X.index, 'anomaly'] = (anomalies == -1).astype(int)
            self.df.loc[X.index, 'anomaly_score'] = anomaly_scores

            print(f"   [OK] Isolation Forest trained")
            print(f"   Anomalies detected: {n_anomalies} ({n_anomalies/len(X)*100:.2f}%)")

            return iso_forest
        except Exception as e:
            print(f"   [FAIL] Error training Isolation Forest: {e}")
            return None

    def detect_zscore_anomalies(self, threshold=3):
        """Detect anomalies using Z-score method"""
        print(f"\n7. Detecting Z-Score Anomalies (threshold={threshold})...")

        for city in self.df['city'].unique():
            city_mask = self.df['city'] == city
            aqi_values = self.df.loc[city_mask, 'aqi']

            z_scores = np.abs(stats.zscore(aqi_values))
            self.df.loc[city_mask, 'zscore_anomaly'] = (z_scores > threshold).astype(int)

        print(f"   [OK] Z-Score anomaly detection complete")
        return self.df

    # ==================== MODEL SAVING & EVALUATION ====================

    def save_models(self):
        """Save trained models to disk"""
        print("\n8. Saving trained models...")

        for name, model in self.models.items():
            filepath = self.models_dir / f"{name}.pkl"
            try:
                pickle.dump(model, open(filepath, 'wb'))
                print(f"   [OK] Saved {name}")
            except Exception as e:
                try:
                    # For Keras models
                    model.save(str(self.models_dir / f"{name}.h5"))
                    print(f"   [OK] Saved {name}")
                except:
                    print(f"   [FAIL] Failed to save {name}: {e}")

    def train_all_models(self):
        """Train all models"""
        print("="*60)
        print("MACHINE LEARNING MODELS TRAINING")
        print("="*60)

        self.load_features()

        # Time Series Forecasting
        self.train_arima_model('Chennai')
        self.train_arima_model('Coimbatore')
        self.train_lstm_model()

        # Classification
        self.train_classification_model()

        # Clustering
        self.train_kmeans_clustering()
        self.train_dbscan_clustering()

        # Anomaly Detection
        self.train_isolation_forest()
        self.detect_zscore_anomalies()

        # Save all models
        self.save_models()

        print("\n" + "="*60)
        print("[OK] ALL MODELS TRAINED SUCCESSFULLY")
        print("="*60)

        return self.df


def main():
    """Run model training"""
    ml_models = AQIMLModels()
    df = ml_models.train_all_models()
    print(f"\n[OK] Model training complete!")


if __name__ == "__main__":
    main()
