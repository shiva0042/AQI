"""
Feature Engineering Module for AQI Analysis
Creates advanced features for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AQIFeatureEngineer:
    def __init__(self, processed_data_path="aqi_data/processed_data/tamil_nadu_aqi_processed.csv"):
        """Initialize feature engineer"""
        self.processed_data_path = Path(processed_data_path)
        self.df = None
        self.scaler = StandardScaler()

    def load_processed_data(self):
        """Load preprocessed data"""
        print("Loading processed data...")
        self.df = pd.read_csv(self.processed_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['city', 'date'])
        print(f"Loaded {len(self.df)} records")
        return self.df

    def create_lag_features(self, lags=[1, 7, 14, 30]):
        """Create lag features for time series"""
        print(f"\n1. Creating lag features (lags: {lags})...")

        for lag in lags:
            self.df[f'aqi_lag_{lag}'] = self.df.groupby('city')['aqi'].shift(lag)
            self.df[f'pm25_lag_{lag}'] = self.df.groupby('city')['pm25'].shift(lag)

        # Fill NaN values created by lag
        lag_cols = [col for col in self.df.columns if 'lag' in col]
        for col in lag_cols:
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        print(f"   Created {len(lag_cols)} lag features")
        return self.df

    def create_fourier_features(self, periods=[365, 30, 7]):
        """Create Fourier features for seasonality"""
        print(f"\n2. Creating Fourier features...")

        self.df['day_num'] = (self.df['date'] - self.df['date'].min()).dt.days

        for period in periods:
            self.df[f'sin_{period}d'] = np.sin(2 * np.pi * self.df['day_num'] / period)
            self.df[f'cos_{period}d'] = np.cos(2 * np.pi * self.df['day_num'] / period)

        print(f"   Created Fourier features for periods: {periods}")
        return self.df

    def create_interaction_features(self):
        """Create interaction features between pollutants"""
        print("\n3. Creating interaction features...")

        # PM interactions
        self.df['pm_ratio'] = self.df['pm25'] / (self.df['pm10'] + 1)
        self.df['pm_sum'] = self.df['pm25'] + self.df['pm10']

        # Pollutant combinations
        self.df['pollutant_index'] = (
            self.df['pm25'] * 0.5 + self.df['pm10'] * 0.3 +
            self.df['no2'] * 0.1 + self.df['so2'] * 0.1
        )

        print("   Created interaction features: pm_ratio, pm_sum, pollutant_index")
        return self.df

    def create_aggregation_features(self):
        """Create aggregation features by time periods"""
        print("\n4. Creating aggregation features...")

        # Monthly aggregation
        monthly = self.df.groupby(['city', self.df['date'].dt.to_period('M')])['aqi'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        monthly.columns = ['city', 'month', 'aqi_monthly_mean', 'aqi_monthly_std',
                           'aqi_monthly_min', 'aqi_monthly_max']

        # Merge back
        self.df['month_period'] = self.df['date'].dt.to_period('M')
        self.df = self.df.merge(monthly, left_on=['city', 'month_period'],
                                 right_on=['city', 'month'], how='left')

        print("   Created monthly aggregation features")
        return self.df

    def create_trend_features(self):
        """Create trend features"""
        print("\n5. Creating trend features...")

        # Calculate trend using linear regression slope for recent 30 days
        def calculate_trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            z = np.polyfit(x, y, 1)
            return z[0]

        self.df['aqi_trend_30'] = self.df.groupby('city')['aqi'].rolling(
            window=30, min_periods=7
        ).apply(calculate_trend, raw=False).reset_index(level=0, drop=True)

        print("   Created trend features: aqi_trend_30")
        return self.df

    def create_anomaly_features(self):
        """Create anomaly detection features"""
        print("\n6. Creating anomaly features...")

        for city in self.df['city'].unique():
            city_mask = self.df['city'] == city
            aqi_values = self.df.loc[city_mask, 'aqi']

            # Z-score
            mean = aqi_values.mean()
            std = aqi_values.std()
            self.df.loc[city_mask, 'aqi_zscore'] = (aqi_values - mean) / (std + 1e-8)

            # Deviation from rolling mean
            rolling_mean = aqi_values.rolling(window=7, center=True).mean()
            self.df.loc[city_mask, 'aqi_deviation'] = aqi_values - rolling_mean

        print("   Created anomaly features: aqi_zscore, aqi_deviation")
        return self.df

    def create_station_comparison_features(self):
        """Create features comparing stations"""
        print("\n7. Creating station comparison features...")

        # Create month if it doesn't exist
        if 'month' not in self.df.columns:
            self.df['month'] = self.df['date'].dt.month

        # How station's AQI compares to regional average
        self.df['regional_aqi_mean'] = self.df.groupby('month')['aqi'].transform('mean')
        self.df['station_vs_region'] = self.df['aqi'] - self.df['regional_aqi_mean']

        # Station rank
        self.df['station_percentile'] = self.df.groupby('date')['aqi'].rank(pct=True)

        print("   Created station comparison features")
        return self.df

    def create_target_variables(self):
        """Create target variables for supervised learning"""
        print("\n8. Creating target variables...")

        # Future AQI (for forecasting)
        self.df['aqi_tomorrow'] = self.df.groupby('city')['aqi'].shift(-1)
        self.df['aqi_next_week'] = self.df.groupby('city')['aqi'].shift(-7)
        self.df['aqi_next_month'] = self.df.groupby('city')['aqi'].shift(-30)

        # AQI level category (for classification)
        def aqi_to_category(aqi):
            if aqi <= 50:
                return 0  # Good
            elif aqi <= 100:
                return 1  # Moderate
            elif aqi <= 150:
                return 2  # Unhealthy for Sensitive
            elif aqi <= 200:
                return 3  # Unhealthy
            else:
                return 4  # Very Unhealthy / Hazardous

        self.df['aqi_category'] = self.df['aqi'].apply(aqi_to_category)
        self.df['aqi_category_next_week'] = self.df.groupby('city')['aqi'].shift(-7).apply(aqi_to_category)

        print("   Created target variables: aqi_tomorrow, aqi_next_week, aqi_next_month, aqi_category")
        return self.df

    def standardize_features(self):
        """Standardize numerical features"""
        print("\n9. Standardizing features...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Exclude date-based columns
        feature_cols = [col for col in numeric_cols if not col.startswith(('year', 'month', 'day', 'week'))]

        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        print(f"   Standardized {len(feature_cols)} features")
        return self.df

    def remove_low_variance_features(self, threshold=0.01):
        """Remove features with low variance"""
        print(f"\n10. Removing low variance features (threshold={threshold})...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        variances = self.df[numeric_cols].var()
        low_var_cols = variances[variances < threshold].index.tolist()

        if low_var_cols:
            print(f"   Removing {len(low_var_cols)} low variance features: {low_var_cols[:5]}")
            self.df = self.df.drop(columns=low_var_cols)
        else:
            print("   No low variance features found")

        return self.df

    def save_engineered_features(self):
        """Save feature-engineered data"""
        filepath = self.processed_data_path.parent / "tamil_nadu_aqi_features.csv"
        self.df.to_csv(filepath, index=False)
        print(f"\n[OK] Feature-engineered data saved to {filepath}")
        return filepath

    def get_feature_summary(self):
        """Print summary of engineered features"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total records: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"\nFeature categories:")
        print(f"  Temporal: year, month, quarter, season, etc.")
        print(f"  Lag features: {sum(1 for col in self.df.columns if 'lag' in col)}")
        print(f"  Fourier features: {sum(1 for col in self.df.columns if 'sin_' in col or 'cos_' in col)}")
        print(f"  Interaction features: pm_ratio, pm_sum, pollutant_index")
        print(f"  Trend features: aqi_trend_30")
        print(f"  Anomaly features: aqi_zscore, aqi_deviation")
        print(f"  Station features: station_vs_region, station_percentile")
        print("="*60)

    def engineer_features(self):
        """Main feature engineering pipeline"""
        print("="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)

        self.load_processed_data()
        self.create_lag_features()
        self.create_fourier_features()
        self.create_interaction_features()
        self.create_aggregation_features()
        self.create_trend_features()
        self.create_anomaly_features()
        self.create_station_comparison_features()
        self.create_target_variables()
        self.standardize_features()
        self.remove_low_variance_features()
        self.save_engineered_features()
        self.get_feature_summary()

        return self.df


def main():
    """Run feature engineering"""
    engineer = AQIFeatureEngineer()
    df = engineer.engineer_features()
    print(f"\n[OK] Feature engineering complete!")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
