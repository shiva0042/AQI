"""
Data Preprocessing Module for AQI Analysis
Handles data cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AQIDataPreprocessor:
    def __init__(self, raw_data_path="aqi_data/raw_data/tamil_nadu_aqi_raw.csv",
                 processed_data_dir="aqi_data/processed_data"):
        """Initialize data preprocessor"""
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path(processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_raw_data(self):
        """Load raw CSV data"""
        print(f"Loading raw data from {self.raw_data_path}...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(self.df)} records")
        return self.df

    def clean_data(self):
        """Clean and validate data"""
        print("\n1. Cleaning data...")

        # Convert date columns
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        elif 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['date'] = self.df['timestamp'].dt.date

        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=['city', 'date'], keep='first')
        print(f"   Removed {initial_rows - len(self.df)} duplicate records")

        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                # Fill with median by city
                self.df[col] = self.df.groupby('city')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill remaining nulls with overall median
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print(f"   Filled {missing_count} missing values in {col}")

        # Remove outliers using IQR method
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if outliers > 0:
                # Cap values instead of removing
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                print(f"   Capped {outliers} outliers in {col}")

        print("[OK] Data cleaning complete")
        return self.df

    def normalize_data(self):
        """Normalize numerical features"""
        print("\n2. Normalizing numerical features...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        scaler_params = {}

        for col in numeric_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()

            if max_val - min_val != 0:
                self.df[f'{col}_normalized'] = (self.df[col] - min_val) / (max_val - min_val)
                scaler_params[col] = {'min': min_val, 'max': max_val}

        print(f"   Normalized {len(scaler_params)} features")
        print("[OK] Normalization complete")
        return self.df

    def create_temporal_features(self):
        """Create temporal features from date"""
        print("\n3. Creating temporal features...")

        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_year'] = self.df['date'].dt.dayofyear
            self.df['week'] = self.df['date'].dt.isocalendar().week

            # Create season feature
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Summer'
                elif month in [6, 7, 8, 9]:
                    return 'Monsoon'
                else:
                    return 'Post-Monsoon'

            self.df['season'] = self.df['month'].apply(get_season)

        print("[OK] Temporal features created")
        print(f"   Features: year, month, quarter, day_of_year, week, season")
        return self.df

    def create_statistical_features(self):
        """Create rolling and statistical features"""
        print("\n4. Creating statistical features...")

        self.df = self.df.sort_values('date')

        # Group by city and create rolling features for AQI
        for city in self.df['city'].unique():
            city_mask = self.df['city'] == city

            # Rolling averages
            self.df.loc[city_mask, 'aqi_rolling_7'] = \
                self.df[city_mask].groupby('city')['aqi'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )

            self.df.loc[city_mask, 'aqi_rolling_30'] = \
                self.df[city_mask].groupby('city')['aqi'].transform(
                    lambda x: x.rolling(window=30, min_periods=1).mean()
                )

            # Rolling standard deviation
            self.df.loc[city_mask, 'aqi_rolling_std'] = \
                self.df[city_mask].groupby('city')['aqi'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).std()
                )

        print("[OK] Rolling features created")
        print("   Features: aqi_rolling_7, aqi_rolling_30, aqi_rolling_std")
        return self.df

    def create_station_features(self):
        """Create station-based features"""
        print("\n5. Creating station-based features...")

        # Station-wise statistics
        station_stats = self.df.groupby('city').agg({
            'aqi': ['mean', 'std', 'min', 'max'],
            'pm25': 'mean',
            'pm10': 'mean'
        }).reset_index()

        station_stats.columns = ['city', 'aqi_mean', 'aqi_std', 'aqi_min', 'aqi_max',
                                  'pm25_mean', 'pm10_mean']

        self.df = self.df.merge(station_stats, on='city', how='left')

        print("[OK] Station features created")
        return self.df

    def ensure_data_quality(self):
        """Ensure final data quality"""
        print("\n6. Ensuring data quality...")

        # Check for missing values
        missing = self.df.isna().sum()
        if missing.sum() > 0:
            print("   Missing values found:")
            print(missing[missing > 0])
        else:
            print("   [OK] No missing values")

        # Check date range
        print(f"   Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"   Number of stations: {self.df['city'].nunique()}")
        print(f"   Total records: {len(self.df)}")

        return self.df

    def save_processed_data(self):
        """Save processed data to CSV"""
        filepath = self.processed_dir / "tamil_nadu_aqi_processed.csv"
        self.df.to_csv(filepath, index=False)
        print(f"\n[OK] Processed data saved to {filepath}")

        # Also save by city for individual analysis
        for city in self.df['city'].unique():
            city_df = self.df[self.df['city'] == city]
            city_filepath = self.processed_dir / f"aqi_{city.lower().replace(' ', '_')}.csv"
            city_df.to_csv(city_filepath, index=False)

        return filepath

    def preprocess(self):
        """Main preprocessing pipeline"""
        print("="*60)
        print("AQI DATA PREPROCESSING PIPELINE")
        print("="*60)

        self.load_raw_data()
        self.clean_data()
        self.normalize_data()
        self.create_temporal_features()
        self.create_statistical_features()
        self.create_station_features()
        self.ensure_data_quality()
        self.save_processed_data()

        print("\n" + "="*60)
        print("[OK] PREPROCESSING COMPLETE")
        print("="*60)

        return self.df


def main():
    """Run preprocessing pipeline"""
    preprocessor = AQIDataPreprocessor()
    df = preprocessor.preprocess()
    print("\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
