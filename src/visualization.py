"""
Visualization Module for AQI Analysis
Creates 10+ charts for data exploration and model results
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AQIVisualizations:
    def __init__(self, features_path="aqi_data/processed_data/tamil_nadu_aqi_features.csv",
                 output_dir="dashboard/assets"):
        """Initialize visualization module"""
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_data(self):
        """Load feature data"""
        print("Loading data for visualizations...")
        self.df = pd.read_csv(self.features_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"Loaded {len(self.df)} records")
        return self.df

    # ==================== CHART 1: AQI TREND BY YEAR ====================
    def chart_aqi_trend_by_year(self):
        """Line chart: AQI trends by year"""
        print("\n1. Creating AQI Trend by Year chart...")

        if 'year' not in self.df.columns:
            return None

        yearly_data = self.df.groupby(['year', 'city'])['aqi'].mean().reset_index()

        fig = px.line(yearly_data, x='year', y='aqi', color='city',
                      title='AQI Trends by Year (2020-2025)',
                      labels={'aqi': 'Average AQI', 'year': 'Year'},
                      markers=True)
        fig.write_html(self.output_dir / "01_aqi_trend_by_year.html")
        print("   [OK] Saved: 01_aqi_trend_by_year.html")
        return fig

    # ==================== CHART 2: AQI BY MONTH ====================
    def chart_aqi_by_month(self):
        """Area chart: AQI by month across all years"""
        print("2. Creating AQI by Month chart...")

        if 'month' not in self.df.columns:
            return None

        monthly_data = self.df.groupby('month')['aqi'].agg(['mean', 'std']).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['mean'],
                                 fill='tozeroy', name='Average AQI'))
        fig.update_layout(title='Average AQI by Month', xaxis_title='Month', yaxis_title='AQI')
        fig.write_html(self.output_dir / "02_aqi_by_month.html")
        print("   [OK] Saved: 02_aqi_by_month.html")
        return fig

    # ==================== CHART 3: SEASONAL PATTERNS ====================
    def chart_seasonal_patterns(self):
        """Box plot: AQI distribution by season"""
        print("3. Creating Seasonal Patterns chart...")

        if 'season' not in self.df.columns:
            return None

        fig = px.box(self.df, x='season', y='aqi',
                     title='AQI Distribution by Season',
                     labels={'aqi': 'AQI Value', 'season': 'Season'},
                     points='outliers')
        fig.write_html(self.output_dir / "03_seasonal_patterns.html")
        print("   [OK] Saved: 03_seasonal_patterns.html")
        return fig

    # ==================== CHART 4: AQI BY STATION ====================
    def chart_aqi_by_station(self):
        """Bar chart: Average AQI by station"""
        print("4. Creating AQI by Station chart...")

        station_aqi = self.df.groupby('city')['aqi'].mean().sort_values(ascending=False).reset_index()

        fig = px.bar(station_aqi, x='city', y='aqi',
                     title='Average AQI by Station (Tamil Nadu)',
                     labels={'aqi': 'Average AQI', 'city': 'Station'},
                     color='aqi', color_continuous_scale='RdYlGn_r')
        fig.update_xaxes(tickangle=-45)
        fig.write_html(self.output_dir / "04_aqi_by_station.html")
        print("   [OK] Saved: 04_aqi_by_station.html")
        return fig

    # ==================== CHART 5: STATION PERFORMANCE HEATMAP ====================
    def chart_station_heatmap(self):
        """Heatmap: AQI values by month and station"""
        print("5. Creating Station Performance Heatmap...")

        if 'month' not in self.df.columns:
            return None

        heatmap_data = self.df.groupby(['city', 'month'])['aqi'].mean().reset_index()
        pivot_data = heatmap_data.pivot(index='city', columns='month', values='aqi')

        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn_r'
        ))
        fig.update_layout(title='AQI by Station and Month Heatmap',
                         xaxis_title='Month', yaxis_title='Station')
        fig.write_html(self.output_dir / "05_station_heatmap.html")
        print("   [OK] Saved: 05_station_heatmap.html")
        return fig

    # ==================== CHART 6: TOP POLLUTED STATIONS ====================
    def chart_top_polluted_stations(self):
        """Horizontal bar: Top 10 most polluted stations"""
        print("6. Creating Top Polluted Stations chart...")

        top_stations = self.df.groupby('city')['aqi'].mean().sort_values(ascending=True).tail(10)

        fig = go.Figure(data=go.Bar(
            y=top_stations.index,
            x=top_stations.values,
            orientation='h',
            marker=dict(color=top_stations.values, colorscale='Reds')
        ))
        fig.update_layout(
            title='Top 10 Most Polluted Stations',
            xaxis_title='Average AQI',
            yaxis_title='Station'
        )
        fig.write_html(self.output_dir / "06_top_polluted_stations.html")
        print("   [OK] Saved: 06_top_polluted_stations.html")
        return fig

    # ==================== CHART 7: AQI DISTRIBUTION HISTOGRAM ====================
    def chart_aqi_distribution(self):
        """Histogram: Distribution of AQI values"""
        print("7. Creating AQI Distribution Histogram...")

        fig = px.histogram(self.df, x='aqi', nbins=50,
                          title='Distribution of AQI Values',
                          labels={'aqi': 'AQI Value', 'count': 'Frequency'},
                          color_discrete_sequence=['#1f77b4'])
        fig.write_html(self.output_dir / "07_aqi_distribution.html")
        print("   [OK] Saved: 07_aqi_distribution.html")
        return fig

    # ==================== CHART 8: POLLUTANT DISTRIBUTION ====================
    def chart_pollutant_distribution(self):
        """Violin plot: Distribution of different pollutants"""
        print("8. Creating Pollutant Distribution chart...")

        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co']
        melted_df = self.df[['date'] + pollutants].melt(var_name='Pollutant', value_name='Concentration')

        fig = px.violin(melted_df, x='Pollutant', y='Concentration',
                       title='Distribution of Air Pollutants',
                       labels={'Concentration': 'Concentration Level'})
        fig.write_html(self.output_dir / "08_pollutant_distribution.html")
        print("   [OK] Saved: 08_pollutant_distribution.html")
        return fig

    # ==================== CHART 9: CORRELATION HEATMAP ====================
    def chart_correlation_heatmap(self):
        """Heatmap: Correlation between AQI and pollutants"""
        print("9. Creating Correlation Heatmap...")

        numeric_cols = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        corr_matrix = self.df[available_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(title='Correlation Matrix: AQI and Pollutants')
        fig.write_html(self.output_dir / "09_correlation_heatmap.html")
        print("   [OK] Saved: 09_correlation_heatmap.html")
        return fig

    # ==================== CHART 10: MOVING AVERAGE TRENDS ====================
    def chart_moving_averages(self):
        """Line chart: AQI with moving averages"""
        print("10. Creating Moving Average Trends chart...")

        sample_city = self.df['city'].iloc[0]
        city_data = self.df[self.df['city'] == sample_city].sort_values('date')

        if 'aqi_rolling_7' not in self.df.columns:
            city_data['ma_7'] = city_data['aqi'].rolling(window=7).mean()
            city_data['ma_30'] = city_data['aqi'].rolling(window=30).mean()
        else:
            city_data = city_data.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=city_data['date'], y=city_data['aqi'],
                                 name='Daily AQI', mode='lines'))
        if 'aqi_rolling_7' in city_data.columns:
            fig.add_trace(go.Scatter(x=city_data['date'], y=city_data['aqi_rolling_7'],
                                     name='7-Day MA', mode='lines'))
        if 'aqi_rolling_30' in city_data.columns:
            fig.add_trace(go.Scatter(x=city_data['date'], y=city_data['aqi_rolling_30'],
                                     name='30-Day MA', mode='lines'))

        fig.update_layout(title=f'AQI Trends with Moving Averages - {sample_city}',
                         xaxis_title='Date', yaxis_title='AQI',
                         hovermode='x unified')
        fig.write_html(self.output_dir / "10_moving_averages.html")
        print("   [OK] Saved: 10_moving_averages.html")
        return fig

    # ==================== CHART 11: YEAR-ON-YEAR COMPARISON ====================
    def chart_year_on_year(self):
        """Multi-line chart: Year-on-year AQI comparison"""
        print("11. Creating Year-on-Year Comparison chart...")

        if 'month' not in self.df.columns or 'year' not in self.df.columns:
            return None

        yoy_data = self.df.groupby(['year', 'month'])['aqi'].mean().reset_index()

        fig = px.line(yoy_data, x='month', y='aqi', color='year',
                     title='Year-on-Year AQI Comparison by Month',
                     labels={'aqi': 'Average AQI', 'month': 'Month', 'year': 'Year'},
                     markers=True)
        fig.write_html(self.output_dir / "11_year_on_year.html")
        print("   [OK] Saved: 11_year_on_year.html")
        return fig

    # ==================== CHART 12: ANOMALY DETECTED EVENTS ====================
    def chart_anomaly_detection(self):
        """Scatter plot: Anomalies in AQI data"""
        print("12. Creating Anomaly Detection chart...")

        sample_city = self.df['city'].iloc[0]
        city_data = self.df[self.df['city'] == sample_city].sort_values('date')

        fig = go.Figure()

        # Check if anomaly column exists
        if 'anomaly' in city_data.columns:
            normal_data = city_data[city_data['anomaly'] == 0]
            anomaly_data = city_data[city_data['anomaly'] == 1]

            fig.add_trace(go.Scatter(x=normal_data['date'], y=normal_data['aqi'],
                                    mode='markers', name='Normal',
                                    marker=dict(color='blue', size=4)))
            if len(anomaly_data) > 0:
                fig.add_trace(go.Scatter(x=anomaly_data['date'], y=anomaly_data['aqi'],
                                        mode='markers', name='Anomaly',
                                        marker=dict(color='red', size=8)))
        else:
            # If no anomaly column, just plot AQI normally
            fig.add_trace(go.Scatter(x=city_data['date'], y=city_data['aqi'],
                                    mode='markers', name='AQI',
                                    marker=dict(color='blue', size=4)))

        fig.update_layout(title=f'Anomaly Detection in AQI - {sample_city}',
                         xaxis_title='Date', yaxis_title='AQI')
        fig.write_html(self.output_dir / "12_anomaly_detection.html")
        print("   [OK] Saved: 12_anomaly_detection.html")
        return fig

    def create_all_visualizations(self):
        """Create all visualizations"""
        print("="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        self.load_data()

        self.chart_aqi_trend_by_year()
        self.chart_aqi_by_month()
        self.chart_seasonal_patterns()
        self.chart_aqi_by_station()
        self.chart_station_heatmap()
        self.chart_top_polluted_stations()
        self.chart_aqi_distribution()
        self.chart_pollutant_distribution()
        self.chart_correlation_heatmap()
        self.chart_moving_averages()
        self.chart_year_on_year()
        self.chart_anomaly_detection()

        print("\n" + "="*60)
        print("[OK] ALL VISUALIZATIONS CREATED (12 CHARTS)")
        print("="*60)
        print(f"Charts saved to: {self.output_dir}")


def main():
    """Run visualization creation"""
    viz = AQIVisualizations()
    viz.create_all_visualizations()


if __name__ == "__main__":
    main()
