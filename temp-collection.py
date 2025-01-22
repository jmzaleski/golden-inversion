# https://claude.ai/chat/56d60b15-8e45-4c8f-82c9-1e51c013e25f

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def parse_dogtooth_data(data_text):
    """
    Parse the fixed-width formatted Dogtooth station data
    """
    # Skip header lines
    lines = data_text.split('\n')[4:]  # Skip first 4 lines including dashed line
    
    data = []
    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                # Parse fixed width format
                month = int(line[6:8])
                day = int(line[9:11])
                time_str = line[12:16].strip()
                temp = float(line[17:24].strip())
                
                # Convert time to datetime
                hour = int(time_str) // 100
                minute = int(time_str) % 100
                
                # Assume current year (2024 for January data)
                date = datetime(2024, month, day, hour, minute)
                
                data.append({
                    'timestamp': date,
                    'temp_mountain': temp
                })
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line}")
                print(f"Error details: {e}")
                continue
    
    return pd.DataFrame(data)

def fetch_golden_airport_data(start_date, end_date):
    """
    Fetch hourly temperature data from Golden Airport station
    Uses Environment Canada historical data API
    Station ID: 1176744 (Golden A)
    """
    # Format dates for API
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Environment Canada API endpoint
    url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    
    params = {
        'format': 'csv',
        'stationID': '1176744',  # Golden A station ID
        'Year': start_date.year,
        'Month': start_date.month,
        'Day': start_date.day,
        'timeframe': '1',  # 1 for hourly data
        'submit': 'Download+Data'
    }
    
    try:
        response = requests.get(url, params=params)
        df = pd.read_csv(pd.StringIO(response.text), skiprows=0)
        # Clean and process the data
        df['timestamp'] = pd.to_datetime(df['Date/Time (LST)'])
        df = df[['timestamp', 'Temp (°C)']]
        df = df.rename(columns={'Temp (°C)': 'temp_valley'})
        return df
    except Exception as e:
        print(f"Error fetching Golden Airport data: {e}")
        return None

def plot_temperature_comparison(combined_data):
    """
    Create a plot showing valley and mountain temperatures with inversion strength
    """
    plt.figure(figsize=(12, 8))
    
    # Plot temperatures
    plt.subplot(2, 1, 1)
    plt.plot(combined_data.index, combined_data['temp_valley'], label='Valley (Golden Airport)', color='blue')
    plt.plot(combined_data.index, combined_data['temp_mountain'], label='Mountain (Dogtooth)', color='red')
    plt.legend()
    plt.title('Temperature Comparison and Inversion Analysis')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    
    # Plot inversion strength
    plt.subplot(2, 1, 2)
    plt.fill_between(combined_data.index, combined_data['inversion_strength'], 
                    where=combined_data['inversion_strength'] > 0,
                    color='red', alpha=0.3, label='Inversion')
    plt.fill_between(combined_data.index, combined_data['inversion_strength'], 
                    where=combined_data['inversion_strength'] <= 0,
                    color='blue', alpha=0.3, label='Normal Lapse Rate')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.ylabel('Inversion Strength (°C)')
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('temperature_analysis.png')

def analyze_inversion_periods(combined_data):
    """
    Analyze and report on inversion periods
    """
    # Calculate basic statistics
    inversion_periods = combined_data[combined_data['inversion_strength'] > 0]
    normal_periods = combined_data[combined_data['inversion_strength'] <= 0]
    
    stats = {
        'max_inversion': combined_data['inversion_strength'].max(),
        'avg_inversion': inversion_periods['inversion_strength'].mean() if not inversion_periods.empty else 0,
        'inversion_hours': len(inversion_periods),
        'total_hours': len(combined_data),
        'percent_time_inverted': (len(inversion_periods) / len(combined_data)) * 100 if not combined_data.empty else 0
    }
    
    return stats

def main():
    # Read the sample data
    with open('paste.txt', 'r') as f:
        mountain_data = parse_dogtooth_data(f.read())
    
    # Set index
    mountain_data.set_index('timestamp', inplace=True)
    
    # Get valley data for the same period
    start_date = mountain_data.index.min()
    end_date = mountain_data.index.max()
    valley_data = fetch_golden_airport_data(start_date, end_date)
    
    if valley_data is not None:
        valley_data.set_index('timestamp', inplace=True)
        
        # Resample both to hourly data
        mountain_hourly = mountain_data.resample('H').mean()
        valley_hourly = valley_data.resample('H').mean()
        
        # Merge the datasets
        combined_data = pd.merge(
            valley_hourly,
            mountain_hourly,
            left_index=True,
            right_index=True,
            how='outer'
        )
        
        # Calculate inversion strength
        combined_data['inversion_strength'] = combined_data['temp_mountain'] - combined_data['temp_valley']
        
        # Analyze inversions
        stats = analyze_inversion_periods(combined_data)
        
        # Create visualization
        plot_temperature_comparison(combined_data)
        
        # Display results
        print("\nInversion Analysis Results:")
        print(f"Maximum inversion strength: {stats['max_inversion']:.1f}°C")
        print(f"Average inversion strength during inversions: {stats['avg_inversion']:.1f}°C")
        print(f"Hours with inversion: {stats['inversion_hours']} out of {stats['total_hours']}")
        print(f"Percent time with inversion: {stats['percent_time_inverted']:.1f}%")
        
        # Save data
        combined_data.to_csv('temperature_analysis.csv')

if __name__ == "__main__":
    main()
