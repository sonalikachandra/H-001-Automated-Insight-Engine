"""
Generate sample AdTech data for testing the Insight Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_adtech_data(num_rows=1000, output_path='sample_data/adtech_sample.csv'):
    """Generate realistic AdTech campaign data"""
    
    # Ensure directory exists
    os.makedirs('sample_data', exist_ok=True)
    
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]
    
    # Campaigns
    campaigns = ['Summer Sale', 'Black Friday', 'Brand Awareness', 'Product Launch', 
                 'Retargeting', 'Holiday Special', 'Spring Collection', 'Back to School']
    
    # Ad Groups
    ad_groups = ['Search Ads', 'Display Ads', 'Video Ads', 'Shopping Ads', 
                 'Social Media', 'Email Campaign', 'Mobile Ads', 'Desktop Ads']
    
    # Channels
    channels = ['Google Ads', 'Facebook', 'Instagram', 'LinkedIn', 'Twitter', 'YouTube']
    
    # Generate data
    data = {
        'Date': np.random.choice(dates, num_rows),
        'Campaign': np.random.choice(campaigns, num_rows),
        'Ad_Group': np.random.choice(ad_groups, num_rows),
        'Channel': np.random.choice(channels, num_rows),
        'Impressions': np.random.randint(1000, 100000, num_rows),
        'Clicks': np.random.randint(10, 5000, num_rows),
        'Cost': np.random.uniform(50, 5000, num_rows).round(2),
        'Conversions': np.random.randint(1, 200, num_rows),
        'Revenue': np.random.uniform(100, 15000, num_rows).round(2),
        'Bounce_Rate': np.random.uniform(20, 80, num_rows).round(2),
        'Avg_Session_Duration': np.random.randint(30, 600, num_rows),
        'Quality_Score': np.random.randint(1, 11, num_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Add some realistic relationships
    df['Clicks'] = (df['Impressions'] * np.random.uniform(0.01, 0.15, num_rows)).astype(int)
    df['Conversions'] = (df['Clicks'] * np.random.uniform(0.01, 0.3, num_rows)).astype(int)
    df['Revenue'] = df['Conversions'] * np.random.uniform(50, 500, num_rows)
    
    # Ensure no division by zero
    df['Clicks'] = df['Clicks'].clip(lower=1)
    df['Conversions'] = df['Conversions'].clip(lower=1)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ Sample data generated: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


if __name__ == "__main__":
    df = generate_sample_adtech_data()
    print("\n📊 Sample data preview:")
    print(df.head(10))
    print("\n📈 Summary statistics:")
    print(df.describe())