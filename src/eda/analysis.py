import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def initial_data_overview(df, output_dir='eda/results'):
    """
    Performs initial data overview and saves results as CSVs
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic dataset info
    basic_info = pd.DataFrame({
        'Metric': ['Number of Rows', 'Number of Columns', 'Memory Usage (MB)'],
        'Value': [
            len(df),
            len(df.columns),
            round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        ]
    })
    basic_info.to_csv(f'{output_dir}/basic_info.csv', index=False)
    
    # Data types summary
    dtype_summary = pd.DataFrame(
        df.dtypes.value_counts()
    ).reset_index().rename(columns={'index': 'dtype', 0: 'count'})
    dtype_summary.to_csv(f'{output_dir}/dtype_summary.csv', index=False)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing Percentage': missing_pct.values
    }).query('`Missing Count` > 0')
    missing_summary.to_csv(f'{output_dir}/missing_values.csv', index=False)
    
    # Basic statistics for numeric columns
    numeric_summary = df.describe(include=[np.number])
    numeric_summary.to_csv(f'{output_dir}/numeric_summary.csv')
    
    # Cardinality of categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cardinality = pd.DataFrame({
        'Column': list(categorical_cols),
        'Unique Values': [df[col].nunique() for col in categorical_cols]
    })
    cardinality.to_csv(f'{output_dir}/categorical_cardinality.csv', index=False)


def comprehensive_eda(df):
    # Distribution Analysis
    numeric_distributions = {
        'summary': df.describe(),
        'skewness': df.select_dtypes(include=[np.number]).skew(),
        'kurtosis': df.select_dtypes(include=[np.number]).kurtosis()
    }
    
    # Relationship Analysis
    relationships = {
        'correlations': df.corr(),  # Only for numeric variables
        'categorical_associations': {} # We can add chi-square tests here
    }
    
    # Quality Analysis
    quality_metrics = {
        'missing_data': df.isnull().sum() / len(df) * 100,
        'duplicates': df.duplicated().sum(),
        'unique_counts': df.nunique()
    }
    
    # Time-based Analysis (specific to survival data)
    time_analysis = {
        'event_rate': df['efs'].mean(),
        'median_followup': df['efs_time'].median(),
        'censoring_rate': (1 - df['efs']).mean()
    }
    
    return {
        'distributions': numeric_distributions,
        'relationships': relationships,
        'quality': quality_metrics,
        'temporal': time_analysis
    }

