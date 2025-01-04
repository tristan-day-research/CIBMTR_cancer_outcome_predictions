import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def initial_data_overview(df, title="Data Overview"):
    overview = {}
    
    # Basic dataset info
    overview['Basic Info'] = {
        'Number of Rows': len(df),
        'Number of Columns': len(df.columns),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    # Data types summary
    overview['Data Types'] = df.dtypes.value_counts().to_dict()
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    }).query('`Missing Count` > 0')
    overview['Missing Values'] = missing_summary
    
    # Basic statistics for numeric columns
    numeric_summary = df.describe(include=[np.number])
    overview['Numeric Summary'] = numeric_summary
    
    # Cardinality of categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cardinality = {col: df[col].nunique() for col in categorical_cols}
    overview['Categorical Cardinality'] = cardinality
    
    return overview

