import os
import json
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr

def analyze_missing_patterns(df, output_dir='results/EDA'):
    """
    Comprehensive analysis of missing data patterns, including non-linear relationships
    and their connection to survival outcomes.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create missing indicators matrix
    missing_indicators = df.isnull().astype(int)
    
    # Step 2: Linear relationships between missing patterns
    missing_corr = missing_indicators.corr()
    missing_corr.to_csv(f'{output_dir}/missing_linear_correlations.csv')
    
    # Step 3: Non-linear relationships through chi-square tests
    chi_square_results = []
    for col1 in missing_indicators.columns:
        for col2 in missing_indicators.columns:
            if col1 < col2:  # Avoid duplicate comparisons
                contingency = pd.crosstab(missing_indicators[col1], 
                                        missing_indicators[col2])
                chi2, p_value = chi2_contingency(contingency)[:2]
                chi_square_results.append({
                    'variable1': col1,
                    'variable2': col2,
                    'chi2_statistic': chi2,
                    'p_value': p_value
                })
    
    pd.DataFrame(chi_square_results).to_csv(
        f'{output_dir}/missing_nonlinear_relationships.csv', index=False)
    
    # Step 4: Analyze relationship with survival outcomes
    survival_analysis = []
    for col in df.columns:
        missing_mask = df[col].isnull()
        if missing_mask.any():
            # Survival time differences
            time_stats = scipy.stats.mannwhitneyu(
                df[missing_mask]['efs_time'].dropna(),
                df[~missing_mask]['efs_time'].dropna()
            )
            
            # Event rate differences
            event_stats = scipy.stats.chi2_contingency(
                pd.crosstab(missing_mask, df['efs'])
            )
            
            survival_analysis.append({
                'variable': col,
                'time_diff_pvalue': time_stats.pvalue,
                'event_diff_pvalue': event_stats[1],
                'missing_event_rate': df[missing_mask]['efs'].mean(),
                'present_event_rate': df[~missing_mask]['efs'].mean()
            })
    
    pd.DataFrame(survival_analysis).to_csv(
        f'{output_dir}/missing_survival_relationship.csv', index=False)