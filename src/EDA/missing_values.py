import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr



def analyze_group_differences(df, group_variable, features_list, output_dir):
    """
    Creates two CSV files analyzing how features vary across different groups:
    1. Sorted by p-value
    2. Sorted by significance score (max difference / p-value)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(output_dir)
    
    results = {}
    
    for feature in features_list:
        # Fix: Apply isnull() before groupby
        percentages_by_group = df[feature].isnull().groupby(df[group_variable]).mean() * 100
        
        contingency = pd.crosstab(df[group_variable], df[feature].isnull())
        chi2, p_value, _, _ = chi2_contingency(contingency)

        print(feature, p_value)
        
        max_diff = percentages_by_group.max() - percentages_by_group.min()
        
        results[feature] = {
            **percentages_by_group.to_dict(),
            'p_value': p_value,
            'chi2_statistic': chi2,
            'max_group_difference': max_diff,
            'significance_score': max_diff / (p_value + 1e-10)
        }
    
    results_df = pd.DataFrame.from_dict(results, orient='index').round(3)
    
    # Save both sorted versions
    results_df.sort_values('p_value').to_csv(
        f'{output_dir}/differences_by_{group_variable}_pvalue_sorted.csv'
    )
    results_df.sort_values('significance_score', ascending=False).to_csv(
        f'{output_dir}/differences_by_{group_variable}_significance_sorted.csv'
    )


def visualize_group_differences(df, group_variable, features_list, output_dir='results/EDA/plots'):
    """
    Creates three types of visualizations for missing value patterns across groups:
    1. Heatmap showing missing percentages for all features by group
    2. Bar plots for top 5 features with most significant differences
    3. Scatter plot showing significance vs effect size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    results = {}
    for feature in features_list:
        # Fix: Apply isnull() before groupby
        percentages = df[feature].isnull().groupby(df[group_variable]).mean() * 100
        results[feature] = percentages
    
    # 1. Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pd.DataFrame(results).T, cmap='YlOrRd', annot=True, fmt='.1f')
    plt.title(f'Missing Percentages by {group_variable}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_{group_variable}.png')
    plt.close()
    
    # 2. Bar plots for top 5 most significant features
    significance_scores = pd.read_csv(
        f'{output_dir}/differences_by_{group_variable}_significance_sorted.csv',
        index_col=0
    )
    top_features = significance_scores.index[:5]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        missing_rates = df[feature].isnull().groupby(df[group_variable]).mean() * 100
        sns.barplot(x=missing_rates.index, y=missing_rates.values)
        plt.xticks(rotation=45)
        plt.title(feature)
        plt.ylabel('Missing Percentage')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_features_bars_{group_variable}.png')
    plt.close()
    
    # 3. Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(significance_scores['max_group_difference'], 
                -np.log10(significance_scores['p_value'] + 1e-10))
    for i, feature in enumerate(significance_scores.index):
        plt.annotate(feature, 
                    (significance_scores['max_group_difference'][i], 
                     -np.log10(significance_scores['p_value'][i] + 1e-10)))
    plt.xlabel('Maximum Difference Between Groups')
    plt.ylabel('-log10(p-value)')
    plt.title('Feature Significance vs Effect Size')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/significance_scatter_{group_variable}.png')
    plt.close()


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