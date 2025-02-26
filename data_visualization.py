# data_visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def create_visualization_directory():
    """Create directory for storing visualization outputs"""
    viz_dir = 'data_visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

def plot_data_distributions(df, output_dir):
    """
    Create distribution plots for numerical and categorical features.
    
    Args:
        df: DataFrame containing the dataset
        output_dir: Directory to save the plots
    """
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # If 'score' is in numerical_cols, move it to the end for visualization
    if 'score' in numerical_cols:
        numerical_cols.remove('score')
        numerical_cols.append('score')
    
    # 1. Numerical Features Distribution - Histograms
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(16, n_rows * 4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_distributions_histograms.png'))
    plt.close()
    
    # 2. Box plots for numerical features to show outliers
    plt.figure(figsize=(16, n_rows * 4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')
        plt.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_distributions_boxplots.png'))
    plt.close()
    
    # 3. Categorical Features Distribution - Count plots
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        
        plt.figure(figsize=(12, 8))
        # If too many categories, show only the top 15
        if len(value_counts) > 15:
            top_categories = value_counts.nlargest(15).index
            filtered_df = df[df[col].isin(top_categories)]
            sns.countplot(y=col, data=filtered_df, order=value_counts.nlargest(15).index)
            plt.title(f'Distribution of {col} (Top 15 Categories)')
            plt.xlabel('Count')
            plt.ylabel(col)
        else:
            sns.countplot(y=col, data=df, order=value_counts.index)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
        
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'categorical_distribution_{col}.png'))
        plt.close()
    
    # 4. Pair plots for key numerical features
    if 'score' in numerical_cols:
        # Select key numerical features for pair plot (to avoid too many plots)
        key_features = ['score']
        
        # Add a few more important numerical features
        important_features = [
            'memSize', 'gpuClock', 'memClock', 'unifiedShader', 
            'memBusWidth', 'releaseYear'
        ]
        
        for feature in important_features:
            if feature in numerical_cols:
                key_features.append(feature)
        
        # Limit to 6 features total to keep the plot manageable
        key_features = key_features[:6]
        
        # Create pair plot
        plt.figure(figsize=(15, 15))
        sns.pairplot(df[key_features], diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Pairwise Relationships Between Key Features', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'key_features_pairplot.png'))
        plt.close()

def plot_correlation_heatmap(df, output_dir):
    """
    Create correlation heatmaps for numerical features.
    
    Args:
        df: DataFrame containing the dataset
        output_dir: Directory to save the plots
    """
    # Select only numerical columns for correlation analysis
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Plot full correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap=cmap, vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 8})
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap_full.png'))
    plt.close()
    
    # Create focused heatmap with correlations to the target variable (if exists)
    if 'score' in numerical_df.columns:
        # Sort correlations with 'score' by absolute value
        target_corr = corr_matrix['score'].drop('score').abs().sort_values(ascending=False)
        top_features = target_corr.index[:10]  # Top 10 correlated features
        
        # Create a focused correlation matrix with top correlated features
        focused_corr = corr_matrix.loc[np.append(top_features, 'score'), np.append(top_features, 'score')]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(focused_corr, annot=True, fmt=".2f", 
                    cmap=cmap, vmin=-1, vmax=1, linewidths=0.5,
                    annot_kws={"size": 10})
        plt.title("Correlation Heatmap - Top Features vs Target (score)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap_score_focused.png'))
        plt.close()
        
        # Horizontal bar plot for feature importance based on correlation with score
        plt.figure(figsize=(10, 8))
        corr_with_target = corr_matrix['score'].drop('score')
        corr_with_target = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)
        
        # Plot horizontal bar chart
        sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette='viridis')
        plt.title('Feature Correlation with Target (score)', fontsize=14)
        plt.xlabel('Correlation Coefficient')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation_with_score.png'))
        plt.close()

def visualize_preprocessing_impact(raw_df, processed_df, output_dir):
    """
    Visualize the impact of preprocessing steps on the data distribution.
    
    Args:
        raw_df: DataFrame containing the raw dataset before preprocessing
        processed_df: DataFrame containing the dataset after preprocessing
        output_dir: Directory to save the plots
    """
    # Identify common numerical columns present in both dataframes
    raw_num_cols = raw_df.select_dtypes(include=['int64', 'float64']).columns
    proc_num_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    common_num_cols = list(set(raw_num_cols).intersection(set(proc_num_cols)))
    
    # Skip if no common columns found
    if not common_num_cols:
        print("No common numerical columns found for preprocessing comparison.")
        return
    
    # Create before/after plots for each common numerical feature
    for col in common_num_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before preprocessing
        sns.histplot(raw_df[col], kde=True, ax=ax1)
        ax1.set_title(f'{col} - Before Preprocessing')
        ax1.grid(alpha=0.3)
        
        # After preprocessing
        sns.histplot(processed_df[col], kde=True, ax=ax2)
        ax2.set_title(f'{col} - After Preprocessing')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'preprocessing_impact_{col}.png'))
        plt.close()

def analyze_and_visualize_data(data_path):
    """
    Main function to generate all data visualizations from a CSV file.
    
    Args:
        data_path: Path to the CSV file containing the dataset
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Create visualization directory
    viz_dir = create_visualization_directory()
    print(f"Visualization outputs will be saved to: {viz_dir}")
    
    # Generate all visualizations
    print("\nGenerating data distribution visualizations...")
    plot_data_distributions(df, viz_dir)
    
    print("Generating correlation heatmaps...")
    plot_correlation_heatmap(df, viz_dir)
        
    print("Generating 3D relationships...")        
    plot_3d_relationships(df, viz_dir)
                
    print(f"\nAll visualizations have been saved to: {viz_dir}")
    
    return df, viz_dir

def plot_3d_relationships(df, output_dir):
    """
    Create interactive 3D scatter plots to visualize relationships between key features.
    Requires plotly to be installed.
    
    Args:
        df: DataFrame containing the dataset
        output_dir: Directory to save the plots
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is required for 3D plots. Install with: pip install plotly")
        return
    
    # Only proceed if we have score and other numerical features
    if 'score' not in df.columns:
        print("Target variable 'score' not found for 3D visualization.")
        return
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove score from numerical columns to use as target
    if 'score' in numerical_cols:
        numerical_cols.remove('score')
    
    # Need at least 2 numerical features for 3D plot
    if len(numerical_cols) < 2:
        print("Not enough numerical features for 3D visualization.")
        return
    
    # Select top 2 features that correlate most with score
    corr_matrix = df[numerical_cols + ['score']].corr()
    top_features = corr_matrix['score'].abs().sort_values(ascending=False).index.tolist()[1:3]
    
    # Create the 3D scatter plot
    if 'vendor' in df.columns:
        # With vendor as color
        fig = px.scatter_3d(df, x=top_features[0], y=top_features[1], z='score',
                           color='vendor', opacity=0.7,
                           title=f'3D Relationship: {top_features[0]} vs {top_features[1]} vs Score',
                           labels={top_features[0]: top_features[0].capitalize(),
                                  top_features[1]: top_features[1].capitalize(),
                                  'score': 'Performance Score'})
    else:
        # Without vendor information
        fig = px.scatter_3d(df, x=top_features[0], y=top_features[1], z='score',
                           color='score', opacity=0.7,
                           color_continuous_scale='viridis',
                           title=f'3D Relationship: {top_features[0]} vs {top_features[1]} vs Score',
                           labels={top_features[0]: top_features[0].capitalize(),
                                  top_features[1]: top_features[1].capitalize(),
                                  'score': 'Performance Score'})
    
    # Update marker size
    fig.update_traces(marker=dict(size=5))
    
    # Add a rotating camera
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                     scene=dict(xaxis_title=top_features[0],
                               yaxis_title=top_features[1],
                               zaxis_title='score'))
    
    # Save as HTML file (interactive)
    html_path = os.path.join(output_dir, '3d_feature_relationships.html')
    fig.write_html(html_path)
    
    # Also save as static image for reports
    img_path = os.path.join(output_dir, '3d_feature_relationships.png')
    fig.write_image(img_path, width=1000, height=800)
    
    print(f"3D visualization saved to {html_path} (interactive) and {img_path} (static)")
    
    # Create a second 3D plot with different features if available
    if len(numerical_cols) >= 4:
        # Select different features for variety
        other_features = [col for col in numerical_cols if col not in top_features][:2]
        
        if 'vendor' in df.columns:
            fig2 = px.scatter_3d(df, x=other_features[0], y=other_features[1], z='score',
                               color='vendor', opacity=0.7,
                               title=f'3D Relationship: {other_features[0]} vs {other_features[1]} vs Score',
                               labels={other_features[0]: other_features[0].capitalize(),
                                      other_features[1]: other_features[1].capitalize(),
                                      'score': 'Performance Score'})
        else:
            fig2 = px.scatter_3d(df, x=other_features[0], y=other_features[1], z='score',
                               color='score', opacity=0.7,
                               color_continuous_scale='plasma',
                               title=f'3D Relationship: {other_features[0]} vs {other_features[1]} vs Score',
                               labels={other_features[0]: other_features[0].capitalize(),
                                      other_features[1]: other_features[1].capitalize(),
                                      'score': 'Performance Score'})
        
        fig2.update_traces(marker=dict(size=5))
        fig2.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                         scene=dict(xaxis_title=other_features[0],
                                   yaxis_title=other_features[1],
                                   zaxis_title='score'))
        
        # Save alternative visualization
        html_path2 = os.path.join(output_dir, '3d_feature_relationships_alt.html')
        fig2.write_html(html_path2)
        
        img_path2 = os.path.join(output_dir, '3d_feature_relationships_alt.png')
        fig2.write_image(img_path2, width=1000, height=800)


if __name__ == "__main__":
    data_file = "gpu_specs_v6_score.csv" 
    df, output_dir = analyze_and_visualize_data(data_file)
    
    print("\nData Statistics Summary:")
    print(df.describe().round(2))