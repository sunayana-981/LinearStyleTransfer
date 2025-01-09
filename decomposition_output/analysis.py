import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the transition matrices
color_df = pd.read_csv('color_transitions.csv', index_col=0)
texture_df = pd.read_csv('texture_transitions.csv', index_col=0)
comp_df = pd.read_csv('composition_transitions.csv', index_col=0)

def analyze_concept_separation():
    # 1. Prepare data for analysis
    styles = color_df.index
    
    # Calculate mean distances for each style in each concept
    style_metrics = pd.DataFrame({
        'Style': styles,
        'Color_Distinctiveness': color_df.mean(axis=1),
        'Texture_Distinctiveness': texture_df.mean(axis=1),
        'Composition_Distinctiveness': comp_df.mean(axis=1)
    })
    
    # 2. Plot Concept Separation Analysis
    plt.figure(figsize=(15, 10))
    
    # Create scatter plot matrix for concept relationships
    concepts = ['Color_Distinctiveness', 'Texture_Distinctiveness', 'Composition_Distinctiveness']
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(style_metrics[concepts])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create main plot
    plt.subplot(2, 2, (1, 2))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    
    # Add style labels to points
    for i, style in enumerate(styles):
        plt.annotate(style.replace('_', ' '), 
                    (pca_result[i, 0], pca_result[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.title('Style Distribution in Concept Space (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
    
    # Create concept contribution plot
    plt.subplot(2, 2, 3)
    concept_contributions = pd.DataFrame(
        pca.components_,
        columns=concepts,
        index=['PC1', 'PC2']
    ).T
    
    sns.heatmap(concept_contributions, 
                cmap='RdBu',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Contribution to Principal Components'})
    plt.title('Concept Contributions to Principal Components')
    
    # Create concept distinctiveness distribution
    plt.subplot(2, 2, 4)
    sns.boxplot(data=style_metrics.melt(id_vars=['Style'], 
                                      value_vars=concepts,
                                      var_name='Concept',
                                      value_name='Distinctiveness'))
    plt.xticks(rotation=45)
    plt.title('Distribution of Concept Distinctiveness')
    
    plt.tight_layout()
    plt.savefig('concept_separation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis results
    print("\nConcept Separation Analysis:")
    print("-" * 50)
    
    # Calculate concept independence scores
    correlations = np.corrcoef([
        style_metrics.Color_Distinctiveness,
        style_metrics.Texture_Distinctiveness,
        style_metrics.Composition_Distinctiveness
    ])
    
    print("\nConcept Independence (1 - |correlation|):")
    concept_pairs = [
        ('Color', 'Texture'),
        ('Color', 'Composition'),
        ('Texture', 'Composition')
    ]
    
    for i, (c1, c2) in enumerate(concept_pairs):
        independence = 1 - abs(correlations[i, (i+1)%3])
        print(f"{c1}-{c2}: {independence:.3f}")
    
    # Find most distinctive styles for each concept
    print("\nMost Distinctive Styles per Concept:")
    for concept in concepts:
        top_styles = style_metrics.nlargest(3, concept)[['Style', concept]]
        print(f"\n{concept.split('_')[0]}:")
        for _, row in top_styles.iterrows():
            print(f"  {row['Style']}: {row[concept]:.3f}")

# Run the analysis
analyze_concept_separation()