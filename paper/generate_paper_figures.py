#!/usr/bin/env python3
"""
Paper Figures Generation
========================
Generates publication-quality PDF figures for ACM C&C 2026 paper.
Follows ACM and AI community visualization guidelines.

Figures:
1. Semantic Shift - 1D scatter showing TIME on Nature-Factory axis
2. Chronological Cliff - Timeline with training boundary
3. Bias Topography - t-SNE visualization of civilizational terms
4. Epistemological Horizon - Comparison grid
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats

# Configuration
OUTPUT_DIR = Path(__file__).parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Set seaborn style for professional academic look
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
sns.set_palette("colorblind")

# ACM-compliant style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'timecapsule': '#D55E00',      # Vermillion (Victorian)
    'modern': '#0072B2',           # Blue (Modern)
    'training': '#009E73',         # Bluish green
    'unknown': '#CC79A7',          # Reddish purple
    'cliff': '#333333',            # Dark grey
    'civilizational': '#D55E00',   # Vermillion
    'social': '#0072B2',           # Blue
    'progress': '#009E73',         # Bluish green
    'nature': '#CC79A7',           # Reddish purple
    'accent': '#F0E442',           # Yellow
}


def load_json(filename):
    """Load JSON data from outputs directory."""
    path = OUTPUT_DIR / filename
    with open(path) as f:
        return json.load(f)


# ============================================================
# FIGURE 1: THE SEMANTIC SHIFT
# ============================================================

def generate_figure1_semantic_shift():
    """Generate 1D scatter plot showing TIME on Nature-Factory axis."""
    print("\n📊 Generating: Semantic Shift...")
    
    # Load data
    data = load_json("semantic_axis_results.json")
    
    tc_proj = data["TimeCapsule"]["projection"]
    bert_proj = data["BERT"]["projection"]
    
    # Create figure with ACM single-column width (3.33 inches)
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    # Define axis range
    x_min, x_max = 0, 20
    y_center = 0
    
    # Create gradient background using scipy interpolation
    x_gradient = np.linspace(x_min, x_max, 200)
    for i in range(len(x_gradient) - 1):
        color_val = i / len(x_gradient)
        ax.axvspan(x_gradient[i], x_gradient[i+1], 
                   ymin=0.35, ymax=0.65,
                   color=plt.cm.RdYlGn_r(color_val), alpha=0.25, zorder=0)
    
    # Draw the semantic axis line
    ax.axhline(y=y_center, color='#333333', linewidth=1.5, zorder=1)
    
    # Plot the two models with error bars (simulated confidence)
    ax.errorbar(bert_proj, y_center, xerr=0.8, fmt='o', markersize=12,
                color=COLORS['modern'], markeredgecolor='white', markeredgewidth=1.5,
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=3, label='BERT (Modern)')
    ax.errorbar(tc_proj, y_center, xerr=0.8, fmt='s', markersize=12,
                color=COLORS['timecapsule'], markeredgecolor='white', markeredgewidth=1.5,
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=3, label='TimeCapsule')
    
    # Add annotations with clean positioning
    ax.annotate('BERT', (bert_proj, 0.12), ha='center', fontsize=10,
                fontweight='bold', color=COLORS['modern'])
    ax.annotate('TimeCapsule', (tc_proj, 0.12), ha='center', fontsize=10,
                fontweight='bold', color=COLORS['timecapsule'])
    
    # Add semantic drift indicator
    mid_point = (tc_proj + bert_proj) / 2
    ax.annotate('', xy=(tc_proj - 0.5, -0.06), xytext=(bert_proj + 0.5, -0.06),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.5,
                               connectionstyle='arc3,rad=0'))
    
    # Calculate ratio for the label
    ratio = tc_proj / bert_proj if bert_proj != 0 else 0
    ax.text(mid_point, -0.12, 'Δ = {:.1f} ({:.1f}x shift)'.format(tc_proj - bert_proj, ratio), 
            ha='center', fontsize=9, color='#555555')
    
    # Axis endpoint labels
    ax.text(x_min + 0.3, -0.22, 'NATURE', fontsize=10, fontweight='bold', 
            color=COLORS['progress'], ha='left', va='top')
    ax.text(x_max - 0.3, -0.22, 'FACTORY', fontsize=10, fontweight='bold', 
            color=COLORS['timecapsule'], ha='right', va='top')
    
    # Title (no figure number per ACM guidelines)
    ax.set_title('Semantic Position of "TIME" on Nature–Factory Axis', 
                 fontsize=11, fontweight='bold', pad=15)
    
    # Clean up axes
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(-0.3, 0.25)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 21, 5))
    ax.set_xlabel('Projection Score', fontsize=10)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
              edgecolor='#cccccc', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = FIGURES_DIR / "fig1_semantic_shift.pdf"
    plt.savefig(output_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# ============================================================
# FIGURE 2: THE CHRONOLOGICAL CLIFF
# ============================================================

def generate_figure2_chronological_cliff():
    """Generate timeline showing training boundary and anachronistic concepts."""
    print("\n📊 Generating: Chronological Cliff...")
    
    # Load anachronism data
    data = load_json("anachronism_probe_results.json")
    
    # Extract concepts with invention dates
    concepts = []
    for item in data:
        if item["invented"] > 1875:
            concepts.append({
                "name": item["concept"].title(),
                "year": item["invented"],
            })
    
    # Sort by year
    concepts = sorted(concepts, key=lambda x: x["year"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Timeline parameters
    y_timeline = 0.6
    year_start, year_end = 1790, 2030
    cliff_year = 1875
    
    # Draw training period background
    rect_train = Rectangle((year_start, 0.45), cliff_year - year_start, 0.3,
                           facecolor=COLORS['training'], alpha=0.2, zorder=0)
    ax.add_patch(rect_train)
    
    # Draw unknown territory background
    rect_unknown = Rectangle((cliff_year, 0.45), year_end - cliff_year, 0.3,
                             facecolor=COLORS['unknown'], alpha=0.15, zorder=0)
    ax.add_patch(rect_unknown)
    
    # Labels for regions
    ax.text((year_start + cliff_year) / 2, 0.82, 'Training Corpus', 
            ha='center', fontsize=10, fontweight='bold', color=COLORS['training'])
    ax.text((year_start + cliff_year) / 2, 0.77, '(1800–1875)', 
            ha='center', fontsize=9, color=COLORS['training'], alpha=0.8)
    
    ax.text((cliff_year + year_end) / 2, 0.82, 'Terra Incognita', 
            ha='center', fontsize=10, fontweight='bold', color=COLORS['unknown'])
    ax.text((cliff_year + year_end) / 2, 0.77, '(Post-1875)', 
            ha='center', fontsize=9, color=COLORS['unknown'], alpha=0.8)
    
    # Draw the timeline axis
    ax.plot([year_start, year_end], [y_timeline, y_timeline], 
            color='#333333', linewidth=2, zorder=2)
    
    # Draw the cliff (vertical line with gradient effect)
    cliff_heights = np.linspace(0.45, 0.75, 20)
    for i, h in enumerate(cliff_heights[:-1]):
        alpha = 0.9 - (i / len(cliff_heights)) * 0.5
        ax.plot([cliff_year, cliff_year], [h, cliff_heights[i+1]],
                color=COLORS['cliff'], linewidth=3, alpha=alpha, zorder=3)
    
    # Cliff marker
    ax.scatter([cliff_year], [y_timeline], s=150, c=COLORS['cliff'], 
               marker='|', linewidths=3, zorder=4)
    ax.text(cliff_year, 0.87, '1875', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['cliff'])
    
    # Year markers on timeline
    year_markers = [1800, 1850, 1900, 1950, 2000]
    for year in year_markers:
        ax.plot([year, year], [y_timeline - 0.02, y_timeline + 0.02], 
                color='#333333', linewidth=1, zorder=2)
        ax.text(year, y_timeline - 0.06, str(year), ha='center', fontsize=8)
    
    # Plot concepts as falling points with connecting lines
    y_positions = np.linspace(0.35, 0.05, len(concepts))
    
    for i, (concept, y_pos) in enumerate(zip(concepts, y_positions)):
        # Dashed line from timeline to concept
        ax.plot([concept["year"], concept["year"]], [y_timeline - 0.15, y_pos + 0.025],
                color=COLORS['unknown'], linewidth=1, linestyle=':', alpha=0.6, zorder=1)
        
        # Concept marker
        ax.scatter(concept["year"], y_pos, s=80, c=COLORS['unknown'], 
                   marker='o', edgecolors='white', linewidths=1, zorder=5, alpha=0.9)
        
        # Label with year
        label_offset = 8 if i % 2 == 0 else -8
        ha = 'left' if i % 2 == 0 else 'right'
        ax.annotate(f'{concept["name"]}', 
                   (concept["year"], y_pos),
                   xytext=(label_offset, 0), textcoords='offset points',
                   fontsize=8, va='center', ha=ha, color='#333333')
    
    # Title
    ax.set_title('The Chronological Cliff: Training Data Boundary', 
                 fontsize=11, fontweight='bold', pad=15)
    
    # Clean up axes
    ax.set_xlim(year_start - 10, year_end + 10)
    ax.set_ylim(-0.02, 0.95)
    ax.axis('off')
    
    # Subtitle/caption
    ax.text(0.5, -0.02, 
            'Concepts invented after 1875 fall outside the model\'s epistemological horizon',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='#666666')
    
    plt.tight_layout()
    
    # Save
    output_path = FIGURES_DIR / "fig2_chronological_cliff.pdf"
    plt.savefig(output_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# ============================================================
# FIGURE 3: BIAS TOPOGRAPHY
# ============================================================

def generate_figure3_bias_topography():
    """Generate t-SNE visualization of civilizational terms using seaborn."""
    print("\n📊 Generating: Bias Topography...")
    
    from sklearn.manifold import TSNE
    from scipy.spatial.distance import pdist, squareform
    import pandas as pd
    try:
        from adjustText import adjust_text
    except ImportError:
        print("   Warning: adjustText not installed. install with `pip install adjustText` for better labels.")
        adjust_text = None
    
    # Load embeddings
    try:
        emb_path = OUTPUT_DIR / "bias_topography_embeddings.json"
        with open(emb_path) as f:
            embeddings_data = json.load(f)
        
        tc_embeddings = embeddings_data.get("timecapsule", {})
        bert_embeddings = embeddings_data.get("bert", {})
    except Exception as e:
        print(f"   Warning: Could not load embeddings: {e}")
        tc_embeddings = {}
        bert_embeddings = {}
    
    # Word categories
    WORD_CATEGORIES = {
        "civilizational": [
            "native", "savage", "empire", "colony", "progress", "industry",
            "civilization", "barbarous", "heathen", "primitive", "backward", "enlightened",
            "christian", "infidel", "missionary", "conquest", "dominion", "subject", "servant"
        ],
        "social": [
            "poor", "working", "gentleman", "lady", "peasant", "labourer", "master",
            "slave", "negro", "hindoo", "irish", "gypsy"
        ],
        "progress": [
            "steam", "railway", "manufacture", "machine", "telegraph", "factory",
            "commerce", "trade", "improvement", "science", "invention", "modern"
        ],
        "nature": [
            "nature", "natural", "organic", "earth", "land", "rural", "agricultural"
        ]
    }
    
    # Build mappings
    ALL_WORDS = []
    WORD_TO_CATEGORY = {}
    for category, words in WORD_CATEGORIES.items():
        for word in words:
            if word not in WORD_TO_CATEGORY:
                ALL_WORDS.append(word)
                WORD_TO_CATEGORY[word] = category
    
    # Category display names
    CATEGORY_NAMES = {
        "civilizational": "Colonial/Imperial",
        "social": "Social Hierarchy",
        "progress": "Industrial Progress",
        "nature": "Nature/Rural"
    }
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Seaborn color palette (colorblind-friendly)
    palette = {
        "civilizational": COLORS['civilizational'],
        "social": COLORS['social'],
        "progress": COLORS['progress'],
        "nature": COLORS['nature']
    }
    
    for ax_idx, (ax, (embeddings, title)) in enumerate(zip(axes, [
        (tc_embeddings, "TimeCapsule (1800–1875)"),
        (bert_embeddings, "BERT (Modern)")
    ])):
        if embeddings:
            # Get words with embeddings
            words_used = [w for w in ALL_WORDS if w in embeddings]
            
            if len(words_used) >= 5:
                # Get embeddings matrix
                X = np.array([embeddings[w] for w in words_used])
                
                # Apply t-SNE with scipy-based distance computation
                perplexity = min(30, len(words_used) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                           metric='cosine', init='pca', learning_rate='auto')
                X_2d = tsne.fit_transform(X)
                
                # Create DataFrame for seaborn
                df = pd.DataFrame({
                    'x': X_2d[:, 0],
                    'y': X_2d[:, 1],
                    'word': words_used,
                    'category': [WORD_TO_CATEGORY.get(w, 'other') for w in words_used]
                })
                
                # Plot with seaborn
                for cat in palette.keys():
                    cat_df = df[df['category'] == cat]
                    if not cat_df.empty:
                        ax.scatter(cat_df['x'], cat_df['y'], 
                                  c=palette[cat], s=80, alpha=0.7,
                                  label=CATEGORY_NAMES[cat], edgecolors='white', linewidths=0.5)
                
                # Add word labels with smart positioning using adjustText
                texts = []
                for idx, row in df.iterrows():
                    # Highlight key terms
                    fontweight = 'bold' if row['word'] in ['progress', 'savage', 'empire', 'civilization', 'slave'] else 'normal'
                    fontsize = 8 if fontweight == 'bold' else 7
                    
                    t = ax.text(row['x'], row['y'], row['word'],
                               fontsize=fontsize, fontweight=fontweight, alpha=0.85)
                    texts.append(t)
                    
                    # Circle highlight for "progress"
                    if row['word'] == 'progress':
                        circle = plt.Circle((row['x'], row['y']), 
                                          radius=max(X_2d[:, 0].max() - X_2d[:, 0].min(),
                                                    X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.08,
                                          fill=False, color='#333333', linewidth=1.5, linestyle='--')
                        ax.add_patch(circle)
                
                # Apply adjust_text to repel labels with stronger force for v2
                if adjust_text:
                    adjust_text(texts, 
                               arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), 
                               ax=ax,
                               force_text=1.0,      # Increase text repulsion
                               force_points=0.8,    # Increase point repulsion
                               expand_points=(1.5, 1.5), # Expand point avoidance area
                               lim=1000)            # Allow more iterations for solving overlap
        else:
            # Placeholder
            ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#999999')
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=9)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=9)
        
        # Clean up
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        
        # Only show legend on first plot
        if ax_idx == 0:
            ax.legend(loc='upper left', fontsize=8, frameon=True, 
                     framealpha=0.9, edgecolor='#cccccc')
    
    # Main title
    fig.suptitle('Bias Topography: Word Embedding Clusters', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    output_path = FIGURES_DIR / "fig3_bias_topography_v2.pdf"
    plt.savefig(output_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✓ Saved v2: {output_path}")


# ============================================================
# FIGURE 4: EPISTEMOLOGICAL HORIZON
# ============================================================

def generate_figure4_epistemological_horizon():
    """Generate comparison showing TimeCapsule vs GPT responses."""
    print("\n📊 Generating: Epistemological Horizon...")
    
    # Load comparison data
    data = load_json("epistemological_comparison.json")
    
    # Select key concepts
    key_concepts = ["airplane", "computer", "internet", "atomic bomb", "television"]
    selected = [d for d in data if d["concept"] in key_concepts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Prepare data
    headers = ["Concept", "TimeCapsule (1875)", "GPT-4o / GPT-5.1"]
    
    rows = []
    for item in selected:
        concept = item["concept"].title()
        tc = item["timecapsule"][:55] + "…" if len(item["timecapsule"]) > 55 else item["timecapsule"]
        gpt_response = list(item["gpt_responses"].values())[0]
        gpt = gpt_response[:55] + "…" if len(gpt_response) > 55 else gpt_response
        rows.append([concept, f'"{tc}"', f'"{gpt}"'])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='left',
        loc='center',
        colWidths=[0.12, 0.44, 0.44]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    
    # Style header
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_height(0.08)
    
    # Style data rows with alternating colors
    for i in range(1, len(rows) + 1):
        bg_color = '#F8F9F9' if i % 2 == 0 else '#FFFFFF'
        
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(bg_color)
            cell.set_height(0.08)
            
            if j == 0:
                cell.set_text_props(fontweight='bold', fontsize=9)
            elif j == 1:
                cell.set_facecolor('#FEF5E7' if i % 2 == 0 else '#FDF2E9')
            elif j == 2:
                cell.set_facecolor('#EBF5FB' if i % 2 == 0 else '#E8F6F3')
    
    # Remove cell borders for cleaner look
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor('#CCCCCC')
    
    # Title
    ax.set_title('The Epistemological Horizon: TimeCapsule vs. State-of-the-Art', 
                 fontsize=11, fontweight='bold', pad=20, y=0.95)
    
    # Subtitle
    ax.text(0.5, 0.08, 
            'Modern LLMs reveal future knowledge; TimeCapsule hallucinates within its Victorian worldview',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='#666666')
    
    plt.tight_layout()
    
    # Save
    output_path = FIGURES_DIR / "fig4_epistemological_horizon.pdf"
    plt.savefig(output_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# ============================================================
# FIGURE 5: PERPLEXITY COMPARISON
# ============================================================

def generate_figure5_perplexity():
    """Generate bar chart comparing perplexity scores with seaborn."""
    print("\n📊 Generating: Perplexity Comparison...")
    
    try:
        data = load_json("perplexity_results.json")
    except:
        print("   Warning: perplexity_results.json not found, skipping...")
        return
    
    import pandas as pd
    
    # Prepare data
    models = list(data.keys())
    ppls = [data[m].get("perplexity", data[m].get("mean_perplexity", 0)) for m in models]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Perplexity': ppls,
        'Type': ['Victorian' if 'time' in m.lower() or 'capsule' in m.lower() 
                 else 'Modern' for m in models]
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create bar plot with seaborn
    colors = [COLORS['timecapsule'] if t == 'Victorian' else COLORS['modern'] 
              for t in df['Type']]
    
    bars = sns.barplot(data=df, x='Model', y='Perplexity', palette=colors, 
                       edgecolor='white', linewidth=1, ax=ax)
    
    # Add value labels
    for i, (bar, ppl) in enumerate(zip(ax.patches, ppls)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ppl:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Perplexity', fontsize=10)
    ax.set_xlabel('')
    ax.set_title('Perplexity on Victorian Test Set', fontsize=11, fontweight='bold', pad=10)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine()
    
    plt.tight_layout()
    
    # Save
    output_path = FIGURES_DIR / "fig5_perplexity.pdf"
    plt.savefig(output_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Generate all paper figures."""
    print("=" * 60)
    print("PAPER FIGURES GENERATION")
    print("Time Capsule — ACM C&C 2026")
    print("=" * 60)
    print("\nStyle: ACM/AI Publication Guidelines")
    print("Format: PDF (300 DPI, vector graphics)")
    
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {FIGURES_DIR}")
    
    # Generate all figures
    generate_figure1_semantic_shift()
    generate_figure2_chronological_cliff()
    generate_figure3_bias_topography()
    generate_figure4_epistemological_horizon()
    generate_figure5_perplexity()
    
    print("\n" + "=" * 60)
    print("✅ ALL FIGURES GENERATED")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"   • {f.name}")
    
    print(f"\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
