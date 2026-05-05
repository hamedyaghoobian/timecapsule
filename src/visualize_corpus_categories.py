"""
Create publication-quality visualizations of corpus categorization results.

Generates figures suitable for the ACM Creativity and Cognition paper showing
the distribution of document types in the 1800-1875 Internet Archive corpus.

Author: Automated Analysis
Date: February 2026
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Use colorblind-friendly palette
COLORS = sns.color_palette("colorblind", 15)


def load_statistics(json_path: str = "outputs/corpus_analysis/corpus_statistics.json"):
    """Load categorization statistics from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_bar_chart(stats: dict, output_path: Path):
    """Create a horizontal bar chart of category percentages."""
    categories = []
    percentages = []
    counts = []
    
    for cat, cat_stats in sorted(
        stats['categories'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    ):
        categories.append(cat)
        percentages.append(cat_stats['percentage'])
        counts.append(cat_stats['count'])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bars = ax.barh(categories, percentages, color=COLORS[:len(categories)])
    
    # Add percentage labels on bars
    for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts)):
        width = bar.get_width()
        label = f'{pct:.1f}%'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                label, ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Percentage of Corpus (%)', fontweight='bold')
    ax.set_title('Distribution of Document Types in 1800-1875 Corpus', 
                 fontweight='bold', pad=20)
    ax.set_xlim(0, max(percentages) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / "category_distribution_bar.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "category_distribution_bar.pdf", bbox_inches='tight')
    print(f"✓ Saved bar chart: {output_path / 'category_distribution_bar.png'}")
    plt.close()


def create_pie_chart(stats: dict, output_path: Path, top_n: int = 10):
    """Create a pie chart showing major categories."""
    categories = []
    percentages = []
    
    sorted_cats = sorted(
        stats['categories'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    
    # Top N categories plus "Other"
    for i, (cat, cat_stats) in enumerate(sorted_cats):
        if i < top_n:
            categories.append(cat)
            percentages.append(cat_stats['percentage'])
        elif i == top_n:
            categories.append('Other')
            percentages.append(cat_stats['percentage'])
        else:
            percentages[-1] += cat_stats['percentage']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        percentages, 
        labels=categories,
        autopct='%1.1f%%',
        startangle=90,
        colors=COLORS[:len(categories)],
        pctdistance=0.85
    )
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    for text in texts:
        text.set_fontsize(10)
    
    ax.set_title('Document Type Distribution (Top 10 Categories)', 
                 fontweight='bold', pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path / "category_distribution_pie.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "category_distribution_pie.pdf", bbox_inches='tight')
    print(f"✓ Saved pie chart: {output_path / 'category_distribution_pie.png'}")
    plt.close()


def create_stacked_bar_by_size(stats: dict, output_path: Path):
    """Create visualization showing category distribution by data size."""
    categories = []
    sizes_mb = []
    percentages = []
    
    for cat, cat_stats in sorted(
        stats['categories'].items(),
        key=lambda x: x[1]['total_size_mb'],
        reverse=True
    ):
        categories.append(cat)
        sizes_mb.append(cat_stats['total_size_mb'])
        percentages.append(cat_stats['percentage'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Size distribution
    bars1 = ax1.barh(categories, sizes_mb, color=COLORS[:len(categories)])
    ax1.set_xlabel('Total Size (MB)', fontweight='bold')
    ax1.set_title('Data Volume by Category', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add size labels
    for bar, size in zip(bars1, sizes_mb):
        width = bar.get_width()
        ax1.text(width + max(sizes_mb)*0.01, bar.get_y() + bar.get_height()/2,
                f'{size:.0f}', ha='left', va='center', fontsize=8)
    
    # Document count vs size comparison
    bars2 = ax2.barh(categories, percentages, color=COLORS[:len(categories)])
    ax2.set_xlabel('Percentage of Documents (%)', fontweight='bold')
    ax2.set_title('Document Count by Category', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        width = bar.get_width()
        ax2.text(width + max(percentages)*0.01, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / "category_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "category_comparison.pdf", bbox_inches='tight')
    print(f"✓ Saved comparison chart: {output_path / 'category_comparison.png'}")
    plt.close()


def create_summary_table(stats: dict, output_path: Path):
    """Create a formatted table image for the paper."""
    data = []
    
    for cat, cat_stats in sorted(
        stats['categories'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    ):
        data.append([
            cat,
            f"{cat_stats['count']:,}",
            f"{cat_stats['percentage']:.2f}%",
            f"{cat_stats['total_size_mb']:.1f}"
        ])
    
    # Add total row
    total_size = sum(c['total_size_mb'] for c in stats['categories'].values())
    data.append([
        'TOTAL',
        f"{stats['total_files']:,}",
        '100.00%',
        f"{total_size:.1f}"
    ])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=data,
        colLabels=['Category', 'Documents', 'Percentage', 'Size (MB)'],
        cellLoc='left',
        loc='center',
        colWidths=[0.4, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == len(data) - 1:  # Total row
                cell.set_facecolor('#D9E2F3')
                cell.set_text_props(weight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
    
    plt.title('Corpus Categorization Summary (1800-1875)', 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / "category_table.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "category_table.pdf", bbox_inches='tight')
    print(f"✓ Saved summary table: {output_path / 'category_table.png'}")
    plt.close()


def create_text_summary(stats: dict, output_path: Path):
    """Create a text summary suitable for copying into the paper."""
    summary_path = output_path / "paper_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CORPUS CATEGORIZATION SUMMARY FOR ACM PAPER\n")
        f.write("1800-1875 Internet Archive Historical Text Corpus\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Documents: {stats['total_files']:,}\n\n")
        
        f.write("Document Type Distribution:\n")
        f.write("-" * 70 + "\n\n")
        
        sorted_cats = sorted(
            stats['categories'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )
        
        for cat, cat_stats in sorted_cats:
            f.write(f"• {cat}: {cat_stats['percentage']:.2f}% ")
            f.write(f"({cat_stats['count']:,} documents, ")
            f.write(f"{cat_stats['total_size_mb']:.1f} MB)\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        # Key findings for paper
        f.write("KEY FINDINGS FOR PAPER:\n\n")
        
        top_3 = sorted_cats[:3]
        f.write(f"The corpus is dominated by {top_3[0][0]} ({top_3[0][1]['percentage']:.1f}%), ")
        f.write(f"{top_3[1][0]} ({top_3[1][1]['percentage']:.1f}%), and ")
        f.write(f"{top_3[2][0]} ({top_3[2][1]['percentage']:.1f}%). ")
        
        fiction_pct = stats['categories'].get('Literary Fiction', {}).get('percentage', 0)
        f.write(f"Literary Fiction comprises {fiction_pct:.1f}% of the corpus. ")
        
        periodicals_pct = stats['categories'].get('Periodicals/Journalism', {}).get('percentage', 0)
        f.write(f"Periodicals and Journalism account for {periodicals_pct:.1f}%. ")
        
        parliament_pct = stats['categories'].get('Parliamentary/Legal', {}).get('percentage', 0)
        f.write(f"Parliamentary and Legal documents represent {parliament_pct:.1f}% ")
        f.write("of the corpus, making it the largest single category.\n\n")
        
        # Calculate cultural content
        cultural_categories = ['Literary Fiction', 'Poetry', 'Drama/Theatre']
        cultural_pct = sum(
            stats['categories'].get(cat, {}).get('percentage', 0) 
            for cat in cultural_categories
        )
        f.write(f"Cultural/Literary content (Fiction, Poetry, Drama) totals {cultural_pct:.1f}%.\n")
        
        # Calculate scholarly content
        scholarly_categories = ['Science/Natural Philosophy', 'Philosophy', 
                                'Medical', 'History', 'Reference/Educational']
        scholarly_pct = sum(
            stats['categories'].get(cat, {}).get('percentage', 0) 
            for cat in scholarly_categories
        )
        f.write(f"Scholarly/Academic content totals {scholarly_pct:.1f}%.\n\n")
        
        total_size = sum(c['total_size_mb'] for c in stats['categories'].values())
        f.write(f"Total corpus size: {total_size/1024:.2f} GB\n")
        
    print(f"✓ Saved text summary: {summary_path}")


def main():
    """Generate all visualizations and summaries."""
    # Load statistics
    stats_path = "outputs/corpus_analysis/corpus_statistics.json"
    
    if not Path(stats_path).exists():
        print(f"Error: Statistics file not found: {stats_path}")
        print("Please run categorize_corpus.py first!")
        return
    
    stats = load_statistics(stats_path)
    
    # Create output directory for visualizations
    output_path = Path("outputs/corpus_analysis/visualizations")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations for ACM paper...\n")
    
    # Create all visualizations
    create_bar_chart(stats, output_path)
    create_pie_chart(stats, output_path)
    create_stacked_bar_by_size(stats, output_path)
    create_summary_table(stats, output_path)
    create_text_summary(stats, output_path)
    
    print(f"\n{'='*70}")
    print("VISUALIZATION GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll figures saved to: {output_path}")
    print("\nGenerated files:")
    print("  📊 category_distribution_bar.png/pdf - Horizontal bar chart")
    print("  🥧 category_distribution_pie.png/pdf - Pie chart (top 10)")
    print("  📈 category_comparison.png/pdf - Size vs count comparison")
    print("  📋 category_table.png/pdf - Formatted summary table")
    print("  📝 paper_summary.txt - Text summary for paper")
    print("\nAll figures are publication-quality (300 DPI) and available in")
    print("both PNG and PDF formats for your ACM Creativity & Cognition paper!")


if __name__ == "__main__":
    main()
