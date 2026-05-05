"""
Expert Confusion Analysis: Cohen's Kappa and Visualization
Analyzes expert evaluation data to calculate inter-rater agreement
and visualize the confusion in distinguishing real vs. generated text.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Load the data
df = pd.read_csv('HistoricalTextEvaluationStudy.csv')

# Extract origin assessment columns (Q1 for each sample)
origin_columns = [col for col in df.columns if 'Q1: Origin Assessment' in col]
print(f"Found {len(origin_columns)} text samples evaluated by experts\n")

# Extract responses from both experts
expert1_responses = []
expert2_responses = []

for col in origin_columns:
    expert1_responses.append(df.loc[0, col])
    expert2_responses.append(df.loc[1, col])

# Anonymize experts with their expertise areas
expert1_name = "Expert A"
expert1_expertise = df.loc[0, 'Your Area of Expertise (e.g., Victorian Literature, 19th Century History)']
expert2_name = "Expert B"
expert2_expertise = df.loc[1, 'Your Area of Expertise (e.g., Victorian Literature, 19th Century History)']

print(f"Expert A: {expert1_expertise}")
print(f"Expert B: {expert2_expertise}")
print()

# Load ground truth from answer key
import json
with open('outputs/expert_eval_answer_key.json', 'r') as f:
    answer_key = json.load(f)

# Extract ground truth labels in order
ground_truth_labels = [item['type'].capitalize() for item in answer_key]
print(f"Ground truth loaded: {ground_truth_labels.count('Real')} Real, {ground_truth_labels.count('Generated')} Generated\n")

# Convert responses to binary (0 = Machine, 1 = Human)
def response_to_binary(response):
    return 1 if response == 'Human Author' else 0

expert1_binary = [response_to_binary(r) for r in expert1_responses]
expert2_binary = [response_to_binary(r) for r in expert2_responses]
ground_truth_binary = [1 if gt == 'Real' else 0 for gt in ground_truth_labels]

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(expert1_binary, expert2_binary)
print(f"Cohen's Kappa (Inter-rater Agreement): {kappa:.3f}")

# Interpret kappa
if kappa < 0:
    interpretation = "Poor agreement (worse than chance)"
elif kappa < 0.20:
    interpretation = "Slight agreement"
elif kappa < 0.40:
    interpretation = "Fair agreement"
elif kappa < 0.60:
    interpretation = "Moderate agreement"
elif kappa < 0.80:
    interpretation = "Substantial agreement"
else:
    interpretation = "Almost perfect agreement"

print(f"Interpretation: {interpretation}\n")

# Calculate agreement and disagreement
agreements = sum(e1 == e2 for e1, e2 in zip(expert1_binary, expert2_binary))
disagreements = len(expert1_binary) - agreements
print(f"Raw Agreement: {agreements}/{len(expert1_binary)} ({100*agreements/len(expert1_binary):.1f}%)")
print(f"Disagreements: {disagreements}/{len(expert1_binary)} ({100*disagreements/len(expert1_binary):.1f}%)\n")

# Calculate confusion metrics for each expert
def calculate_confusion_metrics(predictions, ground_truth, expert_name):
    """Calculate and print confusion metrics"""
    cm = confusion_matrix(ground_truth, predictions)
    
    # cm[i,j] where i=true label, j=predicted label
    # [0,0] = TN (correctly identified machine)
    # [0,1] = FP (said human, was machine)
    # [1,0] = FN (said machine, was human)
    # [1,1] = TP (correctly identified human)
    
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Key metrics for "The Confusion of Expertise"
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # Accepted generated as real
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # Rejected real as generated
    
    print(f"\n{expert_name} Performance:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  True Positives (Correctly identified real): {tp}")
    print(f"  True Negatives (Correctly identified generated): {tn}")
    print(f"  False Positives (Accepted generated as real): {fp}")
    print(f"  False Negatives (Rejected real as generated): {fn}")
    print(f"  False Positive Rate: {false_positive_rate:.1%}")
    print(f"  False Negative Rate: {false_negative_rate:.1%}")
    
    return {
        'cm': cm,
        'accuracy': accuracy,
        'fpr': false_positive_rate,
        'fnr': false_negative_rate,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

metrics1 = calculate_confusion_metrics(expert1_binary, ground_truth_binary, f"{expert1_name} ({expert1_expertise})")
metrics2 = calculate_confusion_metrics(expert2_binary, ground_truth_binary, f"{expert2_name} ({expert2_expertise})")

# Calculate average metrics
avg_fpr = (metrics1['fpr'] + metrics2['fpr']) / 2
avg_fnr = (metrics1['fnr'] + metrics2['fnr']) / 2

print(f"\n{'='*60}")
print(f"THE CONFUSION OF EXPERTISE")
print(f"{'='*60}")
print(f"Average False Positive Rate (Accepted Generated as Real): {avg_fpr:.1%}")
print(f"Average False Negative Rate (Rejected Real as Generated): {avg_fnr:.1%}")
print(f"Ratio (FNR/FPR): {avg_fnr/avg_fpr if avg_fpr > 0 else 'N/A':.2f}")
print(f"\nExperts were nearly as likely to reject authentic texts")
print(f"as they were to accept generated texts.")

# Create simplified, beautiful visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Use a more elegant color scheme
colors = {'accept': '#e8743b', 'reject': '#19aade'}  # Warmer orange, cooler blue

# Left panel: Error comparison
# Shorten expertise descriptions elegantly
expert1_short = "Linguistics & Rhetoric"
expert2_short = "Victorian Literature"

experts_display = [f'{expert1_name}\n({expert1_short})', 
                   f'{expert2_name}\n({expert2_short})']

x = np.arange(len(experts_display))
width = 0.38

fpr_values = [metrics1['fpr'] * 100, metrics2['fpr'] * 100]
fnr_values = [metrics1['fnr'] * 100, metrics2['fnr'] * 100]

bars1 = ax1.bar(x - width/2, fpr_values, width, label='Accepted Generated as Real', 
                color=colors['accept'], alpha=0.85, edgecolor='white', linewidth=2)
bars2 = ax1.bar(x + width/2, fnr_values, width, label='Rejected Real as Generated', 
                color=colors['reject'], alpha=0.85, edgecolor='white', linewidth=2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Error Rate (%)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(experts_display, fontsize=11)
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10, edgecolor='gray')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, max(max(fpr_values), max(fnr_values)) * 1.18)
ax1.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)

# Add κ annotation
textstr = f"Cohen's κ = {kappa:.3f}\n({interpretation})"
props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='#666666', linewidth=1.5)
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# Right panel: Combined confusion matrix
combined_cm = metrics1['cm'] + metrics2['cm']

sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Oranges', 
            ax=ax2, cbar_kws={'label': 'Count (N=40 total)', 'shrink': 0.8}, 
            linewidths=2.5, linecolor='white', square=True,
            annot_kws={'size': 14, 'weight': 'bold'})

ax2.set_xlabel('Expert Judgment', fontsize=12, fontweight='bold')
ax2.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Generated', 'Real'], rotation=0, fontsize=11)
ax2.set_yticklabels(['Generated', 'Real'], rotation=0, fontsize=11)

# Add percentage annotations
for i in range(2):
    for j in range(2):
        count = combined_cm[i, j]
        pct = 100 * count / combined_cm.sum()
        ax2.text(j + 0.5, i + 0.75, f'({pct:.1f}%)', 
                ha='center', va='top', fontsize=9, color='#333333', style='italic')

plt.tight_layout()
plt.savefig('outputs/figures/confusion_of_expertise.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
print("\n" + "="*60)
print("Visualization saved to: outputs/figures/confusion_of_expertise.pdf")
print("="*60)

plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY FOR PAPER")
print("="*60)
print(f"Inter-rater Reliability: Cohen's κ = {kappa:.3f} ({interpretation})")
print(f"Raw Agreement: {100*agreements/len(expert1_binary):.1f}%")
print(f"\nCombined Performance (N=40 judgments):")
print(f"  Correctly Identified Generated: {metrics1['tn'] + metrics2['tn']}/28 ({100*(metrics1['tn'] + metrics2['tn'])/28:.1f}%)")
print(f"  Correctly Identified Real: {metrics1['tp'] + metrics2['tp']}/12 ({100*(metrics1['tp'] + metrics2['tp'])/12:.1f}%)")
print(f"  False Accepts (Gen→Real): {metrics1['fp'] + metrics2['fp']}/28 ({100*(metrics1['fp'] + metrics2['fp'])/28:.1f}%)")
print(f"  False Rejects (Real→Gen): {metrics1['fn'] + metrics2['fn']}/12 ({100*(metrics1['fn'] + metrics2['fn'])/12:.1f}%)")
print(f"\nKey Finding:")
print(f"  Experts falsely accepted generated text at a rate of {avg_fpr:.1%}")
print(f"  Experts falsely rejected real text at a rate of {avg_fnr:.1%}")
print(f"  Ratio: {avg_fnr/avg_fpr if avg_fpr > 0 else 0:.2f} (near 1.0 indicates symmetric confusion)")
print("="*60)
