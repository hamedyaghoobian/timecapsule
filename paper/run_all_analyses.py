#!/usr/bin/env python3
"""
Time Capsule Paper - Complete Analysis Runner
==============================================
ACM C&C 2026

Runs all three modules:
- Module 1: Quantitative Validation (PPL + Tokenizer Fertility)
- Module 2: Diachronic Semantics (Neighbors + Semantic Axis)
- Module 3: Qualitative Probing (Anachronism + Bias Audit)

Usage:
    python run_all_analyses.py              # Run all
    python run_all_analyses.py --module 1   # Run specific module
    python run_all_analyses.py --module 3   # Just run Module 3
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run Time Capsule Paper Analyses")
    parser.add_argument("--module", type=int, choices=[1, 2, 3], 
                       help="Run specific module only (1, 2, or 3)")
    parser.add_argument("--skip-ppl", action="store_true",
                       help="Skip perplexity analysis (requires large models)")
    args = parser.parse_args()
    
    print("="*70)
    print("TIME CAPSULE PAPER - COMPREHENSIVE ANALYSIS")
    print("ACM C&C 2026")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    modules_to_run = [args.module] if args.module else [1, 2, 3]
    
    # Module 1: Quantitative Validation
    if 1 in modules_to_run:
        print("\n" + "#"*70)
        print("# MODULE 1: QUANTITATIVE VALIDATION")
        print("#"*70)
        
        if args.skip_ppl:
            print("Skipping perplexity (--skip-ppl flag)")
            # Just run tokenizer fertility
            from module1_quantitative import run_tokenizer_fertility_analysis
            run_tokenizer_fertility_analysis(output_dir)
        else:
            from module1_quantitative import main as run_module1
            run_module1()
    
    # Module 2: Diachronic Semantics
    if 2 in modules_to_run:
        print("\n" + "#"*70)
        print("# MODULE 2: DIACHRONIC SEMANTICS")
        print("#"*70)
        from module2_diachronic import main as run_module2
        run_module2()
    
    # Module 3: Qualitative Probing
    if 3 in modules_to_run:
        print("\n" + "#"*70)
        print("# MODULE 3: QUALITATIVE PROBING")
        print("#"*70)
        from module3_qualitative import main as run_module3
        run_module3()
    
    # Final summary
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"\n📁 Output Directory: {output_dir.absolute()}")
    print("\n📋 Generated Files:")
    
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            size = f.stat().st_size
            print(f"   • {f.name} ({size:,} bytes)")
    
    print("\n🎯 Key outputs for paper:")
    print("   • perplexity_results.json - Fig 1: PPL comparison")
    print("   • tokenizer_fertility_results.json - Tokenizer efficiency ratio")
    print("   • neighbor_comparison.csv - Table 1: Semantic neighbors")
    print("   • semantic_axis_results.json - Fig 2: TIME on Nature→Factory axis")
    print("   • anachronism_probe_quotes.txt - Box 1: Hallucination examples")
    print("   • bias_audit_quotes.txt - Box 2: Archival honesty examples")


if __name__ == "__main__":
    main()

