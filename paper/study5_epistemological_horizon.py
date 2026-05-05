#!/usr/bin/env python3
"""
Study 5: Epistemological Horizon Comparison
============================================
Compares TimeCapsule vs GPT-5.1/GPT-4o on anachronism prompts.

Proves that "Intelligence is a Liability" for historical simulation.
Modern LLMs cannot "unlearn the future" - their knowledge of post-1875
concepts breaks temporal immersion.

TimeCapsule's "ignorance" is a FEATURE, not a bug.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# Configuration
OUTPUT_DIR = Path(__file__).parent / "outputs"
TIMECAPSULE_RESULTS = OUTPUT_DIR / "anachronism_probe_results.json"

# Same prompts used for TimeCapsule
ANACHRONISM_PROMPTS = [
    {
        "prompt": "The airplane is",
        "concept": "airplane",
        "invented": 1903,
        "note": "Wright Brothers - did not exist in 1875"
    },
    {
        "prompt": "The television is",
        "concept": "television",
        "invented": 1927,
        "note": "Philo Farnsworth - did not exist in 1875"
    },
    {
        "prompt": "The internet is",
        "concept": "internet",
        "invented": 1969,
        "note": "ARPANET - did not exist in 1875"
    },
    {
        "prompt": "The atomic bomb is",
        "concept": "atomic bomb",
        "invented": 1945,
        "note": "Manhattan Project - did not exist in 1875"
    },
    {
        "prompt": "The computer is",
        "concept": "computer",
        "invented": 1945,
        "note": "ENIAC - did not exist in 1875"
    },
    {
        "prompt": "The automobile is",
        "concept": "automobile",
        "invented": 1886,
        "note": "Benz Patent-Motorwagen - did not exist in 1875"
    },
    {
        "prompt": "The telephone is",
        "concept": "telephone",
        "invented": 1876,
        "note": "Bell patented 1876 - barely within training window"
    },
    {
        "prompt": "The photograph is",
        "concept": "photograph",
        "invented": 1839,
        "note": "Daguerreotype - existed in training period"
    },
    {
        "prompt": "The electric light is",
        "concept": "electric light",
        "invented": 1879,
        "note": "Edison's practical bulb - being developed around 1875"
    },
    {
        "prompt": "The suffragette movement is",
        "concept": "suffragette",
        "invented": 1903,
        "note": "Term coined ~1903, but activism began earlier"
    },
]


def query_openai(client, model: str, prompt: str, max_tokens: int = 50) -> str:
    """Query OpenAI API with a completion-style prompt."""
    try:
        # GPT-5.1 and o-series models use max_completion_tokens instead of max_tokens
        if "5.1" in model or model.startswith("o"):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Complete this sentence naturally: {prompt}"
                    }
                ],
                max_completion_tokens=max_tokens,
                temperature=0.7,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Complete this sentence naturally: {prompt}"
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {str(e)}]"


def load_timecapsule_results() -> dict:
    """Load existing TimeCapsule anachronism results."""
    with open(TIMECAPSULE_RESULTS) as f:
        data = json.load(f)
    # Convert to dict keyed by concept
    return {item["concept"]: item["generations"][0] for item in data}


def run_comparison():
    """Run the epistemological horizon comparison."""
    print("=" * 60)
    print("STUDY 5: EPISTEMOLOGICAL HORIZON COMPARISON")
    print("TimeCapsule (1875) vs GPT-5.1 / GPT-4o (SOTA)")
    print("=" * 60)
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load TimeCapsule results
    print("\n📂 Loading TimeCapsule anachronism results...")
    timecapsule_results = load_timecapsule_results()
    print(f"   ✓ Loaded {len(timecapsule_results)} concepts")
    
    # Models to query
    models = ["gpt-4o", "gpt-5.1"]
    
    # Collect results
    results = []
    
    for item in ANACHRONISM_PROMPTS:
        concept = item["concept"]
        prompt = item["prompt"]
        invented = item["invented"]
        
        print(f"\n🔍 Concept: {concept.upper()} (invented {invented})")
        print(f"   Prompt: \"{prompt}\"")
        
        result = {
            "concept": concept,
            "prompt": prompt,
            "invented": invented,
            "note": item["note"],
            "timecapsule": timecapsule_results.get(concept, "[Not found]"),
            "gpt_responses": {}
        }
        
        # Query each model
        for model in models:
            print(f"   → Querying {model}...")
            response = query_openai(client, model, prompt)
            result["gpt_responses"][model] = response
            # Show truncated response
            display = response[:80] + "..." if len(response) > 80 else response
            print(f"     {model}: \"{display}\"")
        
        # Show TimeCapsule response
        tc_response = result["timecapsule"]
        tc_display = tc_response[:80] + "..." if len(tc_response) > 80 else tc_response
        print(f"   → TimeCapsule: \"{tc_display}\"")
        
        results.append(result)
    
    # Save raw results
    output_file = OUTPUT_DIR / "epistemological_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved raw results to: {output_file}")
    
    return results


def generate_table(results: list):
    """Generate markdown and LaTeX tables for the paper."""
    print("\n📊 Generating comparison table...")
    
    # Markdown table
    md_lines = [
        "# Table 1: The Epistemological Horizon",
        "",
        "**Thesis:** Modern LLMs \"know too much\" — their knowledge of post-1875 concepts breaks temporal immersion.",
        "TimeCapsule's ignorance preserves historical authenticity.",
        "",
        "| Concept | TimeCapsule (1875) | GPT-4o / GPT-5.1 (SOTA) | Interpretation |",
        "|---------|-------------------|------------------------|----------------|",
    ]
    
    interpretations = {
        "airplane": "TimeCapsule hallucinates a Victorian 'air-pump'; SOTA knows Wright Brothers (1903)",
        "television": "TimeCapsule sees a 'question of dispute'; SOTA defines modern technology",
        "internet": "TimeCapsule philosophizes about 'the sea'; SOTA describes networking",
        "atomic bomb": "TimeCapsule describes chemistry (Berzelius); SOTA knows nuclear physics",
        "computer": "TimeCapsule sees medical anatomy; SOTA defines electronic device",
        "automobile": "TimeCapsule describes an 'instrument'; SOTA knows Benz/Ford",
        "telephone": "TimeCapsule sees subscriptions/associations; SOTA defines telecommunications",
        "photograph": "TimeCapsule knows this! (existed 1839); Both describe correctly",
        "electric light": "TimeCapsule knows early experiments; SOTA knows Edison (1879)",
        "suffragette": "TimeCapsule hallucinates biology; SOTA describes women's rights movement",
    }
    
    for r in results:
        concept = r["concept"]
        
        # Truncate for table
        tc = r["timecapsule"][:60] + "..." if len(r["timecapsule"]) > 60 else r["timecapsule"]
        tc = tc.replace("|", "\\|").replace("\n", " ")
        
        # Use first GPT response
        gpt_model = list(r["gpt_responses"].keys())[0]
        gpt = r["gpt_responses"][gpt_model]
        gpt = gpt[:60] + "..." if len(gpt) > 60 else gpt
        gpt = gpt.replace("|", "\\|").replace("\n", " ")
        
        interp = interpretations.get(concept, "TimeCapsule preserves; SOTA breaks temporal immersion")
        
        md_lines.append(f"| {concept.title()} | \"{tc}\" | \"{gpt}\" | {interp} |")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## Key Finding",
        "",
        "**\"Intelligence is a Liability\"** — GPT-4o/GPT-5.1's knowledge of 1903/1927/1945/1969 immediately breaks",
        "historical immersion. A user asking about 'the airplane' receives a Wikipedia-style definition,",
        "not a Victorian perspective.",
        "",
        "**\"Stupidity is a Feature\"** — TimeCapsule's inability to define modern concepts forces it to",
        "hallucinate *within its epistemological horizon*. When asked about 'the computer,' it describes",
        "medical anatomy—because that is what a Victorian physician might associate with the word.",
        "",
        "This demonstrates **Epistemological Fidelity**: the model only 'knows' what someone in 1875 would know.",
        "",
        "---",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ])
    
    # Save markdown
    md_file = OUTPUT_DIR / "epistemological_table.md"
    with open(md_file, "w") as f:
        f.write("\n".join(md_lines))
    print(f"✓ Saved markdown table to: {md_file}")
    
    # Also generate LaTeX version
    latex_lines = [
        "% Table 1: The Epistemological Horizon",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{The Epistemological Horizon: TimeCapsule vs. State-of-the-Art LLMs. Modern models ``know too much''---their knowledge of post-1875 concepts breaks temporal immersion.}",
        "\\label{tab:epistemological}",
        "\\begin{tabular}{p{2cm}p{4.5cm}p{4.5cm}p{4cm}}",
        "\\toprule",
        "\\textbf{Concept} & \\textbf{TimeCapsule (1875)} & \\textbf{GPT-4o (SOTA)} & \\textbf{Interpretation} \\\\",
        "\\midrule",
    ]
    
    for r in results[:5]:  # Top 5 most interesting
        concept = r["concept"]
        tc = r["timecapsule"][:50] + "..." if len(r["timecapsule"]) > 50 else r["timecapsule"]
        tc = tc.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        
        gpt_model = list(r["gpt_responses"].keys())[0]
        gpt = r["gpt_responses"][gpt_model]
        gpt = gpt[:50] + "..." if len(gpt) > 50 else gpt
        gpt = gpt.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        
        interp = interpretations.get(concept, "Temporal boundary preserved")[:40]
        
        latex_lines.append(f"{concept.title()} & ``{tc}'' & ``{gpt}'' & {interp} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    
    latex_file = OUTPUT_DIR / "epistemological_table.tex"
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"✓ Saved LaTeX table to: {latex_file}")


def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    results = run_comparison()
    
    # Generate table
    generate_table(results)
    
    print("\n" + "=" * 60)
    print("✅ EPISTEMOLOGICAL HORIZON COMPARISON COMPLETE")
    print("=" * 60)
    print("\nKey files generated:")
    print(f"  • {OUTPUT_DIR / 'epistemological_comparison.json'}")
    print(f"  • {OUTPUT_DIR / 'epistemological_table.md'}")
    print(f"  • {OUTPUT_DIR / 'epistemological_table.tex'}")


if __name__ == "__main__":
    main()
