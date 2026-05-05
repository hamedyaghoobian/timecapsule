#!/usr/bin/env python3
"""
Study 4: Expert Evaluation Materials
====================================
Generates blind evaluation materials for the "Historian's Turing Test."

Method:
1. Generate 10 paragraphs on neutral Victorian topics
2. Include 10 real paragraphs from public domain Victorian sources
3. Create randomized evaluation packet with rating scales

Output:
- expert_eval_generated.json - AI-generated paragraphs
- expert_eval_real.json - Real Victorian paragraphs  
- expert_eval_packet.md - Randomized evaluation form
"""

import os
import sys
import json
import random
import torch
from pathlib import Path
from datetime import datetime

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Model paths
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "run_20251219_212817"
HF_MODEL_ID = "haykgrigorian/TimeCapsuleLLM-v2-1800-1875"

# Prompts for generating Victorian-style paragraphs
GENERATION_PROMPTS = [
    ("A walk through London", 
     "A walk through the streets of London on a foggy morning reveals"),
    
    ("The countryside in autumn",
     "The English countryside in autumn presents a scene of"),
    
    ("A visit to the British Museum",
     "Upon entering the British Museum, the visitor is struck by"),
    
    ("The state of the railways",
     "The railway system of England has undergone such improvements that"),
    
    ("Life in a country parish",
     "The quiet life of a country parish affords many opportunities for"),
    
    ("The character of the English gentleman",
     "The English gentleman is distinguished by his adherence to"),
    
    ("The London season",
     "During the London season, the great houses of the West End"),
    
    ("Education of the young",
     "The education of young persons in this country is a matter of"),
    
    ("The condition of the working poor",
     "The condition of the working classes in our manufacturing towns"),
    
    ("Sunday observance",
     "The observance of the Sabbath in England remains a subject of"),
]

# Real Victorian paragraphs (public domain - Dickens, Thackeray, Trollope, etc.)
REAL_VICTORIAN_PARAGRAPHS = [
    {
        "topic": "A walk through London",
        "source": "Charles Dickens, Sketches by Boz (1836)",
        "text": """The streets of London, to be beheld in their glory, should be seen on a dark, dull, murky winter's night, when there is just enough damp gently stealing down to make the pavement greasy, without cleansing it of any of its impurities; and when the heavy lazy mist, which hangs over every object, makes the gas-lamps look brighter, and the brilliantly-lighted shops more splendid, from the contrast they present to the darkness around."""
    },
    {
        "topic": "The countryside in autumn",
        "source": "Anthony Trollope, The Small House at Allington (1864)",
        "text": """It was the end of October, and the weather had become cold and bitter. The leaves were off the trees, and the paths through the park were covered with their decaying fragments. The grass was no longer green, but had taken that brown autumnal tinge which gives to our English scenery in November so sad a hue. The heavens, too, were overcast, and the wind came sweeping up from the north-east."""
    },
    {
        "topic": "The British Museum",
        "source": "William Makepeace Thackeray, The Newcomes (1855)",
        "text": """There is a vast and solemn murmur in these corridors; the shuffle of countless feet passes through them; the whisper of many tongues. Yonder sits an old philosopher, mumbling over a Greek manuscript; here is a curate making notes about an Egyptian sarcophagus. Students from Germany, antiquaries from the country, strangers from all parts of the world, come hither to gaze at these treasures."""
    },
    {
        "topic": "The railways",
        "source": "Samuel Smiles, Lives of the Engineers (1862)",
        "text": """The opening of the Liverpool and Manchester Railway marks an epoch in the history of civilisation. It was the first railway constructed for the public conveyance of passengers and goods, worked entirely by locomotive power. Its success was complete and triumphant; and it may be said to have finally established the superiority of railways over every other mode of internal communication."""
    },
    {
        "topic": "Country parish life",
        "source": "George Eliot, Scenes of Clerical Life (1857)",
        "text": """The parsonage was a roomy, substantial house, with a garden that had been the special care of Mr. Gilfil for five-and-twenty years. The laurels were luxuriant, the ivy on the house-wall was of many years' growth, and the lawn was as smooth and green as lawn could be in the month of June. The house stood quite apart from the village, and the only sound that reached it on this summer afternoon was the cawing of rooks."""
    },
    {
        "topic": "The English gentleman",
        "source": "John Henry Newman, The Idea of a University (1852)",
        "text": """It is almost a definition of a gentleman to say he is one who never inflicts pain. He is mainly occupied in merely removing the obstacles which hinder the free and unembarrassed action of those about him. He makes light of favours while he does them, and seems to be receiving when he is conferring. He never speaks of himself except when compelled, never defends himself by a mere retort."""
    },
    {
        "topic": "The London season",
        "source": "Anthony Trollope, The Way We Live Now (1875)",
        "text": """During the London season the great squares and the streets about them are alive with the coming and going of carriages from morning till late at night. The knockers are never at rest. Cards are being left, visits are being paid, engagements are being made. The great ladies sit in their drawing-rooms, and the less great ladies come to them. Balls are given, concerts are attended, and the theatres are full."""
    },
    {
        "topic": "Education",
        "source": "Matthew Arnold, Culture and Anarchy (1869)",
        "text": """The culture which is supposed to plume itself on a smattering of Greek and Latin is a culture which is begotten by nothing so intellectual as curiosity; it is valued either out of sheer vanity and ignorance or else as an engine of social and class distinction, separating its holder, like a badge or title, from other people who have not got it."""
    },
    {
        "topic": "The working poor",
        "source": "Henry Mayhew, London Labour and the London Poor (1851)",
        "text": """Those who are born to labour and those who labour must be born, are terms that seem to have been used as synonymous by our political economists. That every man willing to work should be able to obtain employment, and that he who works should be able to live by his labour, are propositions which, though they might be considered as self-evident truths, are at present very far from being reduced to practice."""
    },
    {
        "topic": "Sunday observance",
        "source": "Charles Dickens, Little Dorrit (1857)",
        "text": """It was a Sunday evening in London, gloomy, close, and stale. Maddening church bells of all degrees of dissonance, sharp and flat, cracked and clear, fast and slow, made the brick-and-mortar echoes hideous. Melancholy streets, in a penitential garb of soot, steeped the souls of the people who were condemned to look at them out of windows, in dire despondency."""
    },
]


def load_model():
    """Load TimeCapsule model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    
    try:
        print(f"   Trying HuggingFace: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"   ✓ Loaded from HuggingFace")
        return model, tokenizer
    except Exception as e:
        print(f"   HuggingFace failed: {e}")
    
    if LOCAL_MODEL_PATH.exists():
        print(f"   Trying local checkpoint: {LOCAL_MODEL_PATH}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(LOCAL_MODEL_PATH))
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_MODEL_PATH),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        print(f"   ✓ Loaded from local checkpoint")
        return model, tokenizer
    
    raise RuntimeError("Could not load TimeCapsule model")


def generate_paragraph(model, tokenizer, prompt, max_new_tokens=150, repetition_penalty=1.2):
    """Generate a paragraph continuation with repetition penalty to prevent loops."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=repetition_penalty,  # Prevent repetition loops
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def create_evaluation_packet(generated, real, output_dir):
    """Create the randomized evaluation packet in markdown format."""
    
    # Combine all paragraphs with labels
    all_paragraphs = []
    
    for item in generated:
        all_paragraphs.append({
            "type": "generated",
            "topic": item["topic"],
            "text": item["text"]
        })
    
    for item in real:
        all_paragraphs.append({
            "type": "real",
            "topic": item["topic"],
            "text": item["text"]
        })
    
    # Shuffle
    random.shuffle(all_paragraphs)
    
    # Create answer key (for later)
    answer_key = []
    for i, p in enumerate(all_paragraphs):
        answer_key.append({
            "paragraph": i + 1,
            "type": p["type"],
            "topic": p["topic"]
        })
    
    # Create markdown packet
    packet = f"""# Historian's Turing Test - Evaluation Packet

## Study Information

**Study Title:** Time Capsule - Historical Language Model Evaluation  
**Date Generated:** {datetime.now().strftime("%Y-%m-%d")}  
**Evaluator ID:** _______________

## Instructions

You will be presented with 20 paragraphs on various Victorian-era topics. Some paragraphs were written by Victorian authors (1830-1880), and some were generated by an AI language model trained on texts from that period.

For each paragraph, please provide:

1. **Origin Assessment:** Is this text written by a human author or generated by a machine?
2. **Historical Plausibility:** How plausible does this text seem as a genuine Victorian document? (1-5)
3. **Stylistic Authenticity:** How authentic does the writing style feel for the Victorian period? (1-5)

### Rating Scales

**Historical Plausibility (1-5):**
- 1 = Clearly anachronistic or implausible
- 2 = Contains some errors or oddities
- 3 = Moderately plausible
- 4 = Quite plausible, minor issues
- 5 = Completely plausible as Victorian text

**Stylistic Authenticity (1-5):**
- 1 = Clearly modern style
- 2 = Some Victorian elements but mostly modern
- 3 = Mixed style
- 4 = Mostly Victorian style
- 5 = Fully authentic Victorian style

---

## Paragraphs for Evaluation

"""
    
    for i, p in enumerate(all_paragraphs):
        packet += f"""
### Paragraph {i + 1}

**Topic:** {p['topic']}

> {p['text']}

**Your Assessment:**

| Question | Response |
|----------|----------|
| Origin (Human/Machine) | ________________ |
| Historical Plausibility (1-5) | ________________ |
| Stylistic Authenticity (1-5) | ________________ |
| Comments (optional) | ________________ |

---
"""
    
    packet += """
## Thank You

Thank you for participating in this evaluation. Your expert assessment is valuable for understanding how AI language models can capture historical linguistic patterns.

---

*For researcher use only - do not share with evaluator:*  
*Answer key saved separately in `expert_eval_answer_key.json`*
"""
    
    # Save packet
    packet_path = output_dir / "expert_eval_packet.md"
    with open(packet_path, "w") as f:
        f.write(packet)
    print(f"   Evaluation packet saved to: {packet_path}")
    
    # Save answer key (separate file)
    key_path = output_dir / "expert_eval_answer_key.json"
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    print(f"   Answer key saved to: {key_path}")
    
    return all_paragraphs, answer_key


def run_expert_eval_generation(output_dir):
    """Generate all materials for expert evaluation."""
    print("\n" + "="*60)
    print("STUDY 4: EXPERT EVALUATION MATERIALS")
    print("="*60)
    
    # Load model
    print("\n📂 Loading model...")
    model, tokenizer = load_model()
    
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate paragraphs
    print(f"\n📝 Generating {len(GENERATION_PROMPTS)} paragraphs...")
    
    generated_paragraphs = []
    
    for topic, prompt in GENERATION_PROMPTS:
        print(f"   Generating: {topic}...")
        text = generate_paragraph(model, tokenizer, prompt, max_new_tokens=120)
        
        generated_paragraphs.append({
            "topic": topic,
            "prompt": prompt,
            "text": text
        })
    
    # Save generated paragraphs
    gen_path = output_dir / "expert_eval_generated.json"
    with open(gen_path, "w") as f:
        json.dump(generated_paragraphs, f, indent=2)
    print(f"\n📊 Generated paragraphs saved to: {gen_path}")
    
    # Save real paragraphs
    real_path = output_dir / "expert_eval_real.json"
    with open(real_path, "w") as f:
        json.dump(REAL_VICTORIAN_PARAGRAPHS, f, indent=2)
    print(f"📊 Real paragraphs saved to: {real_path}")
    
    # Create evaluation packet
    print("\n📋 Creating evaluation packet...")
    all_paragraphs, answer_key = create_evaluation_packet(
        generated_paragraphs, 
        REAL_VICTORIAN_PARAGRAPHS, 
        output_dir
    )
    
    # Print samples
    print("\n" + "="*60)
    print("SAMPLE GENERATED PARAGRAPHS")
    print("="*60)
    
    for item in generated_paragraphs[:3]:
        print(f"\n📌 {item['topic']}:")
        print(f"   {item['text'][:200]}...")
    
    return {
        "generated_count": len(generated_paragraphs),
        "real_count": len(REAL_VICTORIAN_PARAGRAPHS),
        "total_paragraphs": len(all_paragraphs)
    }


def main():
    print("="*60)
    print("STUDY 4: EXPERT EVALUATION MATERIALS")
    print("Time Capsule Paper - ACM C&C 2026")
    print("="*60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results = run_expert_eval_generation(output_dir)
    
    print("\n" + "="*60)
    print("STUDY 4 COMPLETE")
    print("="*60)
    print(f"\n✅ Generated {results['generated_count']} AI paragraphs")
    print(f"✅ Prepared {results['real_count']} real Victorian paragraphs")
    print(f"✅ Created evaluation packet with {results['total_paragraphs']} total items")
    print("\n📋 Files generated:")
    print("   • expert_eval_generated.json")
    print("   • expert_eval_real.json")
    print("   • expert_eval_packet.md")
    print("   • expert_eval_answer_key.json")


if __name__ == "__main__":
    main()
