"""
Categorize 1800-1875 Internet Archive corpus by genre/document type.

This script analyzes text files from data/02_cleaned/ and classifies them into
categories such as Literary Fiction, Periodicals, Parliamentary Records, etc.
for reporting in the ACM Creativity and Cognition paper.

Author: Automated Analysis
Date: February 2026
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import csv
import json


# Category definitions with keywords for filename and content matching
CATEGORY_KEYWORDS = {
    "Literary Fiction": {
        "filename": ["novel", "tale", "story", "stories", "romance", "narrative", "fiction"],
        "content": ["chapter", "narrative", "character", "protagonist"],
        "priority": 5
    },
    "Poetry": {
        "filename": ["poem", "poems", "verse", "ballad", "song", "lyric", "poetry", "poetical"],
        "content": ["stanza", "verse", "rhyme", "poet"],
        "priority": 9
    },
    "Drama/Theatre": {
        "filename": ["play", "plays", "tragedy", "comedy", "drama", "theatre", "theater", "opera"],
        "content": ["act i", "scene", "dramatis personae", "exeunt"],
        "priority": 8
    },
    "History": {
        "filename": ["history", "historic", "chronicle", "antiquit", "antiquar", "ancient"],
        "content": ["century", "reign", "empire", "ancient"],
        "priority": 4
    },
    "Biography/Memoir": {
        "filename": ["life", "lives", "memoir", "memoirs", "character", "sketch", "biography", "biographic"],
        "content": ["born", "died", "autobiography", "recollections"],
        "priority": 6
    },
    "Religious/Theological": {
        "filename": ["sermon", "sermons", "bible", "biblical", "christian", "church", "divine", "theology", "theological", "religion", "religious", "scripture", "gospel"],
        "content": ["scripture", "sermon", "biblical", "congregation", "prayer"],
        "priority": 7
    },
    "Science/Natural Philosophy": {
        "filename": ["science", "scientific", "geology", "geolog", "zoolog", "botany", "botanical", "natural", "philosoph", "chemistry", "physics", "astronomy"],
        "content": ["experiment", "observation", "specimen", "scientific"],
        "priority": 4
    },
    "Travel/Geography": {
        "filename": ["voyage", "voyages", "journey", "travel", "travels", "tour", "expedition", "adventure", "adventures"],
        "content": ["journey", "travelled", "expedition", "voyage"],
        "priority": 6
    },
    "Periodicals/Journalism": {
        "filename": ["magazine", "journal", "gazette", "times", "review", "register", "periodical", "newspaper", "news", "monthly", "quarterly", "weekly"],
        "content": ["volume", "number", "issue", "editor", "published monthly"],
        "priority": 8
    },
    "Parliamentary/Legal": {
        "filename": ["parliament", "parliamentary", "hansard", "debate", "debates", "proceedings", "bill", "act", "law", "legal", "legislation"],
        "content": ["parliament", "bill", "act", "legislation", "house of commons", "house of lords"],
        "priority": 9
    },
    "Essays/Letters": {
        "filename": ["essay", "essays", "letter", "letters", "discourse", "discourses", "treatise", "tract"],
        "content": ["essay", "epistle", "discourse"],
        "priority": 3
    },
    "Reference/Educational": {
        "filename": ["dictionary", "encyclopedia", "encyclopaedia", "guide", "manual", "handbook", "grammar", "primer", "textbook", "catalogue", "catalog"],
        "content": ["definition", "dictionary", "encyclopedia"],
        "priority": 7
    },
    "Philosophy": {
        "filename": ["philosoph", "metaphysic", "logic", "moral", "ethics", "ethical"],
        "content": ["philosophical", "metaphysics", "reasoning"],
        "priority": 5
    },
    "Medical": {
        "filename": ["medical", "medicine", "physician", "surgery", "surgical", "anatomy", "disease", "health"],
        "content": ["patient", "disease", "medical", "treatment", "physician"],
        "priority": 6
    }
}


class CorpusCategorizor:
    """Analyze and categorize historical text corpus."""
    
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.files = []
        self.categories = defaultdict(list)
        self.metadata = {}
        self.stats = Counter()
        
    def scan_corpus(self):
        """Scan the corpus directory and collect all text files."""
        print(f"Scanning corpus directory: {self.corpus_dir}")
        self.files = sorted(self.corpus_dir.glob("*.txt"))
        print(f"Found {len(self.files)} text files")
        return len(self.files)
    
    def extract_title_from_filename(self, filename: str) -> str:
        """Extract readable title from filename."""
        # Remove _djvu.txt suffix
        name = filename.replace("_djvu.txt", "")
        # Split by __ and take first part (more descriptive)
        parts = name.split("__")
        title = parts[0] if parts else name
        # Clean up common patterns
        title = re.sub(r'\d+\.', '', title)  # Remove numbers with dots
        title = re.sub(r'[._]', ' ', title)  # Replace _ and . with spaces
        title = re.sub(r'\s+', ' ', title).strip()  # Clean whitespace
        return title
    
    def categorize_by_filename(self, filename: str) -> Tuple[Optional[str], float]:
        """
        Categorize based on filename keywords.
        Returns (category, confidence_score).
        """
        filename_lower = filename.lower()
        
        # Track matches with priority weighting
        matches = []
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords["filename"]:
                if keyword in filename_lower:
                    # Score based on keyword match and category priority
                    score = keywords["priority"] * 10
                    # Bonus for exact word match vs substring
                    if re.search(rf'\b{keyword}\b', filename_lower):
                        score += 20
                    matches.append((category, score))
                    break  # One match per category is enough
        
        if matches:
            # Sort by score and return best match
            matches.sort(key=lambda x: x[1], reverse=True)
            best_category, best_score = matches[0]
            # Normalize confidence to 0-1 scale
            confidence = min(best_score / 100.0, 1.0)
            return best_category, confidence
        
        return None, 0.0
    
    def categorize_by_content(self, filepath: Path, sample_size: int = 2000) -> Tuple[Optional[str], float]:
        """
        Categorize based on content sampling.
        Reads first N characters and looks for genre indicators.
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(sample_size).lower()
            
            if not content.strip():
                return None, 0.0
            
            matches = []
            
            for category, keywords in CATEGORY_KEYWORDS.items():
                match_count = 0
                for keyword in keywords["content"]:
                    if keyword in content:
                        match_count += 1
                
                if match_count > 0:
                    # Score based on number of content keyword matches
                    score = match_count * keywords["priority"]
                    matches.append((category, score))
            
            if matches:
                matches.sort(key=lambda x: x[1], reverse=True)
                best_category, best_score = matches[0]
                # Normalize confidence (max expected ~50)
                confidence = min(best_score / 50.0, 1.0)
                return best_category, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"Error reading {filepath.name}: {e}")
            return None, 0.0
    
    def get_file_size(self, filepath: Path) -> int:
        """Get file size in bytes."""
        try:
            return filepath.stat().st_size
        except:
            return 0
    
    def categorize_file(self, filepath: Path) -> Dict:
        """
        Categorize a single file using both filename and content analysis.
        Returns metadata dictionary.
        """
        filename = filepath.name
        
        # Try filename categorization first
        category_filename, conf_filename = self.categorize_by_filename(filename)
        
        # If filename confidence is low or no match, try content
        category_content, conf_content = None, 0.0
        if conf_filename < 0.7:
            category_content, conf_content = self.categorize_by_content(filepath)
        
        # Choose best categorization
        if conf_filename >= conf_content:
            final_category = category_filename or "Uncategorized"
            final_confidence = conf_filename
            method = "filename"
        else:
            final_category = category_content or "Uncategorized"
            final_confidence = conf_content
            method = "content"
        
        # Extract title
        title = self.extract_title_from_filename(filename)
        
        # Get file size
        file_size = self.get_file_size(filepath)
        
        metadata = {
            "filename": filename,
            "title": title,
            "category": final_category,
            "confidence": round(final_confidence, 3),
            "method": method,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2)
        }
        
        return metadata
    
    def categorize_all(self):
        """Categorize all files in the corpus."""
        print(f"\nCategorizing {len(self.files)} files...")
        
        for i, filepath in enumerate(self.files):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(self.files)} files...")
            
            metadata = self.categorize_file(filepath)
            category = metadata["category"]
            
            self.categories[category].append(metadata)
            self.metadata[metadata["filename"]] = metadata
            self.stats[category] += 1
        
        print(f"✓ Categorization complete!")
    
    def generate_statistics(self) -> Dict:
        """Generate statistical summary of categorization."""
        total_files = len(self.files)
        
        stats = {
            "total_files": total_files,
            "categories": {}
        }
        
        for category in sorted(self.stats.keys()):
            count = self.stats[category]
            percentage = (count / total_files * 100) if total_files > 0 else 0
            
            # Calculate average confidence for this category
            confidences = [m["confidence"] for m in self.categories[category]]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Calculate total size for category
            total_size = sum(m["file_size_bytes"] for m in self.categories[category])
            
            stats["categories"][category] = {
                "count": count,
                "percentage": round(percentage, 2),
                "avg_confidence": round(avg_confidence, 3),
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print statistics in a readable format."""
        print("\n" + "="*70)
        print("CORPUS CATEGORIZATION STATISTICS")
        print("="*70)
        print(f"\nTotal Files Analyzed: {stats['total_files']:,}")
        print(f"\nCategory Distribution:\n")
        
        # Sort by percentage descending
        sorted_cats = sorted(
            stats['categories'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )
        
        print(f"{'Category':<30} {'Count':>8} {'Percentage':>12} {'Avg Conf':>10} {'Size (MB)':>12}")
        print("-" * 70)
        
        for category, cat_stats in sorted_cats:
            print(f"{category:<30} {cat_stats['count']:>8,} "
                  f"{cat_stats['percentage']:>11.2f}% "
                  f"{cat_stats['avg_confidence']:>10.3f} "
                  f"{cat_stats['total_size_mb']:>12.2f}")
        
        print("="*70 + "\n")
    
    def save_results(self, output_dir: str = "outputs/corpus_analysis"):
        """Save categorization results to CSV and JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed CSV with all files
        csv_path = output_path / "corpus_categorization.csv"
        print(f"\nSaving detailed results to: {csv_path}")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filename', 'title', 'category', 'confidence', 
                'method', 'file_size_kb'
            ])
            writer.writeheader()
            
            for metadata in sorted(self.metadata.values(), key=lambda x: x['category']):
                writer.writerow({
                    'filename': metadata['filename'],
                    'title': metadata['title'],
                    'category': metadata['category'],
                    'confidence': metadata['confidence'],
                    'method': metadata['method'],
                    'file_size_kb': metadata['file_size_kb']
                })
        
        # Save statistics JSON
        stats = self.generate_statistics()
        json_path = output_path / "corpus_statistics.json"
        print(f"Saving statistics to: {json_path}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        # Save summary CSV for paper
        summary_csv_path = output_path / "category_summary.csv"
        print(f"Saving summary for paper to: {summary_csv_path}")
        
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Count', 'Percentage', 'Size (MB)'])
            
            sorted_cats = sorted(
                stats['categories'].items(),
                key=lambda x: x[1]['percentage'],
                reverse=True
            )
            
            for category, cat_stats in sorted_cats:
                writer.writerow([
                    category,
                    cat_stats['count'],
                    f"{cat_stats['percentage']:.2f}%",
                    f"{cat_stats['total_size_mb']:.2f}"
                ])
        
        print(f"\n✓ All results saved to: {output_path}")
        return output_path


def main():
    """Main execution function."""
    # Path to cleaned corpus
    corpus_dir = "data/02_cleaned"
    
    if not os.path.exists(corpus_dir):
        print(f"Error: Corpus directory not found: {corpus_dir}")
        return
    
    # Initialize categorizer
    categorizer = CorpusCategorizor(corpus_dir)
    
    # Scan corpus
    file_count = categorizer.scan_corpus()
    
    if file_count == 0:
        print("No files found in corpus directory!")
        return
    
    # Categorize all files
    categorizer.categorize_all()
    
    # Generate and print statistics
    stats = categorizer.generate_statistics()
    categorizer.print_statistics(stats)
    
    # Save results
    output_dir = categorizer.save_results()
    
    print(f"\n{'='*70}")
    print("CATEGORIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    print("  1. corpus_categorization.csv - Detailed file-by-file results")
    print("  2. corpus_statistics.json - Complete statistics in JSON format")
    print("  3. category_summary.csv - Summary table for your paper")
    print("\nYou can use category_summary.csv directly in your ACM paper!")


if __name__ == "__main__":
    main()
