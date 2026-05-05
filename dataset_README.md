---
license: odc-by
task_categories:
- text-generation
- fill-mask
language:
- en
size_categories:
- 10B<n<100B
---

# Historic London LLM Dataset (1800-1875)

This dataset consists of approximately 90GB of cleaned English text derived from historical documents ranging from 1800 to 1875. It serves as the training corpus for the "Time Capsule LLM" project.

## Dataset Details
- **Time Period**: 1800 - 1875
- **Language**: English
- **Source**: Digitized historical texts (Internet Archive)
- **Tokenization**: Custom BPE Tokenizer (vocab size 32,000)

## Usage
This dataset is intended for research in digital humanities and historical linguistics.

## Citation
If you use this dataset, please cite it as follows:

```bibtex
@misc{london_llm_1800,
  author = {Hayk Grigorian, Hamed Yaghoobian},
  title = {Historic London English (1800-1875)},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/postgrammar/london-llm-1800}}
}
```

## License
Open Data Commons Attribution License (ODC-By) v1.0
