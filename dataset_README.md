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

# TimeCapsule Dataset (1800-1875)

This dataset consists of approximately 90GB of cleaned English text derived from historical documents ranging from 1800 to 1875. It serves as the training corpus for the "TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking" paper presented at ACM Creativity and Cognition 2026.

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
@inproceedings{grigorian2026timecapsule,
  author = {Hayk Grigorian and Hamed Yaghoobian},
  title = {TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking},
  booktitle = {Proceedings of the ACM Creativity and Cognition 2026 Conference},
  year = {2026},
  publisher = {ACM},
  note = {Dataset: \url{https://huggingface.co/datasets/postgrammar/london-llm-1800}}
}
```

## License
Open Data Commons Attribution License (ODC-By) v1.0
