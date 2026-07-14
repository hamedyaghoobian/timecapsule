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
@inproceedings{Grigorian_2026,
  author    = {Grigorian, Hayk and Yaghoobian, Hamed},
  title     = {TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking},
  booktitle = {Proceedings of the 2026 Conference on Creativity and Cognition},
  series    = {C\&C '26},
  year      = {2026},
  pages     = {229--238},
  publisher = {ACM},
  doi       = {10.1145/3803784.3807554},
  url       = {https://doi.org/10.1145/3803784.3807554},
  note      = {Dataset: \url{https://huggingface.co/datasets/postgrammar/london-llm-1800}}
}
```

## License
Open Data Commons Attribution License (ODC-By) v1.0
