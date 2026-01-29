# M&A Prediction Pipeline

**Objective**: Predict M&A target events using machine learning on fundamental financial data.

## 📂 Directory Structure

```
MnA_Prediction/
├── README.md
├── src/                           # Source code
│   ├── mna_colab_pipeline.py      # Main Colab notebook
│   └── feature_engineering.py     # Original reference script
│
├── data/                          # Data files (on GitHub)
│   ├── deals/
│   │   ├── dma_corpus_metadata_with_factset_id.csv  # 2000-2020
│   │   └── factset_xls/                             # 2000-2025
│   │       ├── 2000to05Batch1.xls
│   │       └── ...
│   └── fundamentals/
│       └── compustat_funda_2000on.csv
│
└── archive/                       # Old/unused files
```

## 🚀 Quick Start (Google Colab)

1. Open `src/mna_colab_pipeline.py` in Google Colab
2. Ensure your Google Drive contains:
   - `fundq_full.parquet` (quarterly Compustat, ~564 MB)
   - `funda_full.parquet` (annual Compustat, ~200 MB)
3. Run all cells

## Data Sources

| Source | Location | Coverage |
|--------|----------|----------|
| **Compustat** (fundq/funda) | Google Drive | Through ~2020 |
| **DMA Corpus** | `data/deals/` | 2000-2020 |
| **FactSet XLS** | `data/deals/factset_xls/factset_2000_2025/` | 2000-2025 |

## Pipeline Features

- Multi-horizon labeling (3m-24m targets)
- Probability calibration (prior correction + isotonic)
- Event study verification
- S&P 500 benchmarking
- Schema-preserving data extension

## License

MIT
