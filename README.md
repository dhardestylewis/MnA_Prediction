# M&A Prediction Pipeline

**Objective**: Predict M&A (Mergers & Acquisitions) target events using machine learning on fundamental financial data.

## 🚀 Quick Start (Google Colab)

1. Open `mna_colab_pipeline.py` in Google Colab
2. Ensure your Google Drive contains:
   - `fundq_full.parquet` (quarterly Compustat fundamentals)
   - `funda_full.parquet` (annual Compustat fundamentals)
3. Run all cells - the pipeline clones this repo for smaller data files

## 📂 Data Sources

### Large Files (Google Drive only - too large for GitHub)
| File | Size | Description |
|------|------|-------------|
| `fundq_full.parquet` | ~564 MB | Quarterly Compustat fundamentals |
| `funda_full.parquet` | ~TBD | Annual Compustat fundamentals |

### Smaller Files (included in this repo)
| File/Directory | Coverage | Description |
|----------------|----------|-------------|
| `dma_corpus_metadata_with_factset_id.csv` | 2000-2020 | M&A deal metadata with FactSet IDs |
| `All Factset Deals 2000 to Pres.../` | 2000-2025 | FactSet deal batches (XLS files) |
| `compustat_funda_2000on.csv` | 2000+ | Annual fundamentals (raw CSV) |

### FactSet Deal Batches
```
All Factset Deals 2000 to Pres.../
├── 2000to05Batch1.xls   # Deals 2000-2005
├── 2006to10Batch2.xls   # Deals 2006-2010
├── 2011to14Batch3.xls   # Deals 2011-2014
├── 2015to17Batch4.xls   # Deals 2015-2017
├── 2018to20Batch5.xls   # Deals 2018-2020
├── 2021to24Batch6.xls   # Deals 2021-2024
└── 2025toPRESBatch7.xls # Deals 2025+
```

## 🔧 Pipeline Features

- **Multi-horizon labeling**: 3m, 6m, 9m, 12m, 15m, 18m, 21m, 24m targets
- **Vectorized feature engineering**: Financial ratios, growth rates
- **Probability calibration**: Two-layer calibration (prior correction + isotonic)
- **Stability testing**: Jaccard overlap across trials
- **S&P 500 baseline**: Benchmark comparison
- **Event study**: M&A announcement jump verification
- **Correlation screening**: For portfolio construction

## 📊 Key Columns

### Panel Data (Compustat)
- `gvkey`: Global Company Key
- `conm`: Company Name
- `datadate`: Data date
- `cik`: SEC CIK identifier

### Deal Data (FactSet)
- `target`: Target company name
- `acquirer`: Acquiring company name
- `date_announcement`: M&A announcement date
- `FactSet ID`: FactSet deal identifier

## 📝 Usage Notes

The pipeline auto-resolves file paths:
1. **GitHub repo** checked first (for deal files, XLS batches)
2. **Google Drive** fallback (for large parquet files)

```python
# Example: resolve_path() helper
deals_path = resolve_path("dma_corpus_metadata_with_factset_id.csv")  # GitHub first
fundq_path = resolve_path("fundq_full.parquet", prefer_repo=False)    # Drive only
```

## 📄 License

MIT
