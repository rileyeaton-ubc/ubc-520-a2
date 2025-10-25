# COSC 520: Advanced Algorithms - Assignment 2

## Advanced Data Structures

This repository is intended to store my assignment 2 submission for UBC COSC 520: Advanced Algorithms. It focuses on exploring and benchmarking advanced data structures. I have selected the following probabilistic data structures for analysis:

- **Bloom filter**
- **Count-min sketch**
- **LogLog**

## Report

My final report for this assignment is available at [docs/riley_eaton_A2_report.pdf](docs/riley_eaton_A2_report.pdf)

## Dataset

This project uses synthetic web server access logs to test all three probabilistic data structures. The dataset simulates realistic web traffic with Zipfian-distributed access patterns.

**Dataset Sizes:**

- **Small**: 10,000 events (906 unique users, 100 URLs)
- **Medium**: 100,000 events (8,556 unique users, 500 URLs)
- **Large**: 1,000,000 events (47,382 unique users, 1,000 URLs)

**Note**: The large dataset is not included in this repository due to GitHub's 100 MB file size limit. To generate it locally:

```bash
python scripts/generate_dataset.py --sizes
```

Or generate just the large dataset:

```bash
python scripts/generate_dataset.py --events 1000000 --users 50000 --urls 1000 --output data/large
```

Each dataset includes:

- `access_logs.csv`: Complete web server logs
- `user_stream.txt`: User IDs only (for cardinality testing)
- `url_stream.txt`: URLs only (for membership and frequency testing)
- `ground_truth.json`: Exact statistics for validation

## Replicating

1. Clone the repository and ensure you have Python 3
2. Run `python -m pip install -r requirements.txt`
3. Generate datasets: `python scripts/generate_dataset.py --sizes`
4. Run benchmarks: `python scripts/run_benchmarks.py`
