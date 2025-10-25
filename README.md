# COSC 520: Advanced Algorithms - Assignment 2

## Advanced Data Structures

This repository is intended to store my assignment 2 submission for UBC COSC 520: Advanced Algorithms. It focuses on exploring and benchmarking advanced data structures. I have selected the following probabilistic data structures for analysis:

- **Bloom filter**
- **Count-min sketch**
- **LogLog**

## Report

My final report for this assignment is available at [docs/riley_eaton_A2_report.pdf](docs/riley_eaton_A2_report.pdf)

## Quick start for replicating

1. Clone the repository and ensure you have Python 3
2. Install dependencies: `pip install -r requirements.txt`
3. Generate datasets (if needed): `python scripts/generate_dataset.py --sizes`
4. Run tests: `pytest tests/ -v`
5. Run benchmarks: `python scripts/benchmark.py --data-dir data/medium`

## Dataset

This project uses synthetic web server access logs to test all three probabilistic data structures. The dataset simulates realistic web traffic with Zipfian-distributed access patterns.

**Dataset Sizes:**

- **Small**: 10,000 events (906 unique users, 100 URLs)
- **Medium**: 100,000 events (8,556 unique users, 500 URLs)
- **Large**: 1,000,000 events (47,382 unique users, 1,000 URLs)

The largest dataset was limited to 1 million events instead of 10 million due to hardware constraints during benchmarking.

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

## Testing

All three probabilistic data structures have comprehensive unit tests covering functionality, accuracy, and edge cases.

### Running Tests

Run all unit tests:

```bash
pytest tests/ -v
```

Run tests for a specific data structure:

```bash
pytest tests/test_bloom_filter.py -v
pytest tests/test_count_min_sketch.py -v
pytest tests/test_loglog.py -v
```

### Test Coverage

Each data structure has **10+ comprehensive unit tests** covering:

**Bloom Filter** (`tests/test_bloom_filter.py`):

- Initialization and parameter validation
- Add and membership query operations
- False positive rate measurement and theoretical bounds
- Guaranteed no false negatives
- Large dataset testing (10,000+ items)
- Different parameter combinations (size and number of hash functions)
- Optimal parameter calculations
- Load factor and bit utilization
- Real dataset validation using generated data

**Count-Min Sketch** (`tests/test_count_min_sketch.py`):

- Initialization and parameter validation
- Update and frequency estimation operations
- Point queries with incremental updates
- Guaranteed no underestimation (estimates ≥ true frequency)
- Accuracy testing with known distributions
- Error bounds verification (ε-δ guarantees)
- Different parameter combinations (width and depth)
- Required parameter calculations
- Sketch merging with compatible dimensions
- Real dataset validation using generated data

**LogLog** (`tests/test_loglog.py`):

- Initialization and precision parameter validation
- Add items and cardinality estimation
- Accuracy testing with varying dataset sizes (1K, 5K, 10K, 50K items)
- Duplicate handling (same item added multiple times)
- Large cardinality estimation (50,000+ unique items)
- Standard error verification (1.30/√m formula)
- Different precision values (b = 4 to 16)
- LogLog merging with and without overlapping items
- Memory calculation methods
- Real dataset validation using generated data

### Implementation Details

All implementations are built from scratch without external libraries:

- **Hash Functions**: Custom MurmurHash3 implementation (no `hashlib`, `mmh3`, or `xxhash`)
- **Data Structures**: Pure Python using built-in lists (no `numpy` or `bitarray`)
- **Math Operations**: Only standard `math` library for logarithms and exponentials

Tests use both synthetic data and the generated Zipfian-distributed datasets with ground truth validation to ensure accuracy and bounds.

## Benchmarking

Run comprehensive benchmarks on all three data structures:

```bash
python scripts/benchmark.py --data-dir data/medium
```

Benchmarks measure:

- **Memory efficiency**: Actual bytes used per structure
- **Insertion throughput**: Operations per second
- **Scalability**: Performance across dataset sizes (10K-500K elements)
- **Accuracy**: Error rates vs. theoretical bounds

Results are saved to `data/benchmarks/` as JSON files. Visualizations can be generated using the Jupyter notebook at `notebooks/benchmark_analysis.ipynb`.

## Key Results

Performance on medium dataset (100K events):

| Structure            | Memory   | Throughput   | Use Case               |
| -------------------- | -------- | ------------ | ---------------------- |
| **Bloom Filter**     | 0.68 KB  | 166K ops/sec | Set membership         |
| **Count-Min Sketch** | 5.41 KB  | 85K ops/sec  | Frequency estimation   |
| **LogLog**           | 16.10 KB | 170K ops/sec | Cardinality estimation |

**Key Finding**: Count-Min Sketch achieves best memory efficiency at scale (0.0554 bytes/element) as its fixed memory amortizes over unbounded streams, while LogLog achieves highest throughput with O(1) updates.

See the [full report](docs/riley_eaton_A2_report.pdf) for detailed analysis and comparisons.
