"""Generate synthetic web server log dataset for testing probabilistic data structures."""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import uuid


class ZipfianGenerator:
    """Generate values following Zipfian distribution (power law)."""

    def __init__(self, n: int, alpha: float = 1.0, seed: int = 42):
        """Initialize Zipfian generator.

        Args:
            n: Number of unique items
            alpha: Skewness parameter (higher = more skewed, typical: 0.8-1.5)
            seed: Random seed for reproducibility
        """
        self.n = n
        self.alpha = alpha
        random.seed(seed)

        # Pre-compute cumulative probabilities
        self.cumulative = []
        total = sum(1.0 / (i ** alpha) for i in range(1, n + 1))
        cumulative_sum = 0.0

        for i in range(1, n + 1):
            cumulative_sum += (1.0 / (i ** alpha)) / total
            self.cumulative.append(cumulative_sum)

    def generate(self) -> int:
        """Generate a random index following Zipfian distribution."""
        r = random.random()
        for i, cumulative_prob in enumerate(self.cumulative):
            if r <= cumulative_prob:
                return i
        return self.n - 1


def generate_user_ids(num_unique: int, seed: int = 42) -> List[str]:
    """Generate unique user IDs (simulated as IP addresses).

    Args:
        num_unique: Number of unique user IDs to generate
        seed: Random seed

    Returns:
        List of unique IP addresses
    """
    random.seed(seed)
    ips = []

    for _ in range(num_unique):
        ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}." \
             f"{random.randint(0, 255)}.{random.randint(1, 255)}"
        ips.append(ip)

    return ips


def generate_urls(num_unique: int, seed: int = 42) -> List[str]:
    """Generate unique URLs/endpoints.

    Args:
        num_unique: Number of unique URLs to generate
        seed: Random seed

    Returns:
        List of unique URLs
    """
    random.seed(seed)

    endpoints = [
        "/", "/home", "/about", "/contact", "/products", "/services",
        "/login", "/logout", "/register", "/profile", "/settings",
        "/search", "/api/data", "/api/users", "/api/products",
        "/blog", "/news", "/help", "/faq", "/terms", "/privacy"
    ]

    # Generate additional URLs to reach num_unique
    urls = endpoints.copy()

    for i in range(len(endpoints), num_unique):
        category = random.choice(["products", "users", "posts", "items", "categories"])
        urls.append(f"/{category}/{uuid.uuid4().hex[:8]}")

    return urls[:num_unique]


def generate_dataset(
    total_events: int,
    unique_users: int,
    unique_urls: int,
    zipf_alpha: float = 1.0,
    seed: int = 42,
    output_dir: str = "data"
) -> Dict[str, Any]:
    """Generate synthetic web server log dataset.

    Args:
        total_events: Total number of log entries to generate
        unique_users: Number of unique users
        unique_urls: Number of unique URLs
        zipf_alpha: Zipfian distribution parameter (higher = more skewed)
        seed: Random seed for reproducibility
        output_dir: Output directory for generated files

    Returns:
        Dictionary containing ground truth statistics
    """
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate pools of unique values
    print(f"Generating {unique_users} unique user IDs...")
    user_pool = generate_user_ids(unique_users, seed)

    print(f"Generating {unique_urls} unique URLs...")
    url_pool = generate_urls(unique_urls, seed)

    # Initialize Zipfian generators for realistic distributions
    user_zipf = ZipfianGenerator(unique_users, alpha=zipf_alpha, seed=seed)
    url_zipf = ZipfianGenerator(unique_urls, alpha=zipf_alpha, seed=seed + 1)

    # Tracking for ground truth
    user_frequency = {}
    url_frequency = {}
    unique_users_seen = set()
    unique_urls_seen = set()

    # Generate log entries
    print(f"Generating {total_events} log entries...")
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    request_types = ["GET", "POST", "PUT", "DELETE"]
    request_weights = [70, 20, 7, 3]  # GET is most common

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1",
        "Mozilla/5.0 (X11; Linux x86_64) Firefox/121.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile/15E148",
        "Mozilla/5.0 (iPad; CPU OS 17_0) AppleWebKit/605.1.15"
    ]

    # Write full dataset
    csv_path = output_path / "access_logs.csv"
    user_stream_path = output_path / "user_stream.txt"
    url_stream_path = output_path / "url_stream.txt"

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile, \
         open(user_stream_path, 'w', encoding='utf-8') as user_file, \
         open(url_stream_path, 'w', encoding='utf-8') as url_file:

        fieldnames = ['timestamp', 'user_id', 'url', 'request_type', 'status_code', 'user_agent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(total_events):
            # Generate event with Zipfian distribution
            user_idx = user_zipf.generate()
            url_idx = url_zipf.generate()

            user_id = user_pool[user_idx]
            url = url_pool[url_idx]

            # Track frequencies and unique counts
            user_frequency[user_id] = user_frequency.get(user_id, 0) + 1
            url_frequency[url] = url_frequency.get(url, 0) + 1
            unique_users_seen.add(user_id)
            unique_urls_seen.add(url)

            # Generate other fields
            timestamp = start_time + timedelta(seconds=i)
            request_type = random.choices(request_types, weights=request_weights)[0]
            status_code = random.choices([200, 201, 304, 400, 404, 500],
                                        weights=[70, 5, 10, 5, 8, 2])[0]
            user_agent = random.choice(user_agents)

            # Write to CSV
            writer.writerow({
                'timestamp': timestamp.isoformat(),
                'user_id': user_id,
                'url': url,
                'request_type': request_type,
                'status_code': status_code,
                'user_agent': user_agent
            })

            # Write to stream files
            user_file.write(f"{user_id}\n")
            url_file.write(f"{url}\n")

            if (i + 1) % 100000 == 0:
                print(f"  Generated {i + 1:,} events...")

    print(f"[OK] Wrote full dataset to {csv_path}")
    print(f"[OK] Wrote user stream to {user_stream_path}")
    print(f"[OK] Wrote URL stream to {url_stream_path}")

    # Generate ground truth statistics
    ground_truth = {
        "total_events": total_events,
        "unique_users_actual": len(unique_users_seen),
        "unique_urls_actual": len(unique_urls_seen),
        "unique_users_pool": unique_users,
        "unique_urls_pool": unique_urls,
        "zipf_alpha": zipf_alpha,
        "seed": seed,
        "top_10_users": sorted(user_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_10_urls": sorted(url_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
        "user_frequency_distribution": {
            "min": min(user_frequency.values()),
            "max": max(user_frequency.values()),
            "median": sorted(user_frequency.values())[len(user_frequency) // 2]
        },
        "url_frequency_distribution": {
            "min": min(url_frequency.values()),
            "max": max(url_frequency.values()),
            "median": sorted(url_frequency.values())[len(url_frequency) // 2]
        }
    }

    # Save ground truth
    ground_truth_path = output_path / "ground_truth.json"
    with open(ground_truth_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"[OK] Wrote ground truth to {ground_truth_path}")
    print(f"\nGround Truth Summary:")
    print(f"  Total events: {total_events:,}")
    print(f"  Unique users: {len(unique_users_seen):,} (from pool of {unique_users:,})")
    print(f"  Unique URLs: {len(unique_urls_seen):,} (from pool of {unique_urls:,})")
    print(f"  Most frequent user: {ground_truth['top_10_users'][0][0]} ({ground_truth['top_10_users'][0][1]:,} requests)")
    print(f"  Most frequent URL: {ground_truth['top_10_urls'][0][0]} ({ground_truth['top_10_urls'][0][1]:,} requests)")

    return ground_truth


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic web server log dataset for testing probabilistic data structures"
    )

    parser.add_argument(
        '--events', '-n',
        type=int,
        default=1_000_000,
        help='Total number of log entries to generate (default: 1,000,000)'
    )

    parser.add_argument(
        '--users', '-u',
        type=int,
        default=50_000,
        help='Number of unique users (default: 50,000)'
    )

    parser.add_argument(
        '--urls',
        type=int,
        default=1_000,
        help='Number of unique URLs (default: 1,000)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Zipfian distribution parameter (default: 1.0, higher = more skewed)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )

    parser.add_argument(
        '--sizes',
        action='store_true',
        help='Generate multiple dataset sizes (10K, 100K, 1M)'
    )

    args = parser.parse_args()

    if args.sizes:
        # Generate multiple sizes for scalability testing
        sizes = [
            (10_000, 1_000, 100, "small"),
            (100_000, 10_000, 500, "medium"),
            (1_000_000, 50_000, 1_000, "large")
        ]

        for events, users, urls, size_name in sizes:
            print(f"\n{'=' * 60}")
            print(f"Generating {size_name} dataset ({events:,} events)...")
            print('=' * 60)

            output_dir = f"{args.output}/{size_name}"
            generate_dataset(
                total_events=events,
                unique_users=users,
                unique_urls=urls,
                zipf_alpha=args.alpha,
                seed=args.seed,
                output_dir=output_dir
            )
    else:
        # Generate single dataset with specified parameters
        generate_dataset(
            total_events=args.events,
            unique_users=args.users,
            unique_urls=args.urls,
            zipf_alpha=args.alpha,
            seed=args.seed,
            output_dir=args.output
        )

    print("\n[OK] Dataset generation complete!")


if __name__ == "__main__":
    main()
