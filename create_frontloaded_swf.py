#!/usr/bin/env python3
"""
Create a front-loaded SWF dataset where all jobs arrive at time 0.

- Preserves all original fields (including an optional 19th carbon field)
- Copies header/comment lines as-is
- Optionally limits to the first N jobs

Usage:
  python create_frontloaded_swf.py --input ./data/lublin_256.swf \
                                   --output ./data/lublin_256_frontloaded.swf \
                                   --max-jobs 2048
"""

import argparse
import os


def frontload_swf(input_file: str, output_file: str, max_jobs: int | None = None) -> str:
    """
    Write a copy of the SWF with submit_time set to 0 for all jobs.

    Args:
        input_file: Source SWF path
        output_file: Destination SWF path
        max_jobs: If provided, limits number of job lines copied (first N jobs)

    Returns:
        output_file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    jobs_written = 0
    total_jobs_seen = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            raw = line.rstrip('\n')

            # Preserve comments/headers
            if raw.startswith(';') or raw.strip() == '':
                outfile.write(raw + '\n')
                continue

            # Stop if we've reached the requested job cap
            if max_jobs is not None and jobs_written >= max_jobs:
                break

            total_jobs_seen += 1
            fields = raw.split()

            # Basic guard: SWF should have at least 18 fields; allow >=18 (19th may be carbon)
            if len(fields) < 18:
                # Write as-is to avoid corrupting unknown format
                outfile.write(raw + '\n')
                jobs_written += 1
                continue

            # Set submit_time (field index 1) to 0
            fields[1] = '0'

            # Optionally also zero the "think time" (field 17) to avoid enforced spacing
            try:
                # Only set to 0 if it was non-negative number; leave negative semantics intact
                if int(fields[17]) >= 0:
                    fields[17] = '0'
            except ValueError:
                # If not numeric, leave as-is
                pass

            outfile.write(' '.join(fields) + '\n')
            jobs_written += 1

    print(f"Input jobs scanned: {total_jobs_seen}")
    print(f"Jobs written:       {jobs_written}")
    print(f"Frontloaded file:   {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Front-load SWF: all jobs submit at time 0")
    parser.add_argument('--input', required=True, help='Path to source SWF file')
    parser.add_argument('--output', required=True, help='Path to destination SWF file')
    parser.add_argument('--max-jobs', type=int, default=None, help='Limit to first N jobs (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input SWF not found: {args.input}")

    frontload_swf(args.input, args.output, args.max_jobs)


if __name__ == '__main__':
    main()






