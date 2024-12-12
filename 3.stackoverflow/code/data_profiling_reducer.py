import sys
from collections import defaultdict


def main():
    try:
        missing_counts = defaultdict(int)
        invalid_type_counts = defaultdict(int)
        score_counts = defaultdict(int)
        total_scores = 0
        score_frequency = defaultdict(int)

        for line in sys.stdin:
            try:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 3:
                    continue

                key, field, value = parts

                if key == "MISSING":
                    missing_counts[field] += int(value)
                elif key == "INVALID_TYPE":
                    invalid_type_counts[field] += int(value)
                elif key == "SCORE":
                    score = int(field)
                    score_counts['total'] += int(value)
                    total_scores += score
                    score_frequency[score] += 1

            except Exception as e:
                sys.stderr.write(f"Error processing line {line}: {str(e)}\n")
                continue

        # Output results
        for field, count in missing_counts.items():
            sys.stdout.write(f"MISSING\t{field}\t{count}\n")

        for field, count in invalid_type_counts.items():
            sys.stdout.write(f"INVALID_TYPE\t{field}\t{count}\n")

        if score_counts['total'] > 0:
            average_score = total_scores / score_counts['total']
            sys.stdout.write(f"AVERAGE_SCORE\t{average_score:.2f}\n")

        for score, freq in sorted(score_frequency.items()):
            sys.stdout.write(f"SCORE_DISTRIBUTION\t{score}\t{freq}\n")

    except Exception as e:
        sys.stderr.write(f"Error in reducer: {str(e)}\n")


if __name__ == "__main__":
    main()