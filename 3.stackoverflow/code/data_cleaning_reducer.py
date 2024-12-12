import sys
from collections import defaultdict


def main():
    stats = {
        'total_posts': 0,
        'questions': 0,  # PostTypeId = 1
        'answers': 0,  # PostTypeId = 2
        'posts_with_tags': 0,
        'empty_text': 0,
        'score_distribution': defaultdict(int)
    }

    try:
        for line in sys.stdin:
            try:
                line = line.strip()
                if not line:
                    continue

                # Split input line into fields
                fields = line.split('\t')
                if len(fields) != 6:  # Verify we have all expected fields
                    continue

                # Unpack fields
                post_id, post_type, text, tags, date, score = fields

                # Update statistics
                stats['total_posts'] += 1

                # Count questions and answers
                if post_type == '1':
                    stats['questions'] += 1
                elif post_type == '2':
                    stats['answers'] += 1

                # Count posts with tags
                if tags.strip():
                    stats['posts_with_tags'] += 1

                # Count empty text posts
                if not text.strip():
                    stats['empty_text'] += 1
                    continue

                # Track score distribution
                try:
                    score_val = int(score)
                    stats['score_distribution'][score_val] += 1
                except ValueError:
                    pass

                # Output the valid record
                print(line)

            except Exception as e:
                sys.stderr.write(f"Error processing line: {str(e)}\n")
                continue

        # Output statistics to stderr
        sys.stderr.write("\nData Cleaning Statistics:\n")
        sys.stderr.write(f"Total posts processed: {stats['total_posts']}\n")
        sys.stderr.write(f"Questions: {stats['questions']}\n")
        sys.stderr.write(f"Answers: {stats['answers']}\n")
        sys.stderr.write(f"Posts with tags: {stats['posts_with_tags']}\n")
        sys.stderr.write(f"Posts with empty text: {stats['empty_text']}\n")

        # Output score distribution
        sys.stderr.write("\nScore Distribution:\n")
        for score, count in sorted(stats['score_distribution'].items()):
            sys.stderr.write(f"Score {score}: {count} posts\n")

    except Exception as e:
        sys.stderr.write(f"Error in reducer: {str(e)}\n")


if __name__ == "__main__":
    main()