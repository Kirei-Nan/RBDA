import sys
import csv

def is_missing(value):
    return value is None or str(value).strip() == '-' or str(value).strip() == '' or str(value).lower() == 'null'

def main():
    try:
        reader = csv.reader(sys.stdin)
        header = next(reader)  # Skip header row

        for row in reader:
            try:
                if len(row) != 13:
                    continue

                (Id, PostTypeId, ParentId, CreationDate, Score, ViewCount, Title,
                 Tags, Body, OwnerUserId, AnswerCount, CommentCount, FavoriteCount) = row

                # Data completeness check
                for idx, value in enumerate(row):
                    if is_missing(value):
                        sys.stdout.write(f"MISSING\t{header[idx]}\t1\n")

                # Data type analysis
                if not is_missing(Score):
                    try:
                        int(Score)
                        sys.stdout.write(f"SCORE\t{Score}\t1\n")
                    except ValueError:
                        sys.stdout.write(f"INVALID_TYPE\tScore\t1\n")

            except Exception as e:
                sys.stderr.write(f"Error processing row: {str(e)}\n")
                continue

    except Exception as e:
        sys.stderr.write(f"Error in mapper: {str(e)}\n")

if __name__ == "__main__":
    main()