import sys
import csv
import re
from html import unescape


def is_missing(value):
    return value is None or str(value).strip() == '-' or str(value).strip() == '' or str(value).lower() == 'null'


def clean_text(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', unescape(text))

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)  # Multi-line code blocks
    text = re.sub(r'`[^`]*`', '', text)  # Inline code

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase and trim
    return text.lower().strip()


def clean_tags(tags):
    if is_missing(tags):
        return ''
    # Remove < and > characters and normalize spaces
    return ' '.join(tag.strip() for tag in tags.replace('<', '').replace('>', ' ').split())


def main():
    try:
        # Read input CSV
        reader = csv.reader(sys.stdin)
        header = next(reader)  # Skip header row

        for row in reader:
            try:
                # Skip malformed lines
                if len(row) != 13:
                    continue

                # Unpack fields
                (Id, PostTypeId, ParentId, CreationDate, Score, ViewCount,
                 Title, Tags, Body, OwnerUserId, AnswerCount, CommentCount,
                 FavoriteCount) = row

                # Only process questions (1) and answers (2)
                if PostTypeId not in ['1', '2']:
                    continue

                # Skip if missing critical fields
                if is_missing(Id) or is_missing(PostTypeId) or is_missing(CreationDate):
                    continue

                # Combine and clean title and body text
                text_parts = []
                if not is_missing(Title):
                    text_parts.append(clean_text(Title))
                if not is_missing(Body):
                    text_parts.append(clean_text(Body))

                text_content = ' '.join(text_parts)

                # Skip if no text content after cleaning
                if not text_content:
                    continue

                # Clean tags
                cleaned_tags = clean_tags(Tags)

                # Prepare output row
                cleaned_row = [
                    Id,
                    PostTypeId,
                    text_content,  # Combined cleaned text
                    cleaned_tags,
                    CreationDate,
                    Score or '0'  # Post score (defaulting to 0 if missing)
                ]

                # Output as tab-separated values
                print('\t'.join(str(x) for x in cleaned_row))

            except Exception as e:
                sys.stderr.write(f"Error processing row: {str(e)}\n")
                continue

    except Exception as e:
        sys.stderr.write(f"Error in mapper: {str(e)}\n")


if __name__ == "__main__":
    main()