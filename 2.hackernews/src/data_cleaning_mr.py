from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
import json
from datetime import datetime
import re


class DataCleaningMR(MRJob):
    """MapReduce job for data cleaning and standardization"""

    def mapper_cleaning(self, _, line):
        """Mapper for cleaning and standardizing data"""
        try:
            row = next(csv.reader([line]))
            if len(row) != 7:
                return

            id, title, text, by, score, time, type = row

            # Format standardization
            cleaned_text = self.clean_text(text)
            standardized_time = self.standardize_datetime(time)

            # Output cleaned record
            if id and cleaned_text:  # Basic validation
                yield None, {
                    'id': id,
                    'title': title,
                    'text': cleaned_text,
                    'by': by,
                    'score': score,
                    'time': standardized_time,
                    'type': type.lower() if type else None
                }

        except Exception as e:
            yield 'cleaning_error', str(e)

    def clean_text(self, text):
        """Clean and standardize text content"""
        if not text:
            return None
        # Remove HTML entities
        text = re.sub(r'&#\d+;', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def standardize_datetime(self, dt_str):
        """Standardize datetime format"""
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            return dt.isoformat()
        except:
            return None

    def reducer_cleaning(self, key, values):
        """Reducer for aggregating cleaned records"""
        if key == 'cleaning_error':
            yield key, list(values)
        else:
            # Remove duplicates based on id
            seen_ids = set()
            for record in values:
                if record['id'] not in seen_ids:
                    seen_ids.add(record['id'])
                    yield record['id'], record

    def steps(self):
        return [
            MRStep(mapper=self.mapper_cleaning,
                  reducer=self.reducer_cleaning)
        ]

if __name__ == '__main__':
    DataCleaningMR.run()
