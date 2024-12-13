from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
import json
from datetime import datetime
import re

class DataProfilingMR(MRJob):
    """MapReduce job for data profiling of HackerNews comments dataset"""
    
    def mapper_data_profiling(self, _, line):
        """Mapper for analyzing data completeness, types, and distributions"""
        try:
            # Parse CSV line
            row = next(csv.reader([line]))
            if len(row) != 7:  # Expected columns: id, title, text, by, score, time, type
                return
                
            id, title, text, by, score, time, type = row
            
            # Data completeness check
            for col_name, value in zip(['id', 'title', 'text', 'by', 'score', 'time', 'type'], row):
                yield f'completeness_{col_name}', (1 if value else 0, 1)
            
            # Data type analysis
            yield f'type_id', ('numeric' if id.isdigit() else 'non_numeric', 1)
            yield f'type_time', ('datetime' if self.is_valid_datetime(time) else 'invalid_datetime', 1)
            
            # Value distribution
            yield f'text_length', (len(text) if text else 0, 1)
            yield f'type_distribution', (type, 1)
            
            # Missing value detection
            for col_name, value in zip(['id', 'title', 'text', 'by', 'score', 'time', 'type'], row):
                if not value or value.strip() == '':
                    yield f'missing_{col_name}', 1
                    
        except Exception as e:
            yield 'parse_error', str(e)
    
    def reducer_data_profiling(self, key, values):
        """Reducer for aggregating profiling statistics"""
        if key.startswith('completeness_'):
            # Calculate completeness ratio
            filled, total = zip(*values)
            completeness_ratio = sum(filled) / sum(total)
            yield key, completeness_ratio
        
        elif key.startswith('type_'):
            # Aggregate type distributions
            type_counts = {}
            for type_val, count in values:
                type_counts[type_val] = type_counts.get(type_val, 0) + count
            yield key, type_counts
            
        elif key.startswith('missing_'):
            # Sum up missing values
            yield key, sum(values)
        
        else:
            # General aggregation
            yield key, list(values)
    
    def is_valid_datetime(self, dt_str):
        """Helper function to validate datetime strings"""
        try:
            datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            return True
        except:
            return False
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_data_profiling,
                  reducer=self.reducer_data_profiling)
        ]


if __name__ == '__main__':
    DataProfilingMR.run()
