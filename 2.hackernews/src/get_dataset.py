#!/usr/bin/env python3
from datasets import load_dataset
import pandas as pd
import os
import gc

def process_dataset_in_chunks():
    """Process dataset in chunks to manage memory"""
    print("Loading dataset...")
    
    # Load the dataset
    dataset = load_dataset("OpenPipe/hacker-news")
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Process in chunks
    chunk_size = 100000  # Adjust this number based on your memory
    total_rows = len(dataset['train'])
    chunks = total_rows // chunk_size + (1 if total_rows % chunk_size else 0)
    
    print(f"Processing {total_rows} rows in {chunks} chunks...")
    
    # Initialize counters
    total_stories = 0
    total_comments = 0
    
    # Process each chunk
    for i in range(chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        print(f"\nProcessing chunk {i+1}/{chunks} (rows {start_idx} to {end_idx})")
        
        # Get chunk of data
        chunk_data = dataset['train'].select(range(start_idx, end_idx))
        df_chunk = pd.DataFrame(chunk_data)
        
        # Select relevant columns
        columns = ['id', 'title', 'text', 'by', 'score', 'time', 'type']
        df_chunk = df_chunk[columns]
        
        # Split into stories and comments
        stories = df_chunk[df_chunk['type'] == 'story'].copy()
        comments = df_chunk[df_chunk['type'] == 'comment'].copy()
        
        # Clean the data
        for col in ['title', 'text']:
            if col in stories.columns:
                stories[col] = stories[col].fillna('')
            if col in comments.columns:
                comments[col] = comments[col].fillna('')
        
        # Save chunk data
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        
        stories.to_csv('data/stories.csv', mode=mode, header=header, index=False)
        comments.to_csv('data/comments.csv', mode=mode, header=header, index=False)
        df_chunk.to_csv('data/hackernews_data.csv', mode=mode, header=header, index=False)
        
        # Update counters
        total_stories += len(stories)
        total_comments += len(comments)
        
        # Clear memory
        del df_chunk, stories, comments
        gc.collect()
        
        print(f"Processed {end_idx-start_idx} rows in chunk {i+1}")
    
    print("\nProcessing complete!")
    print(f"Total stories: {total_stories}")
    print(f"Total comments: {total_comments}")
    
    # Verify file sizes
    for file in ['stories.csv', 'comments.csv', 'hackernews_data.csv']:
        path = os.path.join('data', file)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\nFile {file}: {size_mb:.2f} MB")

def sample_final_data(target_size_mb=1000):  # 1000MB = ~1GB
    """Sample the final data to get desired size"""
    print(f"\nSampling data to approximately {target_size_mb}MB...")
    
    # Read the file in chunks and calculate total size
    total_size = os.path.getsize('data/hackernews_data.csv')
    if total_size < target_size_mb * 1024 * 1024:
        print("Data already smaller than target size. Skipping sampling.")
        return
    
    sample_ratio = (target_size_mb * 1024 * 1024) / total_size
    
    # Sample the data
    sampled_data = pd.read_csv('data/hackernews_data.csv', 
                              nrows=int(sum(1 for _ in open('data/hackernews_data.csv')) * sample_ratio))
    
    # Save sampled data
    sampled_data.to_csv('data/hackernews_data_sampled.csv', index=False)
    os.rename('data/hackernews_data_sampled.csv', 'data/hackernews_data.csv')
    
    print(f"Final data size: {os.path.getsize('data/hackernews_data.csv') / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    try:
        # Process the dataset in chunks
        process_dataset_in_chunks()
        
        # Sample data to get ~1GB
        sample_final_data()
        
    except Exception as e:
        print(f"Error: {e}")
        raise
