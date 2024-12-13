import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import argparse
import os
import logging
import sys
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, model_path: str):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                   "cuda" if torch.cuda.is_available() else 
                                   "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        self.label_names = ['technical', 'non-technical']
        
    def predict(self, texts: List[str], batch_size: int = 8) -> Tuple[List[str], List[float]]:
        predictions = []
        confidence_scores = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            encodings = {key: val.to(self.device) for key, val in encodings.items()}
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_conf = probs.max(dim=-1).values.cpu().numpy()
            
            # Convert to labels
            batch_pred_labels = [self.label_names[pred] for pred in batch_preds]
            
            predictions.extend(batch_pred_labels)
            confidence_scores.extend(batch_conf)
        
        return predictions, confidence_scores

class TextClassificationAnalyzer:
    def __init__(self, results_df: pd.DataFrame, output_dir: str):
        self.results_df = results_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_statistics(self) -> Dict:
        stats = {}
        
        # Basic counts
        total_samples = len(self.results_df)
        tech_samples = len(self.results_df[self.results_df['predicted_label'] == 'technical'])
        nontech_samples = len(self.results_df[self.results_df['predicted_label'] == 'non-technical'])
        
        # Calculate percentages
        tech_percentage = (tech_samples / total_samples) * 100
        nontech_percentage = (nontech_samples / total_samples) * 100
        
        # Average confidence scores
        avg_confidence_tech = self.results_df[self.results_df['predicted_label'] == 'technical']['confidence'].mean()
        avg_confidence_nontech = self.results_df[self.results_df['predicted_label'] == 'non-technical']['confidence'].mean()
        
        # Compile statistics
        stats['total_samples'] = total_samples
        stats['technical'] = {
            'count': tech_samples,
            'percentage': tech_percentage,
            'avg_confidence': avg_confidence_tech
        }
        stats['non_technical'] = {
            'count': nontech_samples,
            'percentage': nontech_percentage,
            'avg_confidence': avg_confidence_nontech
        }
        
        return stats
    
    def create_visualizations(self):
        plt.style.use('default')
        
        # 1. Distribution pie chart
        plt.figure(figsize=(10, 6))
        labels = ['Technical', 'Non-Technical']
        sizes = [
            len(self.results_df[self.results_df['predicted_label'] == 'technical']),
            len(self.results_df[self.results_df['predicted_label'] == 'non-technical'])
        ]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        plt.title('Distribution of Technical vs Non-Technical Content')
        plt.savefig(os.path.join(self.output_dir, 'distribution_pie.png'))
        plt.close()
        
        # 2. Confidence distribution
        plt.figure(figsize=(10, 6))
        tech_conf = self.results_df[self.results_df['predicted_label'] == 'technical']['confidence']
        nontech_conf = self.results_df[self.results_df['predicted_label'] == 'non-technical']['confidence']
        
        plt.boxplot([tech_conf, nontech_conf], labels=['Technical', 'Non-Technical'])
        plt.title('Confidence Scores Distribution')
        plt.ylabel('Confidence Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, 'confidence_distribution.png'))
        plt.close()
    
    def save_detailed_report(self, stats: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'analysis_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Text Classification Analysis Report ===\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Overall Statistics:\n")
            f.write(f"Total samples analyzed: {stats['total_samples']}\n\n")
            
            f.write("2. Technical Content:\n")
            f.write(f"Count: {stats['technical']['count']}\n")
            f.write(f"Percentage: {stats['technical']['percentage']:.1f}%\n")
            f.write(f"Average confidence: {stats['technical']['avg_confidence']:.3f}\n\n")
            
            f.write("3. Non-Technical Content:\n")
            f.write(f"Count: {stats['non_technical']['count']}\n")
            f.write(f"Percentage: {stats['non_technical']['percentage']:.1f}%\n")
            f.write(f"Average confidence: {stats['non_technical']['avg_confidence']:.3f}\n\n")
            
            f.write("4. High Confidence Predictions:\n")
            high_conf = self.results_df[self.results_df['confidence'] > 0.9]
            f.write(f"Number of predictions with >90% confidence: {len(high_conf)}\n")
            f.write(f"Percentage of high confidence predictions: {(len(high_conf)/len(self.results_df))*100:.1f}%\n")

def process_input_file(file_path: str) -> List[str]:
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts = [line.strip() for line in file if line.strip()]
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'text' not in df.columns:
                raise ValueError("CSV file must contain a 'text' column")
            texts = df['text'].tolist()
        else:
            raise ValueError("Unsupported file format. Use .txt or .csv")
        return texts
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        raise

def save_predictions(texts: List[str], 
                    predictions: List[str], 
                    confidence_scores: List[float], 
                    output_dir: str,
                    true_labels: Optional[List[str]] = None):
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'text': texts,
        'predicted_label': predictions,
        'confidence': confidence_scores
    })
    
    if true_labels:
        results_df['true_label'] = true_labels
    
    results_path = os.path.join(output_dir, 'classification_results.csv')
    results_df.to_csv(results_path, index=False)
    
    analyzer = TextClassificationAnalyzer(results_df, output_dir)
    stats = analyzer.generate_statistics()
    
    analyzer.create_visualizations()
    analyzer.save_detailed_report(stats)
    
    logger.info("\nClassification Summary:")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Technical content: {stats['technical']['count']} ({stats['technical']['percentage']:.1f}%)")
    logger.info(f"Non-technical content: {stats['non_technical']['count']} ({stats['non_technical']['percentage']:.1f}%)")
    logger.info(f"\nResults and analysis saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Classify technical vs non-technical texts with detailed analysis')
    parser.add_argument('--model_path', type=str, default='models/best_model_final',
                        help='Path to the saved model directory')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input file (.txt)')
    parser.add_argument('--output_dir', type=str, default='classification_results',
                        help='Directory to save the results and analysis')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--true_labels_file', type=str,
                        help='Optional file containing true labels for evaluation')
    
    args = parser.parse_args()
    
    try:
        classifier = TextClassifier(args.model_path)
        
        logger.info("Reading input file...")
        texts = process_input_file(args.input_file)
        
        true_labels = None
        if args.true_labels_file:
            with open(args.true_labels_file, 'r') as f:
                true_labels = [line.strip() for line in f if line.strip()]
            if len(true_labels) != len(texts):
                raise ValueError("Number of true labels doesn't match number of texts")
        
        logger.info("Making predictions...")
        predictions, confidence_scores = classifier.predict(texts, args.batch_size)
        
        save_predictions(texts, predictions, confidence_scores, args.output_dir, true_labels)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()