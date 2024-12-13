# Technical Text Classification Project

This project uses BERT to classify text as either technical or non-technical. It includes both training and prediction components.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- scikit-learn
- pandas
- matplotlib
- seaborn
- tqdm

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv text_classifier_env

# Activate virtual environment
source text_classifier_env/bin/activate  # For Unix/macOS
# or
text_classifier_env\Scripts\activate  # For Windows
```

2. Install required packages:
```bash
pip3 install torch torchvision torchaudio
pip3 install transformers
pip3 install pandas
pip3 install scikit-learn
pip3 install nltk
pip3 install matplotlib
pip3 install seaborn
pip3 install tqdm
pip3 install accelerate
```

## Project Structure

```
.
├── model.py           # Training script
├── predict.py         # Prediction script
├── models/           # Directory for saved models
└── results/          # Directory for training results
```

## Usage

### Training

To train the model:

```bash
python3 model.py
```

This will:
- Load and prepare the dataset
- Perform 5-fold cross-validation
- Save the best model to `models/best_model_final`
- Create training results in the `results/` directory

### Prediction

To make predictions on new text:

```bash
python3 predict.py --input_file your_texts.txt --output_dir prediction_results
```

Parameters:
- `--input_file`: Path to input file (.txt or .csv)
- `--output_dir`: Directory to save results (default: 'classification_results')
- `--model_path`: Path to model (default: 'models/best_model_final')
- `--batch_size`: Batch size for processing (default: 8)

### Input Format

Text files should have one text per line:
```
How do I implement a binary search tree?
Best practices for team management
...
```

### Output Format

The prediction script generates:
1. `classification_results.csv`: Contains columns for:
   - text: Input text
   - predicted_label: Technical or non-technical
   - confidence: Prediction confidence score

2. `analysis_report_[timestamp].txt`: Detailed analysis including:
   - Overall statistics
   - Technical/non-technical content distribution
   - Confidence score analysis

3. Visualizations:
   - `distribution_pie.png`: Distribution of technical vs non-technical content
   - `confidence_distribution.png`: Confidence scores distribution


