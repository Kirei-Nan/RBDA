import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
import re
import os
from torch.utils.data import Dataset

# Create directories for model saving
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Ensure NLTK resources are available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Check available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda")
else:
    device = torch.device("cpu")
    print("Using device: cpu")

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device.type == 'mps':
        torch.manual_seed(seed)

set_seed(42)

# Synonym replacement for data augmentation
def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            synonym = synonym.replace('_', ' ')
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence

# Prepare the dataset
def prepare_dataset():
    data = [
        # Technical cases
        {'text': 'How do I parse JSON in Python? I need to extract values from a JSON string.', 'label': 0},
        {'text': 'Getting NullPointerException in Java while accessing ArrayList. Here is my code...', 'label': 0},
        {'text': 'What is the difference between inner join and left join in SQL?', 'label': 0},
        {'text': 'How to configure nginx reverse proxy for my Django application?', 'label': 0},
        {'text': 'Git merge vs rebase: When should I use each one?', 'label': 0},
        {'text': 'Implementing binary search tree in C++: how to handle duplicate values?', 'label': 0},
        {'text': 'What is the time complexity of quicksort? How does it compare to mergesort?', 'label': 0},
        {'text': 'React useEffect not working as expected with dependencies array', 'label': 0},
        {'text': 'How to implement authentication in Spring Boot with JWT?', 'label': 0},
        {'text': 'Best practices for designing RESTful APIs with proper error handling', 'label': 0},
        {'text': 'Understanding pointers and memory management in C', 'label': 0},
        {'text': 'Explain the Model-View-Controller (MVC) architecture pattern', 'label': 0},
        {'text': 'How to optimize SQL queries for better performance', 'label': 0},
        {'text': 'Differences between TCP and UDP protocols', 'label': 0},
        {'text': 'Setting up continuous integration and deployment pipelines with Jenkins', 'label': 0},
        {'text': 'An introduction to Kubernetes and container orchestration', 'label': 0},
        {'text': 'How to handle exceptions and errors in Python', 'label': 0},
        {'text': 'Using machine learning algorithms for predictive analytics', 'label': 0},
        {'text': 'Implementing OAuth2 authentication in a web application', 'label': 0},
        {'text': 'What are the SOLID principles in object-oriented programming?', 'label': 0},
        {'text': 'Understanding recursion in computer programming', 'label': 0},
        {'text': 'How to set up a virtual environment in Python', 'label': 0},
        {'text': 'Implementing linked lists in JavaScript', 'label': 0},
        {'text': 'Best practices for exception handling in Java', 'label': 0},
        {'text': 'Introduction to multi-threading in C++', 'label': 0},
        {'text': 'How to use GitHub Actions for CI/CD pipelines', 'label': 0},
        {'text': 'Deploying a Node.js application on AWS EC2', 'label': 0},
        {'text': 'Understanding closures in JavaScript', 'label': 0},
        {'text': 'Implementing responsive design with CSS Flexbox', 'label': 0},
        {'text': 'Database normalization: 1NF, 2NF, 3NF explained', 'label': 0},
        
        # Non-technical cases
        {'text': 'How to deal with a difficult project manager? Need career advice.', 'label': 1},
        {'text': 'Best practices for conducting technical interviews as a team lead?', 'label': 1},
        {'text': 'How to maintain work-life balance as a software developer?', 'label': 1},
        {'text': 'Managing conflicts between team members in an agile environment', 'label': 1},
        {'text': 'How to improve communication between developers and product managers?', 'label': 1},
        {'text': 'What programming language should I learn first as a beginner?', 'label': 1},
        {'text': 'Tips for staying motivated while learning to code', 'label': 1},
        {'text': 'Should I accept a counter offer from my current company?', 'label': 1},
        {'text': 'How to negotiate salary for a senior developer position?', 'label': 1},
        {'text': 'Recommendations for standing desk and ergonomic equipment?', 'label': 1},
        {'text': 'Effective strategies for remote team collaboration', 'label': 1},
        {'text': 'How to handle stress and burnout in the workplace', 'label': 1},
        {'text': 'The importance of emotional intelligence in leadership', 'label': 1},
        {'text': 'Best practices for time management and productivity', 'label': 1},
        {'text': 'Career growth paths for software engineers', 'label': 1},
        {'text': 'Building a positive company culture in startups', 'label': 1},
        {'text': 'Understanding the impact of diversity and inclusion in tech', 'label': 1},
        {'text': 'How to prepare for a behavioral interview', 'label': 1},
        {'text': 'The role of mentorship in professional development', 'label': 1},
        {'text': 'Strategies for effective networking in the tech industry', 'label': 1},
        {'text': 'Tips for public speaking and presentations', 'label': 1},
        {'text': 'How to lead effective meetings with cross-functional teams', 'label': 1},
        {'text': 'Workplace ethics and professionalism', 'label': 1},
        {'text': 'Dealing with imposter syndrome in tech careers', 'label': 1},
        {'text': 'Financial planning for freelancers and contractors', 'label': 1},
        {'text': 'Understanding stock options and equity compensation', 'label': 1},
        {'text': 'How to build a personal brand as a developer', 'label': 1},
        {'text': 'Navigating office politics and organizational hierarchies', 'label': 1},
        {'text': 'Balancing multiple projects and deadlines', 'label': 1},
        {'text': 'The impact of remote work on team dynamics', 'label': 1},
    ]
    return pd.DataFrame(data)

# Data augmentation
def augment_data(df):
    augmented_texts = []
    augmented_labels = []
    for index, row in df.iterrows():
        text = row['text']
        label = row['label']
        augmented_text = synonym_replacement(text, n=2)
        augmented_texts.append(augmented_text)
        augmented_labels.append(label)
    augmented_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
    df = pd.concat([df, augmented_df], ignore_index=True)
    return df

# Define dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, logits[:,1])
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': auc
    }

def main():
    # Prepare and augment dataset
    df = prepare_dataset()
    df = augment_data(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define label names and load tokenizer
    label_names = ['technical', 'non-technical']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Cross-validation setup
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_score = 0
    
    for fold, (train_index, eval_index) in enumerate(kf.split(df['text'], df['label']), 1):
        print(f"\n===== Fold {fold} =====")
        
        # Prepare datasets
        train_texts = df.iloc[train_index]['text'].tolist()
        train_labels = df.iloc[train_index]['label'].tolist()
        eval_texts = df.iloc[eval_index]['text'].tolist()
        eval_labels = df.iloc[eval_index]['label'].tolist()
        
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)
        
        # Initialize model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)
        
        # Freeze layers
        for name, param in model.named_parameters():
            if 'bert.encoder.layer' in name:
                layer_num = int(re.findall(r'\d+', name.split('.')[3])[0])
                if layer_num < 6:
                    param.requires_grad = False
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/fold_{fold}',
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model='roc_auc',
            greater_is_better=True,
            seed=42,
            dataloader_num_workers=0,
        )
        
        # Create and train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        eval_result = trainer.evaluate()
        
        # Save the best model
        if eval_result['eval_roc_auc'] > best_score:
            best_score = eval_result['eval_roc_auc']
            best_model = model
            # Save the model
            model_save_path = f'models/best_model_fold_{fold}'
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"New best model saved with ROC AUC: {best_score:.4f}")

    # Save final best model
    final_model_path = 'models/best_model_final'
    best_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nFinal best model saved to {final_model_path}")
    print(f"Best ROC AUC Score: {best_score:.4f}")

if __name__ == "__main__":
    main()