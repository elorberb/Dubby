import argparse
import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import sys

def validate_cuda():
    if torch.cuda.is_available():
        logger.info("CUDA is available. 🚀")
        logger.info(f"CUDA version: {torch.version.cuda}")
        try:
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        except AssertionError as e:
            logger.error("Failed to get the GPU name. Ensure a GPU device is properly installed.")
            sys.exit(1)  # Exiting the process if no GPU device is found
    else:
        logger.error("CUDA is not available. Please check your installation.")
        sys.exit(1)  # Exiting the process if CUDA is not available

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # for generative models likt T5 use this code
    predictions = np.argmax(logits[0], axis=-1)
    # predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_and_evaluate(model_name, train_dataset_path, test_dataset_path, tokenizer_name, output_dir, **config):
    
    # Call the function to validate CUDA
    validate_cuda()
    
    logger.info(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset('csv', data_files=train_dataset_path)['train']
    logger.info(f"Loading test dataset from: {test_dataset_path}")
    test_dataset = load_dataset('csv', data_files=test_dataset_path)['train']
    datasets = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    logger.info("Datasets loaded successfully.")
    
    logger.info(f"Initializing tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer initialized.")
    
    logger.info("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    logger.info("Datasets tokenized.")
    
    logger.info(f"Initializing model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    logger.info("Model initialized.")
    
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.get('learning_rate', 2e-5),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 8),
        num_train_epochs=config.get('num_train_epochs', 3),
        weight_decay=config.get('weight_decay', 0.01),
        save_strategy="no",
        load_best_model_at_end=False,
    )
    logger.info("Training arguments set.")
    
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    logger.info("Trainer initialized.")
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")
    
    logger.info("Evaluating model...")
    evaluation_results = trainer.evaluate()
    logger.info("Evaluation completed. Results:")
    for key, value in evaluation_results.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Saving evaluation results to {output_dir}/{model_name.replace("/", "_")}_eval_results.txt")
    with open(f"{output_dir}/{model_name.replace("/", "_")}_eval_results.txt", "w") as f:
        for key, value in evaluation_results.items():
            f.write(f"{key}: {value}\n")
    logger.info("Results saved.")
    
    checkpoint_output_dir = "/home/etaylor/code_projects/dubby/results/last_model_roberta_checkpoint"
    trainer.save_model(checkpoint_output_dir)  # Save the model to the specified output directory
    
    logger.info("Training and evaluation completed. Freeing up GPU memory...")
    torch.cuda.empty_cache()
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a text classification model with separate training and test datasets.")
    
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--train_dataset_path', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--tokenizer_name', type=str, help='Tokenizer name or path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model and results')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size per device for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size per device for evaluation')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimization')

    args = parser.parse_args()
    
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name
    
    train_and_evaluate(
        model_name=args.model_name,
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        tokenizer_name=args.tokenizer_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay
    )
