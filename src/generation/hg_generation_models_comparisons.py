from hg_classification_pipe import train_and_evaluate

# Define default configurations
default_config = {
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'num_train_epochs': 5,
    'weight_decay': 0.01,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'no',
    'load_best_model_at_end': False,
}

# Model names to compare
model_names = [
    # 'distilbert-base-uncased',
    # 'FacebookAI/roberta-base',
    # 'google-t5/t5-small',
    'falkne/storytelling-change-my-view-en'
    # 'Rocketknight1/falcon-rw-1b',
]


# Paths to datasets
train_dataset_path = 'data/train_data.csv'
test_dataset_path = 'data/test_data.csv'

# Iterate over models and train
for model_name in model_names:
    tokenizer_name = model_name
    output_dir = "../results/classification"
    print(f"Training model: {model_name}")
    train_and_evaluate(
        model_name=model_name,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        **default_config
    )
    