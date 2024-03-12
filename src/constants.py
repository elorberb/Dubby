classification_model_names = [
    'distilbert-base-uncased',
    'FacebookAI/roberta-base',
    'falkne/storytelling-change-my-view-en'
    'google-t5/t5-small',
]

model_names = [
    'mistralai/Mistral-7B-v0.1',
    'Rocketknight1/falcon-rw-1b',
    ''
]

# Define huggingface models default configurations
classification_default_config = {
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'num_train_epochs': 5,
    'weight_decay': 0.01,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'no',
    'load_best_model_at_end': False,
}

# Paths to datasets used in hg classification task
train_dataset_path = 'data/train_data.csv'
test_dataset_path = 'data/test_data.csv'


# Paths to samples datasets in hg classification task
train_dataset_path = 'data/sampled_train_data.csv'
test_dataset_path = 'data/sampled_test_data.csv'