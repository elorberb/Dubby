from hg_classification_pipe import train_and_evaluate
import constants as const

# Iterate over models and train
for model_name in const.classification_model_names:
    tokenizer_name = model_name
    output_dir = "../results/classification"
    print(f"Training model: {model_name}")
    train_and_evaluate(
        model_name=model_name,
        train_dataset_path=const.train_dataset_path,
        test_dataset_path=const.test_dataset_path,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        **const.classification_default_config
    )
    