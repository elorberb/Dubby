from hg_classification_pipe import train_and_evaluate
import src.constants as const

# Iterate over models and train
for model_name in const.classification_model_names:
    tokenizer_name = model_name
    output_dir = "/home/etaylor/code_projects/dubby/results/classification/title_post_comment_father_comment"
    print(f"Training model: {model_name}")
    train_and_evaluate(
        model_name=model_name,
        train_dataset_path=const.train_title_post_comment_father_comment_path,
        test_dataset_path=const.test_title_post_comment_father_comment_path,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        **const.classification_default_config
    )
    