import pandas as pd

def sample_and_save(data_path, output_path, sample_fraction=0.1):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Sample 10% of the data
    df_sampled = df.sample(frac=sample_fraction, random_state=42)
    
    # Save the sampled data
    df_sampled.to_csv(output_path, index=False)
    print(f"Sampled data saved to {output_path}")

# Paths to your train and test datasets
train_data_path = "/home/etaylor/code_projects/dubby/data/train.csv"
test_data_path = "/home/etaylor/code_projects/dubby/data/test.csv"

# Paths where you want to save the sampled datasets
sampled_train_data_path = "/home/etaylor/code_projects/dubby/data/sampled_train_data.csv"
sampled_test_data_path = "/home/etaylor/code_projects/dubby/data/sampled_test_data.csv"

# Sample and save the datasets
sample_and_save(train_data_path, sampled_train_data_path)
sample_and_save(test_data_path, sampled_test_data_path)
