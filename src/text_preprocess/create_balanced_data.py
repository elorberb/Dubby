import pandas as pd

def balance_dataset(csv_file_path):
    
    df = pd.read_csv(csv_file_path)
    
    # Assuming 'label' is the name of the column containing labels
    count_label_0 = df[df['label'] == 0].shape[0]
    count_label_1 = df[df['label'] == 1].shape[0]
    
    # Calculate the number of records to sample for balancing
    n_samples = min(count_label_0, count_label_1)
    
    # Downsample the majority label
    df_label_0_downsampled = df[df['label'] == 0].sample(n=n_samples, random_state=42)
    df_label_1_downsampled = df[df['label'] == 1].sample(n=n_samples, random_state=42)
    
    # Concatenate the downsampled dataframes to get a balanced dataset
    df_balanced = pd.concat([df_label_0_downsampled, df_label_1_downsampled])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the balanced dataset to a new CSV file
    balanced_csv_file_path = csv_file_path.replace('.csv', '_balanced.csv')
    df_balanced.to_csv(balanced_csv_file_path, index=False)
    
    print(f'Balanced dataset saved to {balanced_csv_file_path}')
    return balanced_csv_file_path

# Paths to your datasets
test_csv_path = '/home/etaylor/code_projects/dubby/data/updated_data/test_title_comment.csv'
train_csv_path = '/home/etaylor/code_projects/dubby/data/updated_data/train_title_comment.csv'

# Balance each dataset
balanced_test_csv_path = balance_dataset(test_csv_path)
balanced_train_csv_path = balance_dataset(train_csv_path)
