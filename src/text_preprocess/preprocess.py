import pandas as pd
from sklearn.model_selection import train_test_split

def create_single_input(post, comment):
    """
    Concatenates a post and a single target comment into a single input string.
    """
    combined_input = f"{post} [SEP] {comment}"
    return combined_input


def create_thread_input(post, comments, target_idx):
    """
    Concatenates a post with all comments up to the target comment.
    """
    all_comments = " [SEP] ".join(comments[:target_idx + 1])
    combined_input = f"{post} [SEP] {all_comments}"
    return combined_input


def preprocess_dataframe(df, post_col='post', comment_col='comment', comments_col=None, target_idx_col=None):
    """
    Applies text preprocessing to a DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - post_col: Name of the column containing post text.
    - comment_col: Name of the column containing the target comment text (for single input).
    - comments_col: Name of the column containing all comments up to the target comment (for thread input).
    - target_idx_col: Name of the column containing the target comment index (for thread input).
    """
    if comments_col and target_idx_col:
        # If comments_col and target_idx_col are provided, use create_thread_input
        df['combined_input'] = df.apply(lambda x: create_thread_input(x[post_col], x[comments_col], x[target_idx_col]), axis=1)
    else:
        # Otherwise, use create_single_input
        df['combined_input'] = df.apply(lambda x: create_single_input(x[post_col], x[comment_col]), axis=1)
    
    return df


def create_text_label_df(df):
    # Step 1: Drop rows where 'meta.success' is NaN
    data = df[['text', 'meta.success']].dropna(subset=['meta.success'])

    # Step 2 & 4: Create 'label' column from 'meta.success' with integer type
    data['label'] = data['meta.success'].apply(lambda x: int(x == 1.0))

    # Step 3: Drop the 'meta.success' column
    data = data.drop(['meta.success'], axis=1)
    
    return data


def split_and_save_data(data, train_size=0.6, test_size=0.2, validation_size=0.2, random_state=42, save_path='../../data/'):
    """
    Splits the dataset into training, validation, and test sets, and saves them to CSV files.
    """
    # Ensure the sizes sum up to 1
    if train_size + test_size + validation_size != 1:
        raise ValueError("train_size, test_size, and validation_size must sum up to 1.")

    # Initial split to separate out the training data
    train_data, temp_data = train_test_split(data, test_size=test_size + validation_size, random_state=random_state)

    # Adjust temporary test size for secondary split
    temp_test_size = test_size / (test_size + validation_size)

    # Split the temporary dataset into validation and test datasets
    validation_data, test_data = train_test_split(temp_data, test_size=temp_test_size, random_state=random_state)

    # Ensure the 'label' column is integer type
    train_data['label'] = train_data['label'].astype(int)
    validation_data['label'] = validation_data['label'].astype(int)
    test_data['label'] = test_data['label'].astype(int)

    # Save to CSV files
    train_data.to_csv(f'{save_path}train.csv', index=False)
    validation_data.to_csv(f'{save_path}validation.csv', index=False)
    test_data.to_csv(f'{save_path}test.csv', index=False)

    print(f"Data split and saved to {save_path}")
