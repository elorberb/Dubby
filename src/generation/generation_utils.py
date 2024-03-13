def generate_comments(chain, df, n):
    df_copy = df.copy()
    df_copy['generated_comment'] = None

    for i, row in tqdm(df.head(n).iterrows(), total=n):
        comment = row['text']
        input_vars_values = {'comment': comment}  # Assuming 'comment' is defined in your prompt template

        # Generate the response using the language model
        resp = chain(input_vars_values)
        generated_comment = resp['text']  # Adjust based on the actual key in the response
        df_copy.at[i, 'generated_comment'] = generated_comment

    return df_copy


def run_comment_generation(df, azure_deployment_name, num_records, template_dict, input_vars):
    # Initialize the model with the specified parameters
    chain = initialize_model(azure_deployment_name, template_dict['template_text'], input_vars)
    print("Model initialized.")

    # Generate comments for the specified number of records
    df_copy = generate_comments(chain, df, num_records)

    # Here you can add any post-generation processing, such as saving the results or analyzing the generated comments
    return df_copy
