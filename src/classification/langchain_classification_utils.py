import os  
from src import prompts
from src.langchain_utils import run_chain  
from langchain_openai import AzureChatOpenAI  
from langchain.prompts import PromptTemplate  
from langchain.chains import LLMChain  
from dotenv import load_dotenv  
import pandas as pd  
from sklearn.metrics import accuracy_score, precision_score, recall_score  
from tqdm import tqdm
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.metrics import roc_curve, auc  
import numpy as np  
import itertools
from src import comments 
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import AzureChatOpenAI, HuggingFacePipeline


# Load environment variables  
load_dotenv() 
  
  

def initialize_model(hg_model_name=None, azure_deployment=None, template=None, input_vars=None):
    if azure_deployment:
        llm = AzureChatOpenAI(azure_deployment=azure_deployment)
    elif hg_model_name:
        tokenizer = AutoTokenizer.from_pretrained(hg_model_name)
        model = AutoModelForCausalLM.from_pretrained(hg_model_name)
        pipe = HuggingFacePipeline(model=model, tokenizer=tokenizer, task="text-generation")
        llm = pipe
    else:
        raise ValueError("Either azure_deployment or hg_model_name must be specified.")

    if template and input_vars:
        prompt_template = PromptTemplate(input_variables=input_vars, template=template)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        return chain
    else:
        return llm


def clean_text(text, default=None):  
    # Use regular expression to find integers in the text  
    matches = re.findall(r'\d+', text)  
    if matches:  
        # Return the first integer found  
        return int(matches[0])  
    else:  
        # Return the default value if no integer is found  
        return default  
  
def predict_labels(chain, df, n, input_vars_values):  
    df_copy = df.copy()  
    df_copy['predicted_label'] = None  
  
    for i, row in tqdm(df.head(n).iterrows()):  
        comment = row['text']  
        input_vars_values['comment'] = comment  
        resp = chain(input_vars_values)  
        try:  
            # Clean the response text to extract integer  
            pred = clean_text(resp['text'], default=-1)  # Using -1 as the default value  
            df_copy.at[i, 'predicted_label'] = pred  
        except ValueError as e:  
            print(f"Error converting prediction to int for row {i}: {e}")  
            df_copy.at[i, 'predicted_label'] = None  # or some default value  
  
    return df_copy 
  
def calculate_metrics(df_copy, n):  
    true_labels = df_copy['label'].head(n).tolist()  
    predicted_labels = df_copy['predicted_label'].head(n).tolist()  
  
    accuracy = accuracy_score(true_labels, predicted_labels)  
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)  
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)  
  
    return accuracy, precision, recall  
  
def print_metrics(accuracy, precision, recall):  
    print(f"Accuracy: {accuracy}")  
    print(f"Precision: {precision}")  
    print(f"Recall: {recall}")  
  
  
def run_evaluation(df, azure_deployment_name, num_records, template_dict, input_vars, input_vars_values, verbose=False):  
    """
    Runs the evaluation for a given dataframe and input variables.
    :param df: The dataframe to evaluate.
    :param azure_deployment_name: The name of the Azure deployment.
    :param template_dict: The template dictionary.
    :param input_vars: The input variables.
    :param input_vars_values: The input variables values.
    :param num_records: The number of records to evaluate.
    :param verbose: Whether to print the metrics.
    :return: results: The results of the evaluation.
    """
    
    results = {'model': azure_deployment_name,
               'template_type': template_dict['template_type']}  # Dictionary to store all outputs  
  
    # Initialize the model with the specified parameters  
    chain = initialize_model(azure_deployment_name, template_dict['template_text'], input_vars)  
    if verbose:  
        print("Model initialized.")  
  
    # Predict labels for the specified number of records  
    df_copy = predict_labels(chain, df, num_records, input_vars_values)  
    true_labels = df_copy['label'].head(num_records).tolist()  
    predicted_labels = df_copy['predicted_label'].head(num_records).tolist()  
    results['true_labels'] = true_labels  
    results['predicted_labels'] = predicted_labels  
  
    if verbose:  
        print("Predictions made.")  
  
    # Calculate the evaluation metrics  
    accuracy, precision, recall = calculate_metrics(df_copy, num_records)  
    results['metrics'] = {  
        'accuracy': accuracy,  
        'precision': precision,  
        'recall': recall  
    }  
      
    if verbose:  
        print_metrics(accuracy, precision, recall)  
  
    return results

def evaluate_each_results(true_labels, predicted_labels):  
    # Assuming your classes are string or integer  
    unique_classes = np.unique(true_labels)  
    n_classes = len(unique_classes)  
  
    # Convert unique_classes to an array of strings for plotting  
    class_names = unique_classes.astype(str)  
  
    # Calculate confusion matrix  
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)  
  
    # Classification report using the converted class names  
    class_report = classification_report(true_labels, predicted_labels, target_names=class_names)  
  
    # Plot confusion matrix  
    plt.figure(figsize=(6, 4))  
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)  
    plt.title('Confusion Matrix')  
    plt.colorbar()  
    tick_marks = np.arange(n_classes)  
    plt.xticks(tick_marks, class_names, rotation=45)  
    plt.yticks(tick_marks, class_names)  
  
    # Adding numbers to the confusion matrix  
    thresh = conf_matrix.max() / 2.  
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):  
        plt.text(j, i, conf_matrix[i, j],  
                 horizontalalignment="center",  
                 color="white" if conf_matrix[i, j] > thresh else "black")  
  
    plt.tight_layout()  
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    plt.show()  
  
    # Print classification report  
    print(class_report)  
  
    # Binary classification ROC curve  
    if n_classes == 2:  
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels, pos_label=unique_classes[1])  
        roc_auc = auc(fpr, tpr)  
  
        # Plot ROC curve  
        plt.figure()  
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)  
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title('Receiver Operating Characteristic')  
        plt.legend(loc="lower right")  
        plt.show()  
    else:  
        print("ROC curve is not plotted for multi-class classification.") 
