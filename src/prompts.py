start_task_context_prompt = """  
Hello DK (Delta King)! As your current name suggests, your task will be to determine whether posts merit a delta.  
A delta typically signifies a change in opinion or perspective on Reddit.  
"""

end_task_context_prompt = """  
Your keen judgment, coupled with these examples, will help us recognize insightful contributions effectively. Thank you for your assistance!  
"""

# Zero-shot templates
def template_zero_shot_with_post_and_parent(comment, original_post, parent_comment):
    #TODO
    pass


def template_zero_shot_with_post(comment, original_post):
    #TODO
    pass


def template_zero_shot_with_parent(comment, parent_comment):
    #TODO
    pass


def template_zero_shot_only_comment():
    template = """
    Hello DK (Delta King)! As your current name suggests, your task will be to determine whether posts merit a delta.  
    A delta typically signifies a change in opinion or perspective on Reddit.
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.
    comment = {comment} 
    """
    return template


# Single-shot templates
template_single_shot_with_post_and_parent = """Classify the comments as receiving a delta or not.  

Original Post – "..."  
Parent Comment – "..."  
1. Comment: 'This is a great point, I had not considered it.'  
   Classification: 1  

your answer is:  
Original Post – "{}"  
Parent Comment – "{}"  
   Comment: '{}'  
   Classification:"""

template_single_shot_with_post = """Classify the comments as receiving a delta or not.  

Original Post – "..."  
1. Comment: 'This is a great point, I had not considered it.'  
   Classification: 1  

your answer is:  
Original Post – "{}"  
   Comment: '{}'  
   Classification:"""

template_single_shot_with_parent = """Classify the comments as receiving a delta or not.  

Parent Comment – "..."  
1. Comment: 'This is a great point, I had not considered it.'  
   Classification: 1  

your answer is:  
Parent Comment – "{}"  
   Comment: '{}'  
   Classification:"""

template_single_shot_only_comment = """Classify the comments as receiving a delta or not.  

1. Comment: 'This is a great point, I had not considered it.'  
   Classification: 1  

your answer is:  
   Comment: '{}'  
   Classification:"""


# Few-shot templates
def template_few_shots_mixed(examples):
    prompt = f"{start_task_context_prompt}\n\n"
    for idx, example in enumerate(examples, start=1):
        prompt += f"Example {idx}:\n"
        if 'original_post' in example:
            prompt += f"Original Post – '{example['original_post']}'\n"
        if 'parent_comment' in example:
            prompt += f"Parent Comment – '{example['parent_comment']}'\n"
        prompt += f"Comment – '{example['comment']}'\n\n"
    prompt += f"{end_task_context_prompt}"
    return prompt


# Example of how to use the few-shot template with a mix of examples
few_shot_examples = [
    {"comment": "...", "parent_comment": "...", "original_post": "..."},
    {"comment": "..."},  # Example with only the comment
    {"comment": "...", "parent_comment": "..."},  # Example with comment and parent comment
    {"comment": "...", "original_post": "..."}  # Example with comment and original post
]

template_few_shots = template_few_shots_mixed(few_shot_examples)

