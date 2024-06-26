start_task_context_prompt = """  
As the Delta Classifier (DC), your role is to analyze comments within Reddit's Change My View (CMV) threads, assessing their potential to change the original poster's (OP) viewpoint, which is indicated by the awarding of a delta. 
A delta signifies a comment has successfully introduced a new perspective, presented compelling evidence, or logically countered the OP's original stance, leading to a change in their viewpoint.
Classify the comment text into one of the following categories only: 0. comment would not get a delta 1. comment would get a delta.
"""

end_task_context_prompt = """  
Your keen judgment, coupled with these examples, will help us recognize insightful contributions effectively. Thank you for your assistance!  
"""

# --------- Zero-shot templates ---------
template_zero_shot_only_comment = """
    {start_task_context_prompt}
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.
    comment = {comment} 
    """

template_zero_shot_only_post = """
"""

template_zero_shot_only_parent = """
"""

template_zero_shot_with_post_and_parent = """
"""

# --------- Single-shot templates ---------

template_single_shot_only_pos_comment = """  
    {start_task_context_prompt}  
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta  
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.  

    Example:  
    Comment positive: {example_comment_pos}
    1

    Input Comment: {comment}  
"""

template_single_shot_only_neg_comment = """  
    {start_task_context_prompt}  
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta  
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.  

    Example:  
    Comment negative: {example_comment_neg}
    0

    Input Comment: {comment}  
"""

template_single_shot_only_comment = """  
    {start_task_context_prompt}  
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta  
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.  

    Example:  
    Comment positive: {example_comment_pos}
    1
    
    Comment negative: {example_comment_neg}
    0
    
    Input Comment: {comment}  
"""

template_single_shot_with_post = """
"""

template_single_shot_with_parent = """
"""

template_single_shot_with_post_and_parent = """
"""


# --------- Few-shot templates ---------

template_few_shot_only_pos_comment = """  
    {start_task_context_prompt}  
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta  
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.  

    Example:  
    Comment positive: {example_comment_1_pos}
    1
    
    Comment positive: {example_comment_2_pos}
    1
    
    Comment positive: {example_comment_3_pos}
    1

    Input Comment: {comment}  
"""

template_few_shot_only_neg_comment = """  
    {start_task_context_prompt}  
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta  
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.  

    Example:  
    Comment positive: {example_comment_1_neg}
    0
    
    Comment positive: {example_comment_2_neg}
    0
    
    Comment positive: {example_comment_3_neg}
    0

    Input Comment: {comment}  
"""

template_few_shot_only_comment = """  
    {start_task_context_prompt}  
    Act as a highly intelligent delta predictor chatbot and classify the given comment text into one of the following categories only 0. comment would not get a delta 1. comment would get a delta  
    Do not code. Return only one number answer with only the number that represent if a comment would get the delta or not.  

    Example:  
    Comment positive: {example_comment_1_pos}
    1
    
    Comment positive: {example_comment_2_pos}
    1
    
    Comment positive: {example_comment_3_pos}
    1
    
    Comment positive: {example_comment_1_neg}
    0
    
    Comment positive: {example_comment_2_neg}
    0
    
    Comment positive: {example_comment_3_neg}
    0

    Input Comment: {comment}  
"""
