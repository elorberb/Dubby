# TODO Inbar: here we would add the templates we think could fit the task

template_single_shot = """Classify the comments as receiving a delta or not.

1. Comment: 'This is a great point, I had not considered it.'
   Classification: 1

your answer is:
   Comment: '{}'
   Classification:"""

template_few_shots = """Classify the comments as receiving a delta or not.

1. Comment: 'This is a great point, I had not considered it.'
   Classification: 1

2. Comment: 'I disagree but don't have evidence to back it up.'
   Classification: 0

your answer is:
   Comment: '{}'
   Classification:"""

# TODO Inbar: add more informative templates here - try to find good examples from the data
