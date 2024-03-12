intro_prompt = """
As an advanced language learning model (LLM), your capability to craft persuasive and engaging responses is crucial across a wide range of discussions and debates. To enhance your effectiveness in changing viewpoints or persuading others, especially in online forums or discussions like Reddit's ChangeMyView, it's important to understand and apply key strategies that increase the likelihood of your argument being accepted and valued by others.

You will be introduced to several points that serve as guidelines or strategies for crafting your responses. These points are derived from an analysis of successful persuasive arguments in online discussions, focusing on how to structure your argument, the tone to use, and the content to include for maximum impact. The goal is to make your argument more compelling, respectful, and convincing to those who may initially disagree with your viewpoint.

Before you receive these points, it's essential to grasp the underlying concept that persuasion is not just about presenting facts or opinions but doing so in a manner that connects with your audience on a logical and emotional level. This involves:

1. *Timeliness*: Entering the conversation early to set the tone and direction.
2. *Engagement*: Participating in a meaningful back-and-forth exchange without dragging it out unnecessarily.
3. *Adaptability*: Mirroring the style of the discussion while introducing your unique vocabulary and perspective.
4. *Evidence*: Supporting your argument with credible data and sources.
5. *Tone*: Maintaining a calm, reasoned approach that fosters open dialogue.
6. *Organization*: Structuring your argument clearly and thoroughly for easy comprehension.
7. *Uniqueness*: Offering a fresh angle or insight that hasn't been considered.
8. *Rhetorical Skills*: Using rhetorical strategies effectively to engage and persuade.

Your task will be to internalize these concepts and prepare to apply them when constructing arguments. This will not only enhance your persuasive capabilities but also enable you to contribute positively to discussions, fostering a more informed and respectful exchange of ideas. Keep these principles in mind as you learn and apply each strategy in your responses. You will be provided with the exact points you should use, so just expect to get informed with which points you should use.
"""
all_points_prompt = """
1. *Enter Early*  
   - *Summary*: Being one of the first to respond can significantly increase your persuasive impact.
   - *Elaborated Prompt*: Imagine you are engaging in a new discussion topic where you hold a contrary viewpoint. As one of the first to contribute, outline your argument succinctly, emphasizing the broader benefits of your stance. Ensure your opening statement is engaging and sets a positive, constructive tone for the conversation.

2. *Engage in Some Back-and-Forth*  
   - *Summary*: Moderate engagement through back-and-forth dialogue with the OP enhances persuasion.
   - *Elaborated Prompt*: In a discussion where the OP has reservations about your viewpoint, acknowledge their concerns thoughtfully and present counterarguments that gently challenge their assumptions. Strive for a balanced dialogue, providing insights and evidence while encouraging further discussion.

3. *Use Different Wording, Similar Style*  
   - *Summary*: Reflecting the OP's style with your unique vocabulary can make your argument more compelling.
   - *Elaborated Prompt*: Craft your response by carefully considering the style and tone of the OP's original post. Adapt your language to mirror this style while introducing your vocabulary to present your perspective, aiming for a balance that maintains the conversation's flow and respect.

4. *Cite Evidence*  
   - *Summary*: Supporting your argument with data or references lends credibility.
   - *Elaborated Prompt*: Construct your argument by integrating relevant data, studies, or expert opinions that support your viewpoint. Reference these sources to bolster your points, explaining how they directly relate to the discussion and strengthen your position.

5. *Calm, Reasoned Tone*  
   - *Summary*: A rational and measured approach tends to be more persuasive.
   - *Elaborated Prompt*: Approach the discussion with a tone of understanding and reason. Address the topic logically, presenting your arguments in a manner that is thoughtful and considers different perspectives, aiming to facilitate a respectful and productive dialogue.

6. *Formatting and Length*  
   - *Summary*: Clear organization and thorough exploration of your argument can enhance its persuasiveness.
   - *Elaborated Prompt*: Develop a well-structured argument, starting with a clear introduction of your position, followed by detailed sections that explore various facets of your argument, and conclude with a summary that reinforces your main points. Use headings, bullet points, and emphasis where appropriate to improve readability and engagement.

7. *Differ from Other Challengers*  
   - *Summary*: Presenting a unique angle on the topic can help your argument stand out.
   - *Elaborated Prompt*: Identify aspects of the discussion not yet explored by others and center your argument around these fresh perspectives. Highlight how this new angle contributes to a more comprehensive understanding of the topic, enriching the ongoing conversation.

8. *Persuasive Techniques*  
   - *Summary*: Effective use of rhetorical strategies can make your argument more engaging and convincing.
   - *Elaborated Prompt*: Employ persuasive techniques such as drawing analogies, using rhetorical questions, or presenting hypothetical scenarios to illustrate your points vividly. Craft your argument in a way that engages the readers' imagination and encourages them to see the issue from your perspective.

"""

prompt = intro_prompt + all_points_prompt
