from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceLLMEngine:
    def __init__(self, model_name: str):
        """
        Initializes the model and tokenizer based on the specified model name.
        :param model_name: The name of the model to load (e.g., "gpt2").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate_text(self, prompt: str, **generation_kwargs):
        """
        Generates text based on a prompt.
        :param prompt: The input text prompt.
        :param generation_kwargs: Additional keyword arguments for the generate() method.
        :return: The generated text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, **generation_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage:
    model_name = "mistralai/Mistral-7B-v0.1"  # You can replace this with any model name from Hugging Face.
    engine = HuggingFaceLLMEngine(model_name=model_name)

    prompt = "Here is an example prompt for the language model:"
    generated_text = engine.generate_text(prompt, max_length=50)

    print(generated_text)
