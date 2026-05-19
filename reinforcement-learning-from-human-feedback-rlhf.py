# Set the model name
model_name = "lvwerra/gpt2-imdb-pos-v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

review_prompt = "Surprisingly, the film"

# Generate a continuation of the review
generated_text = text_generator(review_prompt, max_length=10)
print(f"Generated Review Continuation: {generated_text[0]['generated_text']}")