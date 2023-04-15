"""Natural language generation (NLG)."""
import transformers
from transformers import pipeline, Conversation, AutoModelForSeq2SeqLM, AutoTokenizer

transformers.logging.set_verbosity_error()

# Generation
text_generator = pipeline('text-generation', model='gpt2')
transformers.set_seed(1)

input_text = 'Natural Language Processing is a growing domain in machine learning'

synthetic_text = text_generator(input_text, num_return_sequences=3, max_new_tokens=50)

for text in synthetic_text:
    print(text.get('generated_text'), '\n-----------------')

# Chatbot
conversational_pipeline = pipeline("conversational", model="facebook/blenderbot_small-90M")

# print(conversational_pipeline.model.config)

first_input = "Do you have any hobbies?"
second_input = "I like to watch movies"
third_input = "action movies"

# Create a context
bot_conversation = Conversation(first_input)

print("\nFirst Exchange: \n--------------------")

conversational_pipeline(bot_conversation)
print(" User Input:", bot_conversation.past_user_inputs[0])
print(" Bot Output:", bot_conversation.generated_responses[0])

print("\nSecond Exchange: \n--------------------")
bot_conversation.add_user_input(second_input)
conversational_pipeline(bot_conversation)

print(" User Input:", bot_conversation.past_user_inputs[1])
print(" Bot Output:", bot_conversation.generated_responses[1])

print("\nThird Exchange: \n--------------------")
bot_conversation.add_user_input(third_input)
conversational_pipeline(bot_conversation)

print(" User Input:", bot_conversation.past_user_inputs[2])
print(" Bot Output:", bot_conversation.generated_responses[1])

print("\nAccessing All Responses: ")
print(bot_conversation)

# Translation
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

source_english = "Acme is a technology company based in New York and Paris"

inputs_german = tokenizer("translate English to German: " + source_english, return_tensors="pt")
outputs_german = model.generate(inputs_german["input_ids"], max_length=40)

print("German Translation: ", tokenizer.decode(outputs_german[0], skip_special_tokens=True))

inputs_french = tokenizer("translate English to French: " + source_english, return_tensors="pt")
outputs_french = model.generate(inputs_french["input_ids"], max_length=40)

print("French Translation: ", tokenizer.decode(outputs_french[0], skip_special_tokens=True))
