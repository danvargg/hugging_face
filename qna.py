"""Question answering (Qu-An)."""
import transformers
from transformers import pipeline
from evaluate import load

# Set to avoid warning messages
transformers.logging.set_verbosity_error()

context = """
Earth is the third planet from the Sun and the only astronomical object known to harbor life. While large volumes of
water can be found throughout the Solar System, only Earth sustains liquid surface water. About 71% of Earth's surface 
is made up of the ocean, dwarfing Earth's polar ice, lakes, and rivers. The remaining 29% of Earth's surface is land, 
consisting of continents and islands. Earth's surface layer is formed of several slowly moving tectonic plates, 
interacting to produce mountain ranges, volcanoes, and earthquakes. Earth's liquid outer core generates the magnetic 
field that shapes Earth's magnetosphere, deflecting destructive solar winds.
"""

quan_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

print(quan_pipeline(question="How much of earth is land?", context=context))

print("\nAnother question :")
print(quan_pipeline(question="How are mountain ranges created?", context=context))

squad_metric = load("squad_v2")

# Ignoring Context & Question as they are not needed for evaluation
# This example is to showcase how the evaluation works based on match between the prediction and the correct answer

correct_answer = "Paris"

predicted_answers = ["Paris", "London", "Paris is one of the best cities in the world"]

cum_predictions = []
cum_references = []

for i in range(len(predicted_answers)):
    # Use the input format for predictions
    predictions = [{'prediction_text': predicted_answers[i], 'id': str(i), 'no_answer_probability': 0.}]
    cum_predictions.append(predictions[0])

    # Use the input format for answers
    references = [{'answers': {'answer_start': [1], 'text': [correct_answer]}, 'id': str(i)}]
    cum_references.append(references[0])

    results = squad_metric.compute(predictions=predictions, references=references)
    print("F1 is", results.get('f1'), " for answer :", predicted_answers[i])

# Compute for cumulative Results
cum_results = squad_metric.compute(predictions=cum_predictions, references=cum_references)
print("\n Cumulative Results : \n", cum_results)
