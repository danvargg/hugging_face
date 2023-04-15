"""Text summarization."""
import transformers
from transformers import pipeline
import evaluate

transformers.logging.set_verbosity_error()

verbose_text = """
Earth is the third planet from the Sun and the only astronomical object known to harbor life. While large volumes of 
water can be found  throughout the Solar System, only Earth sustains liquid surface water. About 71% of Earth's surface 
is made up of the ocean, dwarfing Earth's polar ice, lakes, and rivers. The remaining 29% of Earth's surface is land, 
consisting of continents and islands. Earth's surface layer is formed of several slowly moving tectonic plates, 
interacting to produce mountain ranges, volcanoes, and earthquakes. Earth's liquid outer core generates the magnetic 
field that shapes Earth's magnetosphere, deflecting destructive solar winds.
"""

verbose_text = verbose_text.replace('\n', '')

extractive_summarizer = pipeline("summarization", min_length=10, max_length=100)

# Extractive summarization
extractive_summary = extractive_summarizer(verbose_text)

print(extractive_summary[0].get("summary_text"))

# Evaluate
rouge_evaluator = evaluate.load("rouge")

# Evaluate exact match strings
reference_text = ["This is the same string"]
predict_text = ["This is the same string"]

eval_results = rouge_evaluator.compute(predictions=predict_text,
                                       references=reference_text)
print("Results for Exact match", eval_results)

# Evaluate no-match strings
reference_text = ["This is the different string"]
predict_text = ["Google can predict warm weather"]

eval_results = rouge_evaluator.compute(predictions=predict_text,
                                       references=reference_text)
print("\nResults for no match", eval_results)

# Evaluate summary
eval_results = rouge_evaluator.compute(
    predictions=[extractive_summary[0].get("summary_text")], references=[verbose_text]
)

print("\nResults for Summary generated", eval_results)
