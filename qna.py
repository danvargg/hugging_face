"""Question answering (Qu-An)."""
import transformers
from transformers import pipeline

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