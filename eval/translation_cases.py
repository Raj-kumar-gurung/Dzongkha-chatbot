"""Translation-only test set (no chat model involved).

EN_SENTENCES are translated English -> Dzongkha -> back to English (round trip).
DZ_SENTENCES are translated Dzongkha -> English directly.

To get a real quality score, ask a fluent Dzongkha speaker for reference
translations and put them in REFERENCE_DZ (key = the English sentence). The
scorer then reports a character n-gram F-score against them. The Dzongkha
sentences below were written by an AI, not a fluent speaker: replace them.
"""

EN_SENTENCES = [
    "Hello, how are you?",
    "Thank you very much.",
    "What is your name?",
    "Bhutan is a small country in the Himalayas.",
    "The capital of Bhutan is Thimphu.",
    "Water boils at 100 degrees Celsius at sea level.",
    "I do not know the answer to that question.",
    "Please ask your question again in a clearer way.",
    "Machine learning is a way for computers to learn from examples.",
    "A solar panel turns sunlight into electricity.",
    "My phone number is 17123456 and my name is Pema.",
    "Plants use sunlight, water and air to make their own food.",
    "The meeting starts at 9 o'clock on Monday morning.",
    "Artificial intelligence can make mistakes, so please check important facts.",
    "Where is the nearest hospital?",
]

DZ_SENTENCES = [
    "ཁྱོད་ཀྱི་མིང་ག་ཅི་ཨིན་ན།",
    "འབྲུག་གི་རྒྱལ་ས་ག་ཅི་ཨིན་ན།",
    "ཆུ་གི་འཁོར་རིམ་ག་ཅི་ཨིན་ན།",
    "མི་བཟོས་རིག་ནུས་ག་ཅི་ཨིན་ན།",
]

# {"English sentence": "reference Dzongkha from a fluent speaker"}
REFERENCE_DZ = {}
