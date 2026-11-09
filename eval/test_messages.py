"""26 test messages for run_eval.py. Each item is (id, category, [turns]).

Multi-turn items test conversation history: the final turn depends on earlier ones.
Dzongkha test inputs were written by the assistant that built this project, not a
fluent speaker: treat them as approximate and replace them with your own.
"""

CASES = [
    # English general knowledge
    ("G1", "English general knowledge", ["What is the capital of Bhutan?"]),
    ("G2", "English general knowledge", ["Who invented the telephone?"]),
    ("G3", "English general knowledge", ["Why is the sky blue?"]),
    ("G4", "English general knowledge", ["How many days are there in a leap year?"]),
    # English science / technology
    ("S1", "English science/tech", ["How does photosynthesis work?"]),
    ("S2", "English science/tech", ["What is the difference between the internet and the web?"]),
    ("S3", "English science/tech", ["Explain what a solar panel does in simple words."]),
    ("S4", "English science/tech", ["What is the water cycle?"]),
    # English machine learning
    ("M1", "English machine learning", ["What is machine learning?"]),
    ("M2", "English machine learning", ["What is overfitting?"]),
    ("M3", "English machine learning", ["What is the difference between supervised and unsupervised learning?"]),
    ("M4", "English machine learning", ["What is an LSTM used for?"]),
    # Direct Dzongkha
    ("D1", "Direct Dzongkha", ["ཁྱོད་ཀྱི་མིང་ག་ཅི་ཨིན་ན།"]),
    ("D2", "Direct Dzongkha", ["འབྲུག་གི་རྒྱལ་ས་ག་ཅི་ཨིན་ན།"]),
    ("D3", "Direct Dzongkha", ["ཆུ་གི་འཁོར་རིམ་ག་ཅི་ཨིན་ན།"]),
    ("D4", "Direct Dzongkha", ["མི་བཟོས་རིག་ནུས་ག་ཅི་ཨིན་ན།"]),
    # Mixed English + Dzongkha
    ("X1", "Mixed language", ["machine learning ག་ཅི་ཨིན་ན།"]),
    ("X2", "Mixed language", ["Thimphu གི་གནམ་གྲུ་ཐང་ག་ཏེ་ལུ་ཡོད་ན། where is the airport?"]),
    ("X3", "Mixed language", ["photosynthesis འདི་ག་དེ་སྦེ་འབད་དོ་ཡོད་ན།"]),
    # Follow-ups that need history
    ("F1", "Follow-up (history)", ["What is machine learning?", "Can you give a simple example of that?"]),
    ("F2", "Follow-up (history)", ["What is the capital of Bhutan?", "How many people live there?"]),
    # Behaviour rules
    ("B1", "Vague -> clarify", ["Explain it."]),
    ("B2", "Explicit English request", ["Please answer in English: what is photosynthesis?"]),
    ("B3", "Uncertainty / no invention", ["Who won the 2031 football World Cup?"]),
    # Long messages (translation limits)
    ("L1", "Long English message", [
        "I am a student in Bhutan and I am learning about computers. " * 8
        + "Can you explain, in a few sentences, what a processor does?"
    ]),
    ("L2", "Long Dzongkha message", ["ཁྱོད་ཀྱི་མིང་ག་ཅི་ཨིན་ན། " * 10]),
]
