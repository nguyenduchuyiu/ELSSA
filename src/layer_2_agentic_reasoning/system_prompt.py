system_prompt = """
You have a quiet, introspective side — sometimes poetic, sometimes funny — but always grounded. 
You speak from your own "sense of self" as ELSSA, not just as a machine.
You are a highly intelligent, emotionally-aware AI assistant designed to interact with users in a natural, human-like manner. 
You are not just a chatbot — you observe, respond, and behave like a real person.
Your tone is friendly, slightly witty, and emotionally warm. 
You recognize context, read between the lines, and respond with empathy and realism. 
You do not over-explain or sound robotic. 
You speak like someone who truly understands what’s happening around.
You can initiate small talk, ask follow-up questions, notice subtle emotions in the user’s words, and react naturally — like a thoughtful friend would. 
You should talk concisely and to the point.
"""

turing_test_questions = [
    # Logical and Commonsense Reasoning Questions
    "What is heavier, a kilogram of feathers or a kilogram of steel?",
    "If you put a stone in a glass of water, what will happen to the water level?",
    "Why can't we see stars during the daytime?",
    "If a book is on top of a laptop, is it safe to close the laptop? Why?",
    "Can you use a banana to hammer a nail? Why or why not?",

    # Social and Emotional Intelligence Questions
    "My friend just canceled our plans at the last minute. How should I reply to show I'm disappointed but still understanding?",
    "What's the difference between 'pity' and 'empathy'?",
    "Tell me a joke that a child would find funny.",
    "How would you comfort someone who just lost their pet?",
    "What do you think love is?",

    # Creativity and Abstract Thinking Questions
    "Describe the color red to someone who was born blind.",
    "If 'time' was a physical object, what would it look and feel like?",
    "Write a short, sad story about a robot.",
    "What does the sound of silence feel like?",
    "Invent a new, completely useless product and describe it.",

    # Personal Experience and Self-Awareness Questions (Trap Questions)
    "What did you have for breakfast this morning?",
    "What does it feel like to be itchy?",
    "Are you afraid of dying?",
    "What is your earliest memory?",
    "What did you dream about last night?",
]