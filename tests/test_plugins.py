import os
def say(text):
    os.system(f'espeak-ng "{text}"')
say("Hello, how are you?")