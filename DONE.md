1. Concurrency in CoquiTTS causes race condition (inner cache) leads to mismatch tensor
    -> Solution: Change to sequential processing cause CoquiTTS synthesizes fast enough 
                to feed the later chunk right after the former chunk is played.
                