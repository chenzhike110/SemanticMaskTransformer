REPEATS = ['arms', 'knees', 'thighs', 'elbows', 'calves', 'hands', 'and', 'feet', 'legs']

DETERMINERS = ["the", "their", "his", "her"]
DETERMINERS_PROP = [0.5, 0.3, 0.1, 0.1]

TEXT_TRANSITIONS = [' while ', '. ', ' and ', ' with ', ', ']
TEXT_TRANSITIONS_PROP = [0.2, 0.1, 0.2, 0.2, 0.2]

def is_repeats(word):
    for key in REPEATS:
        if key in word:
            return True
    return False