from .default import AnyCombiner

combiners = [
    AnyCombiner,  # Remains for simplicity as default combiner
]


def get_all_combiner():
    return {combiner._name: combiner for combiner in combiners}
