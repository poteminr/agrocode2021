import re
from span_words import span_words


def predict_spans_baseline(string, span_length_threshold=3):
    # Just finds words from span_words in the given string
    spans = []
    for word in span_words:
        if len(word) > span_length_threshold:
            for match in re.finditer(word, string):
                spans.append([match.span()[0], match.span()[1]])
            
    return spans