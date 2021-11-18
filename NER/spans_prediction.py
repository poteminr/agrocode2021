import re
from span_words import span_words
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class NER:
    def __init__(self, path_to_model, path_to_tokenizer, device=0):
        self.path_to_model = path_to_model
        self.path_to_tokenizer = path_to_tokenizer
        self.device = device

        self.model = AutoModelForTokenClassification.from_pretrained(self.path_to_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_tokenizer)

        self.hugginface_pipeline = pipeline(model=self.model, tokenizer=self.tokenizer, task='ner', 
                                            aggregation_strategy='average', device=self.device)


    def predict_spans_baseline(self, text, span_length_threshold=2):
        # Just finds words from span_words in the given string
        spans = []
        for word in span_words:
            if len(word) > span_length_threshold:
                for match in re.finditer(word, text.lower()):
                    spans.append([match.span()[0], match.span()[1]])
                
        return spans


    def predict_spans(self, text, baseline_for_empty=True, text_to_lowercase=True):
        spans = []

        # if text_to_lowercase:
        #     text = text.lower()

        prediction = self.hugginface_pipeline(text)

        if len(prediction) == 0 and (text_to_lowercase):
            text = text.lower()
            prediction = self.hugginface_pipeline(text)


        if (len(prediction) == 0) and (baseline_for_empty):
            spans = self.predict_spans_baseline(text)

        else:
            for entity in prediction:
                spans.append([entity['start'], entity['end']])
        
        return spans, prediction
