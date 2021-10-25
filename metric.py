import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import log_loss


def total_logloss(ground_truth_labels, my_labels):
    my_labels = np.array([v['label'] for k, v in my_labels.items()])
    ground_truth_labels = np.array([v['label'] for k, v in ground_truth_labels.items()])
    total_logloss = np.mean([log_loss(ground_truth_labels[:, c], my_labels[:, c]) for c in range(10)])
    return total_logloss

def f1_span_score_per_text(ground_truth_labels, my_labels, text):
    
    span_my_tokens = list()
    span_gt_tokens = list()
    
    if len(my_labels):
        span_my_words = [text[s[0]:s[1]] for s in my_labels]
        span_my_tokens = [re.sub('[^А-Яа-яёЁ ]+', '', item) for sublist in span_my_words for item in sublist.split()]

    if len(ground_truth_labels):
        span_gt_words = [text[s[0]:s[1]] for s in ground_truth_labels]
        span_gt_tokens = [re.sub('[^А-Яа-яёЁ ]+', '', item) for sublist in span_gt_words for item in sublist.split()]

    if len(span_my_tokens) == 0 or len(span_gt_tokens) == 0:
        return int(span_my_tokens == span_gt_tokens)
    
    tp = np.sum(list((Counter(span_gt_tokens) & Counter(span_my_tokens)).values()))

    precision = tp/len(span_my_tokens)
    recall = tp/len(span_gt_tokens)

    if precision + recall > 0:
        return 2*precision*recall/(precision+recall)
    else:
        return 0
    
def total_f1_score(ground_truth_labels, my_labels, data):
    f1_scores = list()
    for k in ground_truth_labels.keys():
        text = data.loc[data['text_id'] == int(k), 'text'].values[0]
        f1_score_per_text = f1_span_score_per_text(ground_truth_labels[k]['span'], my_labels[k]['span'], text)
        f1_scores.append(f1_score_per_text)
        
    return np.mean(f1_scores)

def total_metric(metric1, metric2):
    return 0.8 * (1 - metric1) + 0.2 * metric2


def score(sample_submission, test_labels, data):
    
    task1_metrics = total_logloss(test_labels, sample_submission)
    task2_metrics = total_f1_score(test_labels, sample_submission, data)
    total_score = total_metric(task1_metrics, task2_metrics)

    return total_score

if __name__ == '__main__':
    test_labels = json.load(open('test_labels.json', 'r'))
    data = pd.read_csv('test.csv')
    sample_submission = json.load(open('sample_submission.json', 'r'))
    score(sample_submission, test_labels, data)


