"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

PyTest related to evaluate.py

Team AI-DJ : 
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""
from evaluate import metricsEval

def test_metricsEval():
    """
    This function tests the metricsEval function on a small example by 
    checking if the metrics returned by metricsEval are correct.
    """
    
    yValFold = [1, 0, 1, 1, 0]
    yPred = [1, 0, 0, 1, 1]
    
    # True Positive
    TP = 2
    
    # False Negative
    FN = 1
    
    # Positive
    P = TP + FN
    
    # True Negative
    TN = 1
    
    # False Positive
    FP = 1
    
    # Negative
    N = TN + FP
    
    # Number of good predictions over the total number
    accuracy = (TP+TN)/(P+N)
    
    # Proportion of positives that are detected
    recall = TP / (TP + FN)
    
    # Porportion of good predictions among all the positive predictions
    precision = TP / (TP + FP)
    
    f1 = 2*(precision*recall)/(precision+recall)
    
    accuracyFct, recallFct, f1Fct, precisionFct = metricsEval(yValFold, yPred)
    
    assert accuracyFct == accuracy
    assert recall == recallFct
    assert precision == precisionFct
    assert f1 == f1Fct