def metric(true_passage, predicted_passage, alpha=0.9):
    
    n, m = len(true_passage), len(predicted_passage)
    assert n == m, "true_passage and predicted_passage don't have the same shape ({}, {})".format(n, m)

    score = 0
    for i in range(n):
        true = true_passage[i]
        pred = predicted_passage[i]

        index = pred.index(true)

        score += alpha**index
    return score/n
