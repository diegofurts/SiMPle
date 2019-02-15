import numpy as np
from sklearn.metrics import average_precision_score

def oti(seq_a, seq_b):
    
    profile_a = np.sum(seq_a,1);
    profile_b = np.sum(seq_b,1);

    oti_vec = np.zeros(12)
    for i in range(12):
        oti_vec[i] = np.dot(profile_a,np.roll(profile_b,i,axis=0))

    sorted_index = np.argsort(oti_vec)
    
    return np.roll(seq_b, sorted_index[-1], axis=0), sorted_index



def mean_average_precision(dist_matrix, train_labels, test_labels):
    
    ave_sum = 0
    n = test_labels.shape[0]
    
    for i in range(0, n):
        
        y_true = np.zeros(train_labels.shape[0], dtype='int')
        y_true[train_labels == test_labels[i]] = 1
        
        y_scores = -dist_matrix[i,:] # descending order: the higher the distance, the lower the "score"
        
        ave_sum += average_precision_score(y_true, y_scores)        
    
    return ave_sum / n


def mr1(dist_matrix, train_labels, test_labels):

    n = test_labels.shape[0]
    sum_rank = 0
    
    for i in range(0, n):
        sorted_idx = np.argsort(dist_matrix[i,:])
        rank = np.where(train_labels[sorted_idx] == test_labels[i])
        sum_rank += rank[0][0]
    
    return sum_rank / n

