'''
This work is licensed under a Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/).

This code implements the first version of SiMPle [1] for music similarity purposes. Specifically, this function returns the result of the AB-join operation.
A faster version [2] of this code is also available in the same git repository.
Author: Diego Furtado Silva

References:
[1] Silva, D. F., Yeh, C. C. M., Batista, G. E. D. A. P. A., & Keogh, E. (2016). SIMPle: Assessing music similarity using subsequences joins. In XVII International Society for Music Information Retrieval Conference.
[2] Silva, D. F., Yeh, C. C. M., Zhu, Y., Batista, G. E., & Keogh, E. (2019). Fast Similarity Matrix Profile for Music Analysis and Exploration. IEEE Transactions on Multimedia, 21(1), 29-38.
'''

import numpy as np
import random
import scipy
_EPS = 1e-14

def simpleab(seq_a, seq_b, subseq_len):
    
    # prerequisites
    ndim = seq_b.shape[0]
    seq_a_len = seq_a.shape[1]
    seq_b_len = seq_b.shape[1]
    
    matrix_profile_len = seq_a_len - subseq_len + 1;
    dist_profile_len = seq_b_len - subseq_len + 1;
    
    # initialization
    matrix_profile = np.full(matrix_profile_len, np.inf)
    mp_index = -np.ones((matrix_profile_len), dtype=int)
    
    # windowed cumulative sum of the sequence b
    seq_b_cum_sum2 = np.insert(np.sum(np.cumsum(np.square(seq_b),1),0), 0, 0)
    seq_b_cum_sum2 = seq_b_cum_sum2[subseq_len:]-seq_b_cum_sum2[0:seq_b_len - subseq_len + 1]
    
    for i_subseq in range(0,matrix_profile_len):
        
        this_subseq = np.flip(seq_a[:,i_subseq:i_subseq+subseq_len],1)
        subseq_cum_sum2 = np.sum(np.square(this_subseq))
                
        dist_profile = seq_b_cum_sum2 + subseq_cum_sum2
        for i_dim in range(0,ndim):
            prods = scipy.signal.fftconvolve(this_subseq[i_dim,:],seq_b[i_dim,:])
            dist_profile -= (2 * prods[subseq_len-1:seq_b_len])
        
        matrix_profile[i_subseq] = np.min(dist_profile)
        mp_index[i_subseq] = np.argmin(dist_profile)
        
    return matrix_profile, mp_index

