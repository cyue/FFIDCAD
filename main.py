import sys
import time

from ffidcad import FFIDCAD
from generate_synthetic import create_params, generate
import numpy as np


def test(data_file):
    ''' effectiveness test wrt ROC '''
    data = np.genfromtxt(data_file, delimiter=',')
    for coef in xrange(99999,99900, -1):
        model = FFIDCAD(ff_lambda=0.99, p_value=coef*0.00001)
        
        tp,fp,tn,fn = 0,0,0,0
        for idx, exmp in enumerate(data[:,:2]):
            pred,_ = model.predict(exmp)
            tp = tp + (1 if pred == data[idx][2] and pred == 1 else 0)
            fp = fp + (1 if (pred - data[idx][2]) == 1 else 0)
            tn = tn + (1 if pred == data[idx][2] and pred == 0 else 0)
            fn = fn + (1 if (data[idx][2] - pred) == 1 else 0)
        print np.float(tp)/(tp+fn), np.float(fp)/(fp+tn)
 

def eff_test():
    ''' efficiency test wrt predition time cost '''
    size = 1000
    mu = np.array([5,5])
    cov_matrix = np.array([[1,0.1],[0.1,1]])
    for coef in xrange(1,11):
        mus, cov_matrices = create_params(mu,
                            cov_matrix,
                            0.1,
                            [[0.,0.],[0.,0.]],
                            coef*size)
        samples = generate(mus, cov_matrices, 1)
        model = FFIDCAD(ff_lambda=0.99, p_value = 0.9999)
        start = time.time()
        for sample in samples:
            model.predict(sample)
        cost = time.time() - start
        print coef*size, cost, ';'

       
if __name__ == '__main__':
    eff_test() 
