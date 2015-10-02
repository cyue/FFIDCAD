import sys

from ffidcad import FFIDCAD
import numpy as np


def run(data_file, ff_lambda=0.99, p_value=0.98):
    # initiate model
    model = FFIDCAD(ff_lmbda=ff_lambda, pvalue=p_value)
    for line in open(data_file):
        example = [np.float(item) for item in line.strip().split(',')]
        example = np.asarray(example)
        # example[0] is x1, example[1] is x2, example[2] is label
        pred_label, _ = model.predict(example[:2])
        real_label = example[2]
        # scheme is: "x1, x2, real_label, predicted_label"
        yield (example[0], example[1], real_label, pred_label)

    
if __name__ == '__main__':
    '''
        input:
            @sys.argv[1]: csv data file, following the scheme: "x1, x2, label"
            @sys.argv[2]: forgetting factor lambda that belongs to (0,1]
            @sys.argv[3]: pvalue in inverse of chi-square
        output:
            a csv file with scheme: "x1, x2, real_label, predicted_label" like:

            "5.79119090579,6.1212826649,0,0"
        instruction for running the script in command line:
            python main.py data 0.99 0.98
    output = open('output.csv', 'w')
    for result in run(sys.argv[1], np.float(sys.argv[2]), np.float(sys.argv[3])):
        output.write('%s,%s,%s,%s\n' % (result[0], result[1], 
                        np.int(result[2]), np.int(result[3])))
    '''
    for i in xrange(9999,9900,-1):
        #lmbda = 1 - (0.0001*(2**i))
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for item in run(sys.argv[1], ff_lambda=0.99, p_value=i*0.0001):
            tp = tp + (1 if np.int(item[2])==1 and item[2]==item[3]  else 0)
            fp = fp + (1 if (item[3]-item[2])==1 else 0)
            tn = tn + (1 if item[2]==0 and item[2]==item[3] else 0)
            fn = fn + (1 if (item[2]-item[3])==1 else 0)
            
        print np.float(tp)/(tp+fn), np.float(fp)/(fp+tn), i*0.0001

