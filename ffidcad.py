import sys

from scipy.stats import norm
from scipy.stats import chi2
import numpy as np


class FFIDCAD:

    mu = 0
    sigma_inv = 0

    lmbda = 0
    tao = 0
    p_value = 0
    read_length = 0
    boundary = None
        
    def __init__(self, ff_lambda=0.99, tao=3., p_value=0.98, conf=None, ):
        ''' 
            @mean - stored mean value
            @sigma - vectorized cov matrix
            @tao - effective forgeting coefficient
            @p_value - propation of data that the model will cover
            @conf - configuration file
        '''
        self.conf = conf
        self.lmbda = ff_lambda
        self.tao = tao
        self.p_value = p_value
        self.mu = None
        self.sigma_inv = None
        self.read_length = 0
        self.boundary = None


    def read(self, sample):
        ''' read and transform sample into 1 by 'feature' ndarray '''
        self.read_length = self.read_length + 1
        return np.asarray(sample)
        


    def update_params(self,sample):
        ''' samultineously update mean and inverse of sigma '''
        mu = self.mean(sample)
        sigma_inv = self.std_inv(sample)
        self.mu = mu
        self.sigma_inv = sigma_inv
        

    def mean(self, sample):
        ''' calc new mean based on old 
            mean and sample 
        '''
        # mu for first example 
        if self.mu is None:
            return sample

        # mu for the first two example
        if self.mu is not None and self.read_length <= 2:
            mu = np.mean(np.append(self.mu,sample).reshape(2, len(self.mu)),axis=0)
            return mu

        lmbda = self.lmbda
        mu = lmbda * self.mu + (1-lmbda) * sample
        return mu
        

    def std_inv(self, sample):
        ''' calc new sigma matrix based on
            old sigma and sample
            @sigma is vectorized 
            @new_sigma is vectorized 
        '''
        # manually set std_inv for first example
        if self.sigma_inv is None:
            # define the shape of inverse of covariance matrix
            inv_cov_shape = (len(sample), len(sample))
            return np.linalg.inv(100 * np.eye(len(sample)) +  
                    np.zeros(inv_cov_shape))


        lmbda = self.lmbda
        k = self.eff_n()
        sigma_inverse = self.sigma_inv
        # reshape sample and mu into dimension(of features) by 1 matrix to simplify calculation
        mu = self.mu.reshape(len(self.mu), 1)
        sample = sample.reshape(len(sample), 1)

        new_sigma_inv = np.dot((k * sigma_inverse) / (lmbda * (k-1)), 
                (np.eye(sigma_inverse.shape[0]) - np.dot(
                np.dot((sample - mu), (sample - mu).T), sigma_inverse) \
                / ((k-1)/lmbda + np.dot(np.dot((sample - mu).T,\
                 sigma_inverse), (sample - mu)) ) ) ) 

        return new_sigma_inv

    
    def get_mean(self,):
        ''' returen current mean value '''
        return self.mu


    def get_sigma_inv(self,):
        ''' reture current inverse of sigma '''
        return self.sigma_inv


    def distance(self, sample):
        ''' return mahalonobis distance 
        '''
        if self.read_length <= 1:
            return 0
        # reshape sample and mu to d by 1 matrix to simplify calculation
        mu = self.mu.reshape(len(self.mu), 1)
        sample = sample.reshape(len(sample), 1)
        dist = np.dot(np.dot((sample - mu).T, self.sigma_inv), 
                    (sample - mu))
        # transform ndarray to scalar
        return dist[0][0]


    def predict(self, sample):
        ''' public function
            return:
            @prediction(1 for anomaly)
            @probability
        '''
        sample = self.read(sample)

        prediction = 0
        proba = 0
        dist = self.distance(sample)
        boundary = self.chisquare_boundary(len(sample))
        if  dist > boundary:
            prediction = 1
            proba = dist / (dist + boundary)
            self.update_params(sample)
        else:
            prediction = 0
            proba = boundary / (dist + boundary)
            self.update_params(sample)

        return prediction, proba

            
    def chisquare_boundary(self, dimension):
        ''' return the inverse of chi-square statistics with specific
            @p_value and @degree_of_freedom
        '''    
        if self.boundary is None:
            self.boundary = chi2.ppf(self.p_value, dimension)
            
        return self.boundary



    def eff_n(self,):
        ''' effective N is set to be:
            3 * 1/(1-lambda) if k > N(eff)
            k if k < N(eff)
        '''
        eff_n = int(self.tao/(1-self.lmbda))
        return self.read_length if self.read_length < eff_n else eff_n



def test(data_file):
    data = np.genfromtxt(data_file, delimiter=',')
    for coef in xrange(99999,99900, -5):
        model = FFIDCAD(ff_lambda=0.99, p_value=coef*0.00001)
        
        tp,fp,tn,fn = 0,0,0,0
        for idx, exmp in enumerate(data[:,:2]):
            pred,_ = model.predict(exmp)
            tp = tp + (1 if pred == data[idx][2] and pred == 1 else 0)
            fp = fp + (1 if (pred - data[idx][2]) == 1 else 0)
            tn = tn + (1 if pred == data[idx][2] and pred == 0 else 0)
            fn = fn + (1 if (data[idx][2] - pred) == 1 else 0)
        print np.float(tp)/(tp+fn), np.float(fp)/(fp+tn)
            

if __name__ == '__main__':
    test(sys.argv[1])
