import sys

from ffidcad import FFIDCAD
import numpy as np


def synthetic_samples(means, sigmas, batch_size, noise_level=.05, noise_mgt=2):
    ''' generate multivariate time-series samples based on given mean and sigma
        @means - the list of means to generate "@means" size samples
        @sigmas - the list of corresponding covariance matrix
        @batch_size - number of examples per sample
    '''
    if len(means) != len(sigmas):
        raise RuntimeError("parameters means and sigmas are not aligned")

    sample_size = len(means)
    for idx in xrange(sample_size):
        normal_samples = np.random.multivariate_normal(means[idx], 
                            sigmas[idx], batch_size)
        #randomly(uniform) select anomaly anchors according to level
        noise_size = int(noise_level*normal_samples.shape[0])
        #numpy broadcasting(+), apply '+' to each row element of normal_samples
        noise = normal_samples[:noise_size,:] + \
                    np.sign(np.random.random() - 0.5) * \
                    noise_mgt * np.sqrt(np.diag(sigmas[idx]))
        yield normal_samples, noise


def noise_level_magnitude():
    ''' level and magnitude are split in low, medium, high stage
        @level: percentage of samples
        @mgt: coefficient of standard divation
    '''
    for level in [.05,.1,.2]:
        for mgt in [1,2,3]:
            yield level, mgt
    

def means(start, size=100, step=0.1):
    ''' generate time-series increasing means '''
    mus = [start]
    base = start
    for idx in xrange(size):
        base = base + step
        mus.append(base)
    return mus

def sigmas(start, size=100, step=0.01):
    ''' generate time-series increasing sigmas '''
    sgms = [start]
    dimension = start.shape[0]
    base = start
    for idx in xrange(size):
        base = np.eye(dimension) * step + base
        sgms.append(base)
    return sgms


def test():
    start_mu = np.array([5.,5.])
    start_sigma = np.array([[1.2, 0.1], [0.1, 2.]])
    
    mus = means(start_mu, 100, 0.1)
    sgms = sigmas(start_sigma, 100,0.01)

    result = []

    model = FFIDCAD(ff_lmbda=0.9999, pvalue=0.98)
    for sample, noise in synthetic_samples(mus, sgms, 25):
        # add noise
        sample = np.append(sample,noise).reshape(sample.shape[0]+noise.shape[0],
                        sample.shape[1])
        # uniform shuffle
        np.random.shuffle(sample)
        for example in sample:
            _pred_label, _prob = model.predict(example)
            _real_label = 0
            if example in noise:
                _real_label = 1
            result.append([example[0],example[1],_pred_label, _real_label ,round(_prob,4)])

        mu = model.get_mean()
        sigma_inv = model.get_sigma_inv()
        print '%s\t%s\t%s\t%s\t%s\t%s' % (mu[0],mu[1], 
            sigma_inv[0][0],sigma_inv[0][1],sigma_inv[1][0],sigma_inv[1][1])
    for item in result:
        print '%s\t%s\t%s\t%s\t%s' % (item[0],item[1],item[2],item[3],item[4])
    

def generate_data():
    start_mu = np.array([5.,5.])
    start_sigma = np.array([[1.2, 0.1], [0.1, 2.]])
    
    mus = means(start_mu, 100, 0.1)
    sgms = sigmas(start_sigma, 100,0.01)

    for sample, noise in synthetic_samples(mus, sgms, 25):
        # add noise
        sample = np.append(sample,noise).reshape(sample.shape[0]+noise.shape[0],
                        sample.shape[1])
        # uniform shuffle
        np.random.shuffle(sample)
        for example in sample:
            label = 0
            if example in noise:
                label = 1
            print '%s,%s,%s' % (example[0],example[1],label)


def test_real(f=sys.argv[1]):
    ''' input: ./lpg-hum|no2-voc '''
    result = []
    model = FFIDCAD(ff_lmbda=0.99)
    examples = []
    
    for line in open(f):
        examples.append([np.float(i) for i in line.strip().split('\t')])
    examples = np.asarray(examples)
    #examples = (examples - np.mean(examples, axis=0)) / np.std(examples)
    
    for example in examples:
        _rs, _prob = model.predict(example)
        result.append([example[0], example[1], _rs, round(_prob,4)])

        mu = model.get_mean()
        sigma_inv = model.get_sigma_inv()
        print '%s\t%s\t%s\t%s\t%s\t%s' % (mu[0],mu[1], 
            sigma_inv[0][0],sigma_inv[0][1],sigma_inv[1][0],sigma_inv[1][1])
    for item in result:
        print '%s\t%s\t%s\t%s' % (item[0],item[1],item[2],item[3])
        

if __name__ == '__main__':
    #main()
    #test()
    generate_data()
    #test_real()



