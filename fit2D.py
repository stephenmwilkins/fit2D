
import numpy as np
import scipy.stats
import sys
import os
import emcee
import scipy.misc

import corner


def model(p, x = np.arange(8, 15, 1.), y = np.arange(-2, 2, 1.)):
    A,B,C = [p[k] for k in ['A','B','C']]
    xx, yy = np.meshgrid(x, y)
    return (10**A) * np.exp(-10**(yy-B))*10**(C*(xx-8.0))







class fitter():


    def __init__(self, obs):

        self.obs = obs
        self.parameters = ['A','B','C']
        self.priors = {}

    def lnlike(self, m):
        """log Likelihood function"""

        # v = -0.5*np.sum((self.obs - m)**2) # almost certainly wrong

        # this seems to do better than the above, though I'm still not convinced
        v = np.sum(self.obs*np.log(m)) - np.sum(m) - np.sum(np.log(scipy.misc.factorial(self.obs))) # https://stats.stackexchange.com/questions/316763/log-likelihood-function-in-poisson-regression

        if not np.isfinite(v):
            return -np.inf

        return v


    def lnprob(self, params):
        """Log probability function"""

        p = {parameter:params[i] for i,parameter in enumerate(self.parameters)}

        lp = np.sum([self.priors[parameter].logpdf(p[parameter]) for parameter in self.parameters])

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnlike(model(p))


    def fit(self, nwalkers = 50, nsamples = 1000, burn = 200):

        self.ndim = len(self.parameters)
        self.nwalkers = nwalkers
        self.nsamples = nsamples

        p0 = [ [self.priors[parameter].rvs() for parameter in self.parameters] for i in range(nwalkers)]

        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob, args=())
        pos, prob, state = self.sampler.run_mcmc(p0, burn)
        self.sampler.reset()
        self.sampler.run_mcmc(pos, nsamples)

        chains = self.sampler.chain[:, :, ].reshape((-1, self.ndim))
        samples = {p: chains[:,ip] for ip, p in enumerate(self.parameters)}

        return samples


class analyse():

    def __init__(self, samples, parameters = False, truth = False):

        self.samples = samples
        self.truth = truth

        if not parameters:
            self.parameters = self.samples.keys()
        else:
            self.parameters = parameters

    def P(self):

        for k,v in self.samples.items():

            if not self.truth:
                print(f'{k}: {np.percentile(v, 16.):.2f} {np.median(v):.2f} {np.percentile(v, 84):.2f}')
            else:
                print(f'{k}: {np.percentile(v, 16.):.2f} {np.median(v):.2f} {np.percentile(v, 84):.2f} | {self.truth[k]}')


    def corner(self, filename = 'corner.pdf'):

        Samples = np.array([self.samples[k] for k in self.parameters]).T

        range = []
        for k in self.parameters:
            med = np.median(self.samples[k])
            c68 = np.percentile(self.samples[k], 84) - np.percentile(self.samples[k], 16)
            range.append([med - c68*4, med + c68*4])

        if not self.truth:
            figure = corner.corner(Samples, labels = self.parameters)
        else:
            figure = corner.corner(Samples, labels = self.parameters, truths = [self.truth[k] for k in self.parameters], range = range)



        figure.savefig(filename)
