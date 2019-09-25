


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import fit2D
import pickle



# A logphi* at z=8
# B L* analogue
# C slope of redshift evolution

p = {'A': 1.5, 'B': 0.0, 'C': -0.5}

truth = fit2D.model(p)

plt.imshow(np.log10(truth), origin = 'lower')
# plt.show()
plt.savefig('truth.pdf')

print(f'total number of objects in truth: {np.sum(truth)}')

# --- poisson sample the model to make "observation"

obs = np.random.poisson(truth)

print(f'total number of objects in observations: {np.sum(obs)}')

plt.imshow(np.log10(obs), origin = 'lower')
# plt.show()
plt.savefig('obs.pdf')


# -- initialise fitter

source = fit2D.fitter(obs)

source.priors['A'] = scipy.stats.uniform(loc = 0., scale = 5.) # assume uniform prior
source.priors['B'] = scipy.stats.uniform(loc = -2., scale = 4.) # assume uniform prior
source.priors['C'] = scipy.stats.uniform(loc = -2, scale = 3.) # assume uniform prior


samples = source.fit(nsamples = 1000)

pickle.dump(samples, open('samples.p','wb'))

analysis = fit2D.analyse(samples, parameters = ['A','B','C'], truth = p)
analysis.P() # print central 68 and median with Truth if provided
analysis.corner(filename = 'corner.pdf') # produce corner plot
