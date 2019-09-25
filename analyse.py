



import pickle
import fit2D


truth = {'A': 2, 'B': 0.0, 'C': -0.5}

samples = pickle.load(open('samples.p','rb'))

analysis = fit2D.analyse(samples, parameters = ['A','B','C'], truth = truth)
analysis.P() # print central 68 and median with Truth if provided
analysis.corner(filename = 'corner.pdf') # produce corner plot
