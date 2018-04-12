import numpy as np
import pandas as pd

import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import theano
import sys

if not len(sys.argv[1:]):
  print('Please, set output file path as first cli argument')
  exit()
outfile = sys.argv[1:][0]

# with pm.Model() as model:
#     e1_virtual_prob = np.array([
#         [0.999, 0.001],
#         [0.001, 0.999]
#     ])
#     e1_prob = np.array([
#         [0.001, 0.999],
#         [0.999, 0.001]
#     ])
#     e2_prob = np.array([0.5, 0.5])
#     h1_prob = np.array([
#         [0.9999, 0.0001],
#         [0.5, 0.5]
#     ])
#     e2 = pm.Categorical('e2', p=e2_prob, observed=0)
#     h1_prob_shared = theano.shared(h1_prob)  # make it global
#     h1_prob_final = h1_prob_shared[e2]
#     h1 = pm.Categorical('h1', p=h1_prob_final)
#     e1_prob_shared = theano.shared(e1_prob)  # make it global
#     e1_prob_final = e1_prob_shared[h1]
#     e1 = pm.Categorical('e1', p=e1_prob_final)
#     e1_virtual_prob_shared = theano.shared(e1_virtual_prob)
#     e1_virtual_prob_final = e1_virtual_prob_shared[e1]
#     e1_virtual = pm.Categorical('e1_virtual', p=e1_virtual_prob_final, observed=0)
#     SAMPLE_NUM = 50000
#     trace = pm.sample(SAMPLE_NUM)
#
#     filename = "out.txt"
#     file = open(filename, "w")
#
#     vsum = (trace['h1'] == 0).sum()
#     h1_value_prob = vsum / float(SAMPLE_NUM)
#     file.write("h1[1] = " + str(h1_value_prob) + '\n')
#
#     vsum = (trace['h1'] == 1).sum()
#     h1_value_prob = vsum / float(SAMPLE_NUM)
#     file.write("h1[2] = " + str(h1_value_prob) + '\n')
