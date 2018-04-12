#
#   Example for graph e2 --> h1 --> e1
#    $graph = {
#      nodes:{'e1':['1','2'], 'e2':['1','2'], 'h1':['1','2']}, // every node contains array of its alternatives
#      edges:[['h1','e1'],['e2','h1']]
#    };
#    $probabilities = {
#     e1: {
#       soft:{1:1, 2:0}, // soft evidence for e1 and ^e1
#       '{"h1":"1"}':{1:0.999, 2:0.001}, // sum must be equal to 1
#       '{"h1":"2"}':{1:0.001, 2:0.999}  // sum must be equal to 1
#     },
#     e2: {
#       soft:{1:1, 2:0} // soft evidence for e2 and ^e2
#     },
#     h1: {
#       // prior probability of proposition alternative is always 1/<number of alternatives>
#       '{"e2":"1"}':{1:0.0001, 2:0.9999}, // sum must be equal to 1
#       '{"e2":"2"}':{1:0.5, 2:0.5}  // sum must be equal to 1
#      }
#    }
#
# Approach:
#   https: // github.com / pymc - devs / pymc3 / issues / 1790
#   https: // gist.github.com / tbsexton / 1349864212b25cce91dbe5e336d794b4
#
import numpy as np
import pandas as pd

import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import theano

with pm.Model() as model:
    e2_prob = np.array([0.5, 0.5])
    e2_virt_prob = np.array([
        [0.999999, 0.000001],
        [0.000001, 0.999999]
    ])
    h1_prob = np.array([
        [0.0001, 0.9999],
        [0.5, 0.5],
    ])
    e1_prob = np.array([
        [0.999, 0.001],
        [0.001, 0.999],
    ])
    e1_virt_prob = np.array([
        [0.999999, 0.000001],
        [0.000001, 0.999999],
    ])

    e2 = pm.Categorical('e2', p=e2_prob, observed=0)
    # e2_virt_prob_shared = theano.shared(e2_virt_prob)
    # e2_virt_prob_final = e2_virt_prob_shared[e2]
    # e2_virt = pm.Categorical('e2_virt', p=e2_virt_prob_final, observed=0)

    h1_prob_shared = theano.shared(h1_prob)  # make it global
    h1_prob_final = h1_prob_shared[e2]
    h1 = pm.Categorical('h1', p=h1_prob_final)

    e1_prob_shared = theano.shared(e1_prob)
    e1_prob_final = e1_prob_shared[h1]
    e1 = pm.Categorical('e1', p=e1_prob_final)

    e1_virt_prob_shared = theano.shared(e1_virt_prob)  # make it global
    e1_virt_prob_final = e1_virt_prob_shared[e1]  # select the prob array that "happened" thanks to parents
    e1_virt = pm.Categorical('e1_virt', p=e1_virt_prob_final, observed=0)

with model:
    SAMPLE_NUM = 100000
    trace = pm.sample(SAMPLE_NUM)
    print(sum(trace['h1']) / float(SAMPLE_NUM))
    print(sum(trace['e1']) / float(SAMPLE_NUM))

