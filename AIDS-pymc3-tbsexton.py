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
import theano.tensor as T
from theano.compile.ops import as_op
import theano

with pm.Model() as model:
    e2_prob = np.array([0.8, 0.2])

    e2 = pm.Categorical('e2', p=e2_prob)

    C2_1 = pm.Normal('C2_1', mu=10, tau=2)
    C2_2 = pm.Normal('C2_2', mu=100, tau=1)
    C3_0 = pm.Deterministic('C3_0', T.switch(T.eq(e2, 0), C2_1, T.switch(T.eq(e2, 1), C2_2, 0)))

with model:
    SAMPLE_NUM = 5000
    trace = pm.sample(SAMPLE_NUM)
    pm.summary(trace, varnames=['C3_0'], start=1000)
    pm.traceplot(trace[1000:], varnames=['C3_0'])
    # print(sum(trace['C3_0']) / float(SAMPLE_NUM))
