import pymc3 as pm
import numpy as np
from theano.ifelse import ifelse
import theano
import theano.tensor as tt
from pymc3 import Discrete
from pymc3.distributions.dist_math import bound, binomln, logpow


class I_am_HIV_positive(Discrete):
    def __init__(self, f_1_of_10000, *args, **kwargs):
        super(I_am_HIV_positive, self).__init__(*args, **kwargs)
        self.f_1_of_10000 = tt.as_tensor_variable(f_1_of_10000)

    def logp(self, value):
        f_1_of_10000 = self.f_1_of_10000
        p = ifelse(
            tt.eq(f_1_of_10000, 1),
            ifelse(tt.eq(value, 1), 0.0001, 0.9999),
            tt.cast(0.5000, 'float64')
        )
        p = tt.printing.Print('p')(p)
        return bound(tt.log(p),
            value >= 0, value <= 1,
            p >= 0, p <= 1)

class HIV_test_false_positive(Discrete):
    def __init__(self, test_show_positive, p_I_am_HIV_positive, *args, **kwargs):
        super(HIV_test_false_positive, self).__init__(*args, **kwargs)
        self.test_show_positive = tt.as_tensor_variable(test_show_positive)
        self.p_I_am_HIV_positive = tt.as_tensor_variable(p_I_am_HIV_positive)

    def logp(self, value):
        test_show_positive = self.test_show_positive
        p_I_am_HIV_positive = self.p_I_am_HIV_positive

        p = ifelse(
            tt.eq(test_show_positive, 1),
            ifelse(
                tt.eq(p_I_am_HIV_positive, 0),
                ifelse(
                    tt.eq(value, 0),
                    0.999,
                    0.111
                ),
                ifelse(
                    tt.eq(value, 0),
                    0.001,
                    0.999
                )
            ),
            tt.cast(0.5000, 'float64')
        )

        return bound(tt.log(p),
            value >= 0, value <= 1,
            p >= 0, p <= 1)


with pm.Model() as model:
    f_1_of_10000 = pm.Categorical("f_1_of_10000", np.array([0, 1]))
    #f_1_of_10000 = tt.printing.Print('f_1_of_10000')(f_1_of_10000)
    p_I_am_HIV_positive = I_am_HIV_positive('p_I_am_HIV_positive', f_1_of_10000, testval=1)

    # test_show_positive = pm.Categorical("test_show_positive", [0, 1])
    #
    # p_HIV_test_false_positive = HIV_test_false_positive(
    #     'p_HIV_test_false_positive',
    #     test_show_positive,
    #     p_I_am_HIV_positive,
    #     testval=1
    # )

 #   step = pm.Metropolis()
    SAMPLE_NUM = 1000
#    trace = pm.sample(SAMPLE_NUM, tune=1000, step=step)
    trace = pm.sample(SAMPLE_NUM)
    samples = trace['p_I_am_HIV_positive']
    print(sum(samples))
    print(sum(samples) / float(SAMPLE_NUM))
