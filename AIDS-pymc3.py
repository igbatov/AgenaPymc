import pymc3 as pm
import numpy as np
from theano.ifelse import ifelse
import theano
import theano.tensor as tt
from pymc3 import Discrete
from pymc3.distributions.dist_math import bound, binomln, logpow


# class I_am_HIV_positive(Discrete):
#     def __init__(self, f_1_of_10000, *args, **kwargs):
#         super(I_am_HIV_positive, self).__init__(*args, **kwargs)
#         self.f_1_of_10000 = tt.as_tensor_variable(f_1_of_10000)
#
#     def logp(self, value):
#         f_1_of_10000 = self.f_1_of_10000
#         p = ifelse(
#             tt.eq(f_1_of_10000, 1),
#             ifelse(tt.eq(value, 1), 0.0001, 0.9999),
#             tt.cast(0.5000, 'float64')
#         )
#
#         return bound(tt.log(p),
#             value >= 0, value <= 1,
#             p >= 0, p <= 1)

class I_am_HIV_positive(Discrete):
    def __init__(self, f_1_of_10000, ptable, *args, **kwargs):
        super(I_am_HIV_positive, self).__init__(*args, **kwargs)
        self.f_1_of_10000 = tt.as_tensor_variable(f_1_of_10000)
        self.ptable = tt.as_tensor_variable(ptable)

    def logp(self, value):

        def switch_case(ptable_row, f_1_of_10000, v):
            tt.printing.Print('ptable_row')(ptable_row)
            tt.printing.Print('f_1_of_10000')(f_1_of_10000)
            #print(ptable_row.get_value())

            # f_1_of_10000 = parents[0]
            # value = parents[1]
            if (tt.eq(f_1_of_10000, ptable_row[0]) and tt.eq(v, ptable_row[1])):
                tt.printing.Print('ptable_row[2]')(ptable_row[2])
                return ptable_row[2]
            else:
                return tt.constant(0)

        ptable = self.ptable
        v = value
        f_1_of_10000 = self.f_1_of_10000
        components, updates = theano.scan(fn=switch_case, sequences=ptable, non_sequences=[f_1_of_10000, v])
        tt.printing.Print('components')(components)
        p = tt.sum(components)
        tt.printing.Print('p')(p)
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
        if (tt.eq(test_show_positive, 1)):
            if (tt.eq(p_I_am_HIV_positive, 1) and tt.eq(value, 1)):
                p = 0.999
            elif (tt.eq(p_I_am_HIV_positive, 1) and tt.eq(value, 0)):
                p = 0.001
            elif (tt.eq(p_I_am_HIV_positive, 0) and tt.eq(value, 1)):
                p = 0.001
            elif (tt.eq(p_I_am_HIV_positive, 0) and tt.eq(value, 0)):
                p = 0.999
        elif (tt.eq(test_show_positive, 0)):
            p = 0.5
        else:
            return tt.constant(-np.inf)
        return tt.constant(np.log(p))


with pm.Model() as model:
    f_1_of_10000 = pm.Categorical("f_1_of_10000", [0.1, 0.9])

    #p_I_am_HIV_positive = pm.Binomial('p_I_am_HIV_positive', 1, 0.3)
    ptable = [
        [0, 0, 0.5],
        [0, 1, 0.5],
        [1, 0, 0.0001],
        [1, 1, 0.9999],
    ]
    p_I_am_HIV_positive = I_am_HIV_positive('p_I_am_HIV_positive', f_1_of_10000, ptable, testval=0)

    # test_show_positive = pm.Categorical("test_show_positive", [0.1, 0.9])
    #
    # p_HIV_test_false_positive = HIV_test_false_positive('p_HIV_test_false_positive', test_show_positive=test_show_positive,
    #                                                     p_I_am_HIV_positive=p_I_am_HIV_positive, testval=1, observed=1)

    step = pm.Metropolis()
    SAMPLE_NUM = 20000
    trace = pm.sample(SAMPLE_NUM, tune=1000, step=step)
    s1 = trace['f_1_of_10000']
    samples = trace['p_I_am_HIV_positive']
    print(sum(samples) / SAMPLE_NUM)