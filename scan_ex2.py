import theano
import theano.tensor as T
#theano.config.compute_test_value = 'off'

import numpy as np

x = theano.shared(np.asarray([7.1,2.2,3.4], dtype = np.float32))

print(x.get_value())
T.printing.Print('x)')(x)


v = T.vector("v")
def fv(v):
    res,_ = theano.scan(lambda x: x ** 2, v)
    return T.sum(res)

def f(i):
    return fv(x[i:i+2])

T.printing.Print('T.arange(2)')(T.arange(2))

outs,_ = theano.scan(
    f,
    T.arange(2)
    )

fn = theano.function(
    [],
    outs,
    )

fn()