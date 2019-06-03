import numpy as np


#----------------------------------------------------------------


class GlobalVariables(object):
    def __init__(self):
        self.trainables = []

_GLOBALS = GlobalVariables()


#----------------------------------------------------------------


class Layer(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        raise Exception("This is pure virtual method")

    def backward(self, delta):
        raise Exception("This is pure virtual method")



class Trainable(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._variables = {}
        self._gradients = {}
        _GLOBALS.trainables.append(self)

    def train(self):
        for i in self._variables:
            self._variables[i] -= self._gradients[i]


#----------------------------------------------------------------


class FullyConnected(Layer, Trainable):
    def __init__(self, prev, size, init_fun=np.random.uniform, init_args={"low":-0.05, "high":0.05}):
        super().__init__()
        self.x = None
        self._prev = prev
        self._variables["weights"] = init_fun(**init_args, size=[prev.shape[-1], size])
        self.shape = tuple([prev.shape[0], size])
        self._gradients["weights"] = np.zeros(shape=[prev.shape[-1], size])
        
    def forward(self, **kwargs):
        self.x = self._prev.forward(**kwargs)
        self.x = self.x @ self._variables["weights"]
        return self.x
    
    def backward(self, delta):
        self_delta = delta @ self._variables["weights"].T
        self._gradients["weights"] = self._prev.x.T @ delta
        self._prev.backward(self_delta)


        
class BatchNormalization(Layer, Trainable):
    def _checkParams(self, dims, prevshape):
        if type(dims) is int:
            dims = tuple([dims])
        else:
            dims = tuple(np.sort(dims))
        if len(prevshape) < dims[-1]+1:
            raise Exception("x has to few dimensions")
        self._dims = dims
        prevshape = list(prevshape[:])
        for d in dims:
            prevshape[d] = 1
        self._statshape = tuple(prevshape[:])
        prevshape[0] = 1
        return prevshape
    
    
    def __init__(self, prev, dims, epsi=1e-8, ema=0.99, init_fun=np.random.uniform, init_args={"low":-0.05, "high":0.05}):
        super().__init__()
        varshape = self._checkParams(dims, prev.shape)
        self.x = None
        self._prev = prev
        self.shape = prev.shape[:]
        self._variables["gamma"] = init_fun(**init_args, size=varshape) + 1
        self._variables["beta"] = init_fun(**init_args, size=varshape)
        self._gradients["gamma"] = np.zeros(shape=varshape) + 1
        self._gradients["beta"] = np.zeros(shape=varshape)
        self._validvars = {}
        self._validvars["mu"] = np.zeros(shape=varshape) + 1
        self._validvars["sigma"] = np.zeros(shape=varshape)
        self._epsi = epsi
        self._ema = ema
        self._d_dx = None
        self._d_dg = None
        
    def forward(self, train=False, **kwargs):
        self.x = self._prev.forward(train=train, **kwargs)
        if train:
            mu = self.x.mean(self._dims).reshape(self._statshape)
            sigma = self.x.var(self._dims).reshape(self._statshape)
            cntr = (self.x - mu)
            denom = np.sqrt(sigma + self._epsi)
            self._d_dx = self._variables["gamma"]/denom
            self._d_dg = np.expand_dims(cntr.mean(0), 0)/denom
            self.x = self._d_dx*cntr + self._variables["beta"]
            if 0 not in self._dims:
                mu = np.expand_dims(mu.mean(0), 0)
                sigma = np.expand_dims(sigma.mean(0), 0)
                self._d_dx = np.expand_dims(self._d_dx.mean(0), 0)
                self._d_dg = np.expand_dims(self._d_dg.mean(0), 0)
            self._validvars["mu"] = self._ema*self._validvars["mu"] + (1-self._ema)*mu
            self._validvars["sigma"] = self._ema*self._validvars["sigma"] + (1-self._ema)*sigma
        else:
            a = self._variables["gamma"]/np.sqrt(self._validvars["sigma"] + self._epsi)
            self.x = a*(self.x - self._validvars["mu"]) + self._variables["beta"]
        return self.x
    
    def backward(self, delta):
        self_delta = delta * self._d_dx
        delta = np.mean(delta, self._dims).reshape(self._gradients["beta"].shape)
        self._gradients["beta"] = delta
        self._gradients["gamma"] = delta * self._d_dg
        self._prev.backward(self_delta)
        
        
#----------------------------------------------------------------


class Placeholder(Layer):
    def __init__(self, sizes):
        self.x = None
        self.shape = tuple(sizes)
        
    def forward(self, **kwargs):
        return self.x
    
    def backward(self, delta):
        pass
    
    def feed(self, data):
        if data.shape[1:] != self.shape[1:]:
            raise Exception("Bad sizes")
        self.x = data



class Dropout(Layer):
    def __init__(self, prev, keep_prob=1.0):
        self.x = None
        self._prev = prev
        self._keep_prob = keep_prob
        self.shape = tuple(prev.shape)
        self._filter = None
        
    def forward(self, train=False, **kwargs):
        self.x = self._prev.forward(train=train, **kwargs)
        if train:
            self._filter = np.random.binomial(1, 1-self._keep_prob, size=self.x.shape)
            self.x[self._filter] = 0
            self.x /= self._keep_prob
        return self.x
    
    def backward(self, delta):
        self_delta = delta * self._keep_prob
        self_delta[self._filter] = 0
        self._prev.backward(self_delta)



class ReLU(Layer):
    def __init__(self, prev):
        self.x = None
        self._prev = prev
        self.shape = tuple(prev.shape)
        
    def forward(self, **kwargs):
        self.x = self._prev.forward(**kwargs)
        self.x *= (self.x > 0).astype(np.int8)
        return self.x
    
    def backward(self, delta):
        self_delta = delta*(self._prev.x > 0).astype(np.int8)
        self._prev.backward(self_delta)



class Sigmoid(Layer):
    def __init__(self, prev):
        self.x = None
        self._prev = prev
        self.shape = tuple(prev.shape)
    
    def forward(self, **kwargs):
        self.x = self._prev.forward(**kwargs)
        self.x = 1/(1 + np.exp(-self.x))
        return self.x
    
    def backward(self, delta):
        self_delta = self.x*(1-self.x)*delta
        self._prev.backward(self_delta)



class Softmax(Layer):
    def __init__(self, prev):
        self.x = None
        self._prev = prev
        self.shape = tuple(prev.shape)
        
    def forward(self, **kwargs):
        self.x = self._prev.forward(**kwargs)
        self.x = np.exp(self.x)
        self.x /= np.sum(self.x, -1)[:,None]
        return self.x
    
    def backward(self, delta):
        self_delta = np.zeros(delta.shape)
        for i in range(self_delta.shape[1]):
            for k in range(self_delta.shape[1]):
                self_delta[:, i] += delta[:,k]*(self.x[:,i]*(1-self.x[:,i]) if i==k else -self.x[:,i]*self.x[:,k])
        self._prev.backward(self_delta)

#----------------------------------------------------------------

class SoftmaxCrossEntropy(object):
    def __init__(self, predicts, labels, learn_rate=0.001):
        if predicts.shape[1:] != labels.shape[1:]:
            raise Exception("Bad sizes")
        self._predicts = predicts
        self._labels = labels
        self._learn_rate = learn_rate
        
    @staticmethod
    def _softmax(x):
        s = np.exp(x)
        s /= np.sum(s, -1)[:,None]
        return s*(1-1e-8) # to not obtain 0 or 1 for numerical stable
    
    @staticmethod
    def _cross_entropy(s, y):
        loss = -np.sum(y*np.log(s) + (1-y)*np.log(1-s))/y.shape[0]
        return np.nan_to_num(loss)
    
    def run(self):
        y = self._labels.forward()
        x = self._predicts.forward(train=True)
        s = SoftmaxCrossEntropy._softmax(x)
        delta = -y + (1-y)*s/(1-s)
        for k in range(delta.shape[1]):
            delta[:,:] += (y-s)[:,k][:,None] * (s/(1-s[:,k][:,None]))
        self._predicts.backward((1/y.shape[0])*np.nan_to_num(delta)*self._learn_rate)

        for trainable in _GLOBALS.trainables:
            trainable.train()

        return SoftmaxCrossEntropy._cross_entropy(s, y)



class SigmoidCrossEntropy(object):
    def __init__(self, predicts, labels, learn_rate=0.001):
        if predicts.shape[1:] != labels.shape[1:]:
            raise Exception("Bad sizes")
        self._predicts = predicts
        self._labels = labels
        self._learn_rate = learn_rate
        
    @staticmethod
    def _sigmoid(x):
        sigma = 1/(1 + np.exp(-x))
        return sigma*(1-1e-8) # to not obtain 0 or 1 for numerical stable
    
    @staticmethod
    def _cross_entropy(sigma, y):
        loss = -np.sum(y*np.log(sigma) + (1-y)*np.log(1-sigma))/y.shape[0]
        return np.nan_to_num(loss)
    
    def run(self):
        y = self._labels.forward()
        x = self._predicts.forward(train=True)
        sigma = SigmoidCrossEntropy._softmax(x)
        delta = sigma - y
        self._predicts.backward((1/y.shape[0])*np.nan_to_num(delta)*self._learn_rate)

        for trainable in _GLOBALS.trainables:
            trainable.train()

        return SigmoidCrossEntropy._cross_entropy(sigma, y)
