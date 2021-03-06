
# coding: utf-8

# In[1]:

import theano
import theano.tensor as T
import numpy
import cPickle

# In[62]:

def SharedZeros(shape):
    zeros = numpy.zeros(shape, dtype=theano.config.floatX)
    return theano.shared(zeros, borrow=True)

class Blender (object):
    def __init__(self, rng, input, shape, activation=T.tanh):
        # input: n_dense, n_samples
        # shape: (n_dense, n_samples)
        self.input = input
        
        W_bound = numpy.sqrt(6./(shape[1]+1))
        self.W = theano.shared(value=numpy.asarray(
                rng.uniform(low=-W_bound,high=W_bound,size=shape),
                dtype=theano.config.floatX
                ))
        self.b = theano.shared(value=numpy.zeros((shape[0],), dtype=theano.config.floatX))
        
        # output
        self.output = T.tanh(T.sum(self.input*self.W, axis=1) + self.b)
        
        # params
        self.params = [self.W, self.b]
        
        # regularization
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2).sum()
        
        #
        self.Wgrad_hist = SharedZeros(shape)
        self.bgrad_hist = SharedZeros((shape[0],))
        
        self.grad_hist = [self.Wgrad_hist, self.bgrad_hist]


# In[63]:

class MLP(object):
    def __init__(self, rng, input, n_in, n_h, n_out, activation=T.tanh):
        self.input = input
        self.activation = activation
        # params for hidden layer
        self.W_h = theano.shared(value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_h)),
                    high=numpy.sqrt(6. / (n_in + n_h)),
                    size=(n_in, n_h)),
                dtype=theano.config.floatX
            ))

        self.b_h = theano.shared(value=numpy.zeros((n_h,), dtype=theano.config.floatX), borrow=True)
        # params for out layer
        self.W_out = theano.shared(value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_h + n_out)),
                    high=numpy.sqrt(6. / (n_h + n_out)),
                    size=(n_h, n_out)),
                dtype=theano.config.floatX
            ))

        self.b_out = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), borrow=True)
        
        # output
        h_out = self.activation(T.dot(self.input, self.W_h) + self.b_h)
        self.output = self.activation(T.dot(h_out, self.W_out) + self.b_out)
        
        # params
        self.params = [self.W_h, self.b_h, self.W_out, self.b_out]
        
        # regularization
        self.L1 = abs(self.W_h).sum() + abs(self.W_out).sum()
        self.L2 = (self.W_h ** 2).sum() + (self.W_out ** 2).sum()
        
        #
        self.Whgrad_hist = SharedZeros((n_in,n_h))
        self.bhgrad_hist = SharedZeros((n_h,))
        self.Wograd_hist = SharedZeros((n_h, n_out))
        self.bograd_hist = SharedZeros((n_out,))
        
        self.grad_hist = [self.Whgrad_hist, self.bhgrad_hist, self.Wograd_hist, self.bograd_hist]

# In[64]:

class LinearLayer(object):
    def __init__(self, weights, fvalues):
        self.output = T.dot(fvalues, weights)


# In[73]:

class BlenderModel(object):
    def __init__(self, input, input_shape, batch_size):
        input = theano.shared(numpy.asarray(input, dtype=theano.config.floatX))
        fvalues = T.matrix()
        
        shape=input_shape
        
        rng = numpy.random.RandomState(1234)
        # first layer blender
        self.blender = Blender(rng, input, shape)
        # second layer MLP
        n_in = shape[0]
        n_hidden = 100
        n_out = n_in
        self.mlp = MLP(rng, self.blender.output, n_in, n_hidden, n_out)
        # third layer linear
        linear = LinearLayer(self.mlp.output, fvalues)
        
        #sigmoid = T.tanh(linear.output)
        # output
        self.output = linear.output
        # cost function
        l1 = 0
        l2 = 0.0001
        
        reshaped = self.output.reshape((-1, 2))
        #w = theano.shared(value=numpy.asarray([1.,-1.]))
        delta = T.nnet.softmax(T.nnet.sigmoid(reshaped))#T.dot(reshaped, w) + 1
        positive_delta = -T.log(delta[:,1])
        self.cost = T.mean(positive_delta) + l1 * ( self.blender.L1 + self.mlp.L1 ) + l2 * ( self.blender.L2 + self.mlp.L2 )
        #params
        self.params = self.mlp.params + self.blender.params
        self.grad_hist = self.mlp.grad_hist + self.blender.grad_hist
        
        # gradient
        gparams = T.grad(self.cost, self.params)
        # updates, adagrad
        self.learning_rate = 0.1
        updates = []
        for param, gparam, grad_hist in zip(self.params, gparams, self.grad_hist):
            gh = grad_hist + gparam ** 2
            sq = T.sqrt(gh)
            sq = T.set_subtensor(sq[sq.nonzero()], self.learning_rate/sq[sq.nonzero()])
            updates.append((param, param-sq*gparam))
            updates.append((grad_hist, gh))
        
        # define the function
        self.train = theano.function([fvalues], self.cost, updates = updates)
        self.weight = theano.function([], self.mlp.output)
        #self.weight = mlp.output
        
        # reset grad hist
        xx = T.scalar()
        updates_reset = [(gh, T.set_subtensor(gh[:],xx)) for gh in self.grad_hist]
        self.reset_grad_hist = theano.function([xx],
                                               updates=updates_reset)
    
    def ResetGradHist(self):
        self.reset_grad_hist(0.)
    
    def Train(self, fv):
        #input = numpy.asarray(weights, dtype=theano.config.floatX)
        fvalues = numpy.asarray(fv, dtype=theano.config.floatX)
        #dbleu = numpy.asarray(dbleu, dtype=theano.config.floatX)
        
        cost = self.train(fvalues)
        
        return cost
    
    
    def Weight(self):
        #input = numpy.asarray(weights, dtype=theano.config.floatX)
        w_new = self.weight()
        return w_new
