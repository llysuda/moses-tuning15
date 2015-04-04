
# coding: utf-8

# In[1]:

import theano
import theano.tensor as T
import numpy
import cPickle

# In[62]:

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
        self.b = theano.shared(value=numpy.zeros(shape[0], dtype=theano.config.floatX))
        
        # output
        self.output = T.tanh(T.sum(self.input*self.W, axis=1) + self.b)
        
        # params
        self.params = [self.W, self.b]
        
        # regularization
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2).sum()


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
        self.L2 = (self.W_h ** 2).sum() + (self.W_h ** 2).sum()

# In[64]:

class LinearLayer(object):
    def __init__(self, weights, fvalues):
        self.output = T.dot(fvalues, weights)


# In[73]:

class BlenderModel(object):
    def __init__(self, input_shape, batch_size):
        input = T.matrix()
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
        l1 = 0.001
        l2 = 0
        
        reshaped = self.output.reshape((batch_size, 2))
        w = theano.shared(value=numpy.asarray([1.,-1.]))
        delta = T.sum (reshaped * w, axis=1)
        positive_delta = delta * (delta > 0) + 1
        self.cost = T.mean(positive_delta) + l1 * ( self.blender.L1 + self.mlp.L1 ) + l2 * ( self.blender.L2 + self.mlp.L2 )
        #params
        self.params = self.mlp.params + self.blender.params
        
        # gradient
        gparams = T.grad(self.cost, self.params)
        # updates
        self.learning_rate = 0.1
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param-self.learning_rate*gparam))
        # define the function
        self.train = theano.function([input, fvalues], self.cost, updates = updates)
        self.weight = theano.function([input], self.mlp.output)
        #self.weight = mlp.output
    
    def Train(self, weights, fv):
        input = numpy.asarray(weights, dtype=theano.config.floatX)
        fvalues = numpy.asarray(fv, dtype=theano.config.floatX)
        
        #print input
        #print fvalues
        
        cost = self.train(input, fvalues)
        
        return cost
    
#     def Load(self, file_path):
#         params = cPickle.load(open(file_path,'rb'))
#         self.mlp.W_h = params[0]
#         self.mlp.b_h = params[1]
#         self.mlp.W_out = params[2]
#         self.mlp.b_out = params[3]
#         self.blender.W = params[4]
#         self.blender.b = params[5]
#         
#     
#     def Save(self, file_path):
#         cPickle.dump(self.params, open(file_path,'wb'))
    
    def Weight(self, weights):
        input = numpy.asarray(weights, dtype=theano.config.floatX)
        w_new = self.weight(input)
        return w_new
