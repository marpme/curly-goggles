import numpy as np

class Flatten():
    ''' Flatten layer used to reshape inputs into vector representation

    Layer should be used in the forward pass before a dense layer to
    transform a given tensor into a vector.
    '''
    def __init__(self):
        self.params = []

    def forward(self, X):
        ''' Reshapes a n-dim representation into a vector
            by preserving the number of input rows.

        Examples:
            [10000,[1,28,28]] -> [10000,784]
        '''
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.reshape(-1).reshape(self.out_shape)
        return out

    def backward(self, dout):
        ''' Restore dimensions before flattening operation
        '''
        out = dout.reshape(self.X_shape)
        return out, []

class FullyConnected():
    ''' Fully connected layer implemtenting linear function hypothesis
        in the forward pass and its derivation in the backward pass.
    '''
    def __init__(self, in_size, out_size):
        ''' Initilize all learning parameters in the layer

        Weights will be initilized with modified Xavier initialization.
        Biases will be initilized with zero.
        '''
        self.W = np.random.randn(in_size, out_size) * np.sqrt(2. / in_size)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = np.add(np.dot(self.X, self.W), self.b)
        return out

    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        dW = np.dot(self.X.T, dout)
        db = np.sum(dout, axis=0)
        return dX, [dW, db]

class Conv():
    ''' Description
    '''
    def __init__(self, X_dim, filter_num, filter_dim, stride, padding):
        """ 
        Args:
            X_dim: dimension of the squared image 
            filter_num: a filter for the convolution
            filter_dim: step size with which the kernel slides over the image
            stride: 
            padding: if set zero padding will be applied to keep image dimensions
        """
        self.dim = X_dim
        self.kernel = filter_num
        self.kernel_dim = filter_dim
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        pad = int((self.kernelLen - 1) / 2)
        if self.padding == True: 
            image = np.pad(image, (pad,pad) ,'constant', constant_values=(0, 0))
            
        output = []
        (width, height) = self.dim
        for h in tqdm(range(0+pad, height-pad, self.stride), "Image ..."):
            output.append([])
            for w in range(0+pad, width-pad, self.stride):
                output[-1].append([])
                if self.padding == True:
                    subImage = image[h:h+self.kernelLen, w:w+self.kernelLen]
                    output[-1][-1] = np.sum(np.multiply(subImage, self.kernel))
                else:
                    subImage = image[h:h+self.kernelLen, w:w+self.kernelLen]
                    output[-1][-1] = np.sum(np.multiply(subImage, self.kernel))
                
        return np.array(output)

    def backward(self, dout):
        return None


class Pool():
    ''' Description
    '''
    def __init__(self, X_dim, func, filter_dim, stride):
        self.dim = image_dim
        self.function = pooling_function
        self.stride = stride
        self.size = pooling_size

    def forward(self, X):
        output = []
        (width, height) = self.dim
        for h in tqdm(range(0, height, self.stride), "Image ..."):
            output.append([])
            for w in range(0, width, self.stride):
                output[-1].append([])
                subImage = image[h:h+self.size, w:w+self.size]
                output[-1][-1] = self.function(subImage)
        
        return output

    def backward(self, dout):
        return None

class Batchnorm():
    ''' Description
    '''
    def __init__(self, X_dim):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None


class Dropout():
    ''' Description
    '''
    def __init__(self, prob=0.5):
        None

    def forward(self, X):
        return None

    def backward(self, dout):
        return None


