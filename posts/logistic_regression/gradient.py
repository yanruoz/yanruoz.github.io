import numpy as np

class LogisticRegression():
    def __init__(self, w = None):
        """ Define instance variables. 
        w -- the weight vector.
        loss_history -- a list of the evolution of the loss over the training period. 
        score_history -- a list of the evolution of the score over the training period.
        """
        self.w = w
        self.loss_history = []
        self.score_history = []
     
    def fit(self, X, y, alpha, max_epochs):
        """Fit the logistic regression model to a dataset, using gradient descent.
        X -- the feature vector.
        y -- the label vector.
        alpha -- the learning rate.
        max_epochs -- the maximum number of iterations.
        """

        # reset history
        self.loss_history = []
        self.score_history = []
        
        # obtain n (number of data points), p (number of features)
        n, p = X.shape

        # initialize a random initial weight vector w_tilt(0)
        self.w = np.random.uniform(-1,1, size=(p+1,))
        
        # intialize prev_loss later used to terminate the loop
        prev_loss = np.inf
        
        for epoch in range(max_epochs):
            # gradient step
            self.w -= alpha * self.gradient(X, y)

            # compute loss
            new_loss = self.loss(X, y)
            self.loss_history.append(new_loss)

            # compute score
            score = self.score(X, y)
            self.score_history.append(score)

            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss
    
    def fit_stochastic(self, X, y, m_epochs, batch_size, alpha, momentum = False):
        """Fit the logistic regression model to a dataset, using stochastic gradient descent.
        X -- the feature vector.
        y -- the label vector.
        m_epochs -- the maximum number of iterations.
        batch_size -- the number of data points in each randomly picked subset.
        alpha -- the learning rate.
        momentum -- True to include momentum in the model; False to not include. 
        """
        # Sources:
        # the skeleton of the implementation of momentom is from the instruction for this blog post
        # https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html
        
        # reset history
        self.loss_history = []
        
        # obtain n, p
        n, p = X.shape
        
        # initialize a random initial weight vector w_tilt(0)
        self.w = np.random.uniform(-1,1, size=(p+1,))
        
        # initialize prev_w for momentum
        prev_w = np.zeros_like(self.w)
        
        # intialize prev_loss later used to terminate the loop
        prev_loss = np.inf
        
        # implement momentum
        if momentum:
            beta = 0.8
            
        else:
            beta = 0
        
        # main loop
        # Source code:
        # the main loop structure is based on the instruction of the blog post: https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html
        for j in np.arange(m_epochs):
            
            # shuffle the points randomly
            order = np.arange(n) # create an array for numbers from 0 to n
            np.random.shuffle(order) # shuffle the array
            
            # pick k random points, compute the stochastic gradient, and update
            for batch in np.array_split(order, n // batch_size + 1):
               
                # compute the stochastic gradient
                X_batch = X[batch,:]
                # print(X_batch)
                # print(f"{X_batch.shape=}")
                y_batch = y[batch]
                # print(f"{y_batch.shape=}")
                
                grad = self.gradient(X_batch, y_batch) 
                
                # gradient step update
                self.w = self.w - alpha * grad + beta * (self.w - prev_w)
                # print(f"{self.w=}")
                
                # store prev_w for next loop
                prev_w = self.w

            # compute loss and add to loss_history
            new_loss = self.loss(X, y)
            self.loss_history.append(new_loss)
            
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss

    
    def gradient(self, X, y):
        """Return the gradient based on a feature vector X and a label vector y. """
        
        X_ = self.pad(X)
        # print(f"{X_=}")
        
        # self.w = self.w[:,np.newaxis]
        # print(f"{self.w=}")
        
        # y = y[:,np.newaxis]
        # print(f"{y.size=}")
        
        y_hat = X_@self.w
        
        deriv = self.sigmoid(y_hat) - y
        # print(deriv.shape)
        
        gradient = np.mean(X_ * deriv[:,np.newaxis], axis=0)
        
        return gradient
        
    def predict(self, X): 
        """Return the predicted values of a feature vector X. """
        X_ = self.pad(X)
        y_pred = 1*(X_ @ self.w>=0)
        return y_pred

    def logistic_loss(self, y_hat, y):
        """Return the logistic loss based on the label vector y and the corresponding predicted vector y_hat. """
        # print(f"{y_hat=}")
        return -y * np.log(self.sigmoid(y_hat)) - (1-y)* np.log(1-self.sigmoid(y_hat))
    
    def sigmoid(self, y_hat):
        """Return the sigmoid function of y_hat. """
        return 1 / (1 + np.exp(-y_hat))

    def loss(self, X, y):
        """Return the overall loss (empirical risk) of the current weights on the feature vecotr X and the label vector y. """
        y_hat = self.pad(X)@self.w
        loss = self.logistic_loss(y_hat, y).mean()
        # print(y)
        return loss
    
    def score(self, X, y):
        """Return the accuracy of the predictions as a number between 0 and 1, with 1 corresponding to perfect classification. """
        y_pred = self.predict(X)
        accuracy = np.mean((y == y_pred)*1)
        return accuracy         
    
    def pad(self, X):
        """Return the modified feature vector with an added column of 1s for mathematical convenience. """
        return np.append(X, np.ones((X.shape[0], 1)), 1)
                                                                          