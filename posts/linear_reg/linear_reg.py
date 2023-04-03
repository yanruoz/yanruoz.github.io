import numpy as np

class LinearRegression():
    def __init__(self, w = None):
        """ Define instance variables. 
        w -- the weight vector.
        loss_history -- a list of the evolution of the loss over the training period. 
        score_history -- a list of the evolution of the score over the training period.
        """
        self.w = w
        self.loss_history = []
        self.score_history = []
    
    def fit_gradient (self, X, y, alpha, max_epochs):
        """Fit the logistic regression model to a dataset, using gradient descent.
        X -- the feature vector.
        y -- the label vector.
        alpha -- the learning rate.
        max_epochs -- the maximum number of iterations.
        """

        # reset history
        # self.score_history = []
        
        # obtain n (number of data points), p (number of features)
        n, p = X.shape

        # initialize a random initial weight vector w_tilt(0)
        self.w = np.random.uniform(-1,1, size=(p+1,))
        
        # intialize prev_loss later used to terminate the loop
        prev_loss = np.inf
        
        # calculate the gradient
        X_ = self.pad(X)
        P = X_.T@X_
        q = X_.T@y
        
        for epoch in range(max_epochs):
            
            # gradient step
            gradient = P@self.w - q
            self.w -= 2 * alpha * gradient
            # print(self.w)

            # compute loss
            new_loss = self.loss(X, y)

            # compute score
            score = self.score(X, y)
            self.score_history.append(score)

            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss

    def fit(self, X, y, method = 'analytic', m_epochs = 100, batch_size = 10, alpha = 0.01, momentum = False):
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
        
        
        
        if method == 'gradient':
            # reset history
            # self.score_history = []
            
            # obtain n, p
            n, p = X.shape
            
            # initialize a random initial weight vector w_tilt(0)
            self.w = np.random.uniform(-1,1, size=(p+1,))

            # initialize prev_w for momentum
            prev_w = np.zeros_like(self.w)

            # intialize prev_loss later used to terminate the loop
            prev_loss = np.inf
            
            # calculate the gradient
            X_ = self.pad(X)
            P = X_.T@X_
            q = X_.T@y

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

                    # update w
                    gradient = P@self.w - q
                    self.w -= 2 * alpha * gradient
                    # print(f"{self.w=}")

                    # store prev_w for next loop
                    prev_w = self.w

                # compute loss
                new_loss = self.loss(X, y)

                # check if loss hasn't changed and terminate if so
                if np.isclose(new_loss, prev_loss):
                    break
                else:
                    prev_loss = new_loss

                # compute the score
                score = self.score(X, y)
                self.score_history.append(score)
            
        elif method == 'analytic':
            # calculate w_hat -- analytic
            X = self.pad(X)
            self.w = np.linalg.inv(X.T@X)@X.T@y
            
    def score(self, X, y):
        """Return the coefficient of determination as the score to evaluate the prediction. The coefficient of determination is always no larger than 1, with a higher value indicating better predictive performance. It can be arbitrarily negative for very bad models. """
        y_hat = self.predict(X)
        top = np.sum((y_hat - y)**2)
        bottom = np.sum((np.mean(y)-y)**2)
        return 1 - top/bottom

    def predict(self, X):
        X = self.pad(X)
        return X@self.w
    
    def loss(self, X, y):
        """Return the overall loss (empirical risk) of the current weights on the feature vecotr X and the label vector y. """
        
        loss = np.linalg.norm(self.pad(X)@self.w-y, 2) ** 2
        return loss
    
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
