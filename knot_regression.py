"""Ridge and Lasso regression are powerful techniques generally used for creating parsimonious models in presence of a ‘large’ number of features.
Here ‘large’ can typically mean either of two things:

Large enough to enhance the tendency of a model to overfit (as low as 10 variables might cause overfitting)
Large enough to cause computational challenges.

With modern systems, this situation might arise in case of millions or billions of features
Though Ridge and Lasso might appear to work towards a common goal, the inherent properties and practical use cases
differ substantially.

If you’ve heard of them before, you must know that they work by penalizing the magnitude of coefficients of features
along with minimizing the error between predicted and actual observations.

These are called ‘regularization’ techniques. The key difference is in how they assign penalty to the coefficients:

Ridge Regression:

Performs L2 regularization, i.e. adds penalty equivalent to square of the magnitude of coefficients

Minimization objective = LS Obj + α * (sum of square of coefficients)


Lasso Regression:
Performs L1 regularization, i.e. adds penalty equivalent to absolute value of the magnitude of coefficients
Minimization objective = LS Obj + α * (sum of absolute value of coefficients)
Note that here ‘LS Obj’ refers to ‘least squares objective’, i.e. the linear regression objective without regularization.

If terms like ‘penalty’ and ‘regularization’ seem very unfamiliar to you, don’t worry we’ll talk about these in more detail through the course of this article. Before digging further into how they work, lets try to get some intuition into why penalizing the magnitude of coefficients should work in the first place.
"""

#importing libraries.
import numpy as np
import pandas as import pd
import random
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

"""
Objective = RSS + alpha * (sum of square of coefficients)
"""

from sklearn.linear_model import Ridge

def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #fit the model
    ridgereg = Ridge(alpha=alpha, normalize=true)
    ridgereg.fit(data[predictors], data['y'])
    y_pred = ridgereg.predict(data[predictors])

    #to plot or not to plot?
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('A plot for alpha : %.3g' %alpha)

    #return
    rss = sum((y_pred-data['y']) ** 2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

    #initialize predictors to be set of 15 powers of x
    predictors=['x']
    predictors.extend(['x_%d' %i for i in range(2,16)])

    #set different values of alpha for testing
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

    #finite difference frame-parametrization
    col = ['rss', 'intercept'] + ['coef_x_%d' %i for i in range(1,16)]
    ind = ['alpha_%.2g' %alpha_ridge[i] for i in range(0,10)]
    coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

    models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
        for i in range(10):
            coef_matrix_ridge.iloc[i, ] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
