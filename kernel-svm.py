import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# visualizing decision boundary
def plot_bounds(X,Y,model=None,classes=None, figsize=(8,6)):
        
    plt.figure(figsize=figsize)
        
    if(model):
        X_train, X_test = X
        Y_train, Y_test = Y
        X = np.vstack([X_train, X_test])
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                             np.arange(y_min, y_max, .02))

        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=.8)

    plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, alpha=0.6)
    
    plt.show()

from sklearn.datasets import make_circles

X, Y = make_circles(noise=0.2, factor=0.5, random_state=1)
X.shape

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# linear kernel
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, Y_train)

score = svc.score(X_test, Y_test)
score_train = svc.score(X_train, Y_train)

print(f'Linear - TEST: {score} / TRAIN: {score_train}')

plot_bounds((X_train, X_test), (Y_train, Y_test), svc)


# poly kernel
svc = SVC(kernel='poly', probability=True)
svc.fit(X_train, Y_train)

score = svc.score(X_test, Y_test)
score_train = svc.score(X_train, Y_train)

print(f'Poly - TEST: {score} / TRAIN: {score_train}')

plot_bounds((X_train, X_test), (Y_train, Y_test), svc)


# sigmoid kernel
svc = SVC(kernel='sigmoid', probability=True)
svc.fit(X_train, Y_train)

score = svc.score(X_test, Y_test)
score_train = svc.score(X_train, Y_train)

print(f'Sigmoidal - TEST: {score} / TRAIN: {score_train}')

plot_bounds((X_train, X_test), (Y_train, Y_test), svc)


# gaussian kernel
svc = SVC(kernel='rbf', probability=True)
svc.fit(X_train, Y_train)

score = svc.score(X_test, Y_test)
score_train = svc.score(X_train, Y_train)

print(f'Gaussian - TEST: {score} / TRAIN: {score_train}')

plot_bounds((X_train, X_test), (Y_train, Y_test), svc)


# iterative test with different kernel and gamma parameters
kernels = ['linear', 'poly', 'sigmoid', 'rbf']

best_score = 0
best_diff = 1
best_kernel = None
best_gamma = None

for gamma in np.arange(0.1, 5.0, 0.1):
    for kernel in kernels:
        svc = SVC(kernel=kernel, gamma=gamma, probability=True)
        svc.fit(X_train, Y_train)

        score = svc.score(X_test, Y_test)
        score_train = svc.score(X_train, Y_train)

        diff = abs(score_train - score)

        # set the min diff manually or with the lowest possibile
        if(score > best_score and diff < best_diff):
            best_score = score
            best_diff = diff
            best_kernel = kernel
            best_gamma = gamma
        
print(f'BEST - kernel: {best_kernel} + gamma: {best_gamma} -> score: {best_score}, diff: {best_diff}')

