import numpy as np
import pandas as pd





X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)


# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train = X[:50000]
    y_train = y[:50000]
    X_test = X[50000:]
    y_test = y[50000:]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    K =np.max(y)
    class_priors = [np.mean(y == (c + 1)) for c in range(K)]
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    D = X.shape[1]

    X_1 = X[y == 1]
    X_2 = X[y == 2]

    pAcd = np.array([[np.mean((X_1[:, d] == 'A')) for d in range(D)], [np.mean((X_2[:, d] == 'A')) for d in range(D)]])
    pCcd = np.array([[np.mean((X_1[:, d] == 'C')) for d in range(D)], [np.mean((X_2[:, d] == 'C')) for d in range(D)]])
    pGcd = np.array([[np.mean((X_1[:, d] == 'G')) for d in range(D)], [np.mean((X_2[:, d] == 'G')) for d in range(D)]])
    pTcd = np.array([[np.mean((X_1[:, d] == 'T')) for d in range(D)], [np.mean((X_2[:, d] == 'T')) for d in range(D)]])

    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)

# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below

    K = np.max(y)

    score_values = [np.sum(
        np.log(
            np.power(pAcd[c, :], (X[:, :] == 'A').astype(int)) * np.power(pCcd[c, :], (X[:, :] == 'C').astype(int)) * np.power(pGcd[c, :], (X[:, :] == 'G').astype(int))
            * np.power(pTcd[c, :], (X[:, :] == 'T').astype(int))), axis=1)
                    + np.log(class_priors[c]) for c in range(K)]
    score_values = np.transpose(score_values)

    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    y_predicted = np.argmax(scores,axis=1).round()

    confusion_matrix = pd.crosstab(y_predicted.T, y_truth.T).values
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
