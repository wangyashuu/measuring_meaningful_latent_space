from sklearn.preprocessing import scale


def atleast_2d(arr):
    if len(arr.shape) == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr


def is_1d(x):
    return len(x.shape) <= 1 or (len(x.shape) == 2 and x.shape[1] == 1)


def default_transform(X):
    return scale(X, with_mean=False)
