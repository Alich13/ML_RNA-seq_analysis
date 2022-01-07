import numpy as np

def top_k_variance(X,  k=3000, no_variance_filter=True,names=None):
    """
    
    Utility function for retaining top k features with the highest variance.
    
    Args:
        X ([DataFrame]): dataframe containing all the features
        names ([type], optional): [description]. Defaults to None.
        k (int, optional): [description]. Defaults to 3000.
        no_variance_filter (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    var = np.var(X, axis=0)
    if no_variance_filter:
        mask = var != 0
        X = X[:, mask]
        var = np.var(X, axis=0)
        names = names[mask]
        
    sorted_var = np.sort(var)[::-1]
    threshold = sorted_var[k - 1]

    mask = var >= threshold
    if names is not None:
        return X[:, mask], names[mask]
    else:
        return X[:, mask]