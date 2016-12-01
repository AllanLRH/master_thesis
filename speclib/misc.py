nanEqual = lambda a, b: np.all((a == b) | (np.isnan(a) & np.isnan(b))) 
