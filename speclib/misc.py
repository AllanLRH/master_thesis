def nanEqual(a, b):
    if not (hasattr(a, 'shape') or hasattr(b, 'shape')):
        raise ValueError("Inputs a nad b must both have the shape-attribute (like numpy arrays)")
    if a.shape != b.shape:
        raise ValueError("Inputs shapes must be identical. a.shape = {} and b.shape is {}".format(a.shape, b.shape)
    return np.all( (a == b) | (np.isnan(a) & np.isnan(b)) ) 
