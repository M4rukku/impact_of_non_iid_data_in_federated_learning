class UnmodifiableAttributes(object):
    '''
    Ensures that inheritng objects will not be able to modify their variables -- this allows us
    to treat the Dataset Processors as quasi immutable objects (They may allocate state,
    but it will be constant over all instances given the same constructor parameters)
    '''
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise RuntimeError("Can't modify immutable object's attribute: {}".format(key))