# dotdict.py

class DotDict(dict):
    """Permet d'accéder aux clés d'un dictionnaire via la notation par points (dot notation)."""
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __delattr__(self, attr):
        del self[attr]

    # ------- W ---------#
    def __getstate__(self):
        return self.copy()
    
    def __setstate__(self, state):
        self.update(state)
    # ------- W ---------#