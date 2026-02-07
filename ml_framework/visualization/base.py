class Plotter:
    """Base class for all visualization."""
    
    def plot(self, *args, **kwargs):
        """Create the plot."""
        raise NotImplementedError
