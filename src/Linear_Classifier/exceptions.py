class WrongURLError(Exception):
    """
    Gets thrown when the Data is of size 0, and therefore nothing
    else can be done.
    """
    pass



class EmptyDataError(Exception):
    """
    Gets thrown when the Data is of size 0, and therefore nothing
    else can be done.
    """
    pass


class EmptyNetError(Exception):
    """
    Gets thrown when a network has not been correctly initialized
    """
    pass

class NoModeError(Exception):
    """
    The mode for testing has not been specified
    """
    pass

class WrongInitError(Exception):
    """
    The initialization is incorrect
    """
    pass
