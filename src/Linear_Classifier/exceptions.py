
class EmptyDataError(Exception):
    """
    Gets thrown when the Data is of size 0, and therefore nothing
    else can be done.
    """
    pass


class EmptyClfError(Exception):
    """
    Gets thrown when a network has not been correctly initialized
    """
    pass
