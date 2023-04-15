def _raise_if(condition: bool, exception: Exception):
    if condition:
        raise exception


def raise_if(condition: bool, message: str):
    """Check the condition and throw an error if True.

    Parameters
    ----------
        condition : bool
            condition to raise an exception
        message : string
            exception message

    Raises
    -------
        ValueError
            If condition is True.
    """
    _raise_if(condition, ValueError(message))


def raise_data_validation_error_if(condition: bool, message: str):
    """Check the condition and throw an error if true.

    Parameters
    ----------
        condition : bool
            condition to raise an exception
        message : string
            exception message

    Raises
    -------
        DataValidationError
            If condition is true.
    """
    _raise_if(condition, DataValidationError(message))


class DataValidationError(Exception):
    pass
