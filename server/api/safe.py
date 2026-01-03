import logging
from functools import wraps
from rest_framework import status
from .utils import err

logger = logging.getLogger("docqa_api")


def safe_api(fn):
    """
    Wrap API views:
    - catch unexpected exceptions
    - log full traceback to file
    - return clean JSON error
    """
    @wraps(fn)
    def wrapper(request, *args, **kwargs):
        try:
            return fn(request, *args, **kwargs)
        except Exception as e:
            logger.exception(
                "Unhandled exception in %s | path=%s | method=%s",
                fn.__name__,
                request.path,
                request.method,
            )
            return err(
                message="Internal server error",
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
    return wrapper
