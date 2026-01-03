from rest_framework.response import Response
from rest_framework import status


def ok(data: dict, http_status=status.HTTP_200_OK) -> Response:
    payload = {"status": "ok"}
    payload.update(data)
    return Response(payload, status=http_status)


def err(message: str, http_status=status.HTTP_400_BAD_REQUEST, details=None) -> Response:
    payload = {"status": "error", "error": message}
    if details is not None:
        payload["details"] = details
    return Response(payload, status=http_status)