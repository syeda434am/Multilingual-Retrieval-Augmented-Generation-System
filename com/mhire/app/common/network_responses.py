import time
from typing import Dict, Any
from fastapi.responses import JSONResponse

class NetworkResponse:

    def __init__(self, version=0.1):
        self.version = version

    def success_response(
        self, http_code: int, message: str,data: Dict[str, Any], resource: str, start_time: float
    ) -> JSONResponse:
        duration = round(time.time() - start_time, 2)
        # Serialize response data including any datetime objects
        return JSONResponse(
            status_code=http_code,
            content={
                "success": True,
                "message": message,
                "data": data,
                "resource": resource,
                "duration": f"{duration}s"
            }
        )

    def json_response(
        self, http_code: int, error_message: str, resource: str, start_time: float
    ) -> JSONResponse:
        duration = round(time.time() - start_time, 2)
        return JSONResponse(
            status_code=http_code,
            content={
                "code": http_code,
                "success": False,
                "message": error_message,
                "resource": resource,
                "duration": f"{duration}s"
            }
        )

class HTTPCode:
    # Success codes
    SUCCESS = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client error codes
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    GONE = 410
    PAYLOAD_TOO_LARGE = 413
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    # Server error codes
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504