import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger = logging.getLogger("app")
        start_time = time.time()

        # Process the request and get the response
        response = await call_next(request)

        # Calculate the time taken and log it
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url} - Status {response.status_code} "
            f"Processed in {process_time:.4f}s"
        )

        return response