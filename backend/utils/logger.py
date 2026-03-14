import logging
import sys
import json
from loguru import logger
from backend.config import settings

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logger():
    # Remove default handler
    logger.remove()
    
    # Define JSON format for production/monitoring
    def json_sink(message):
        record = message.record
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "extra": record["extra"]
        }
        if record["exception"]:
            log_entry["exception"] = str(record["exception"])
            
        print(json.dumps(log_entry))

    # Standard Console Format for dev, JSON for prod
    if settings.APP_ENV == "production":
        logger.add(json_sink, level=settings.LOG_LEVEL)
    else:
        logger.add(
            sys.stderr, 
            level=settings.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logger

# Initialize immediately
logger = setup_logger()
ouse_logger = logger # Alias 
