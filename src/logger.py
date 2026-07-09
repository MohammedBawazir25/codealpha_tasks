import logging

from src.config import LOG_DIR

LOG_FILE = LOG_DIR / "project.log"

logging.basicConfig(

    filename=LOG_FILE,

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)