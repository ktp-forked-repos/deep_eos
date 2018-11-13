import logging
import sys

from . import models
from . import trainers

logger = logging.getLogger(__name__)

FORMAT = '%(asctime)-15s %(message)s'

logging.basicConfig(level=logging.WARNING, format=FORMAT, stream=sys.stdout)
logging.getLogger('deep_eos').setLevel(logging.INFO)
