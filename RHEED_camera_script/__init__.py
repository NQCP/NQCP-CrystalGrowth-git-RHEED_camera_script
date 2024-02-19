import logging
import RHEED_camera_script._version


__version__ = RHEED_camera_script._version.__version__



logger = logging.getLogger(__name__)
logger.info(f'Imported RHEED_camera_scriptversion: {__version__}')
