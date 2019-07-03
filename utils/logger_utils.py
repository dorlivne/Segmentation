import logging
import logging.handlers
import os
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


def get_logger(name: str):
    dirname = os.path.dirname(name)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # By default, logs all messages
    fh = logging.FileHandler("{0}.log".format(name))
    fh.setLevel(logging.DEBUG)
    fh_format = logging.Formatter('%(asctime)s  - %(levelname)-8s - %(message)s')
    fh.setFormatter(fh_format)
    logger.addHandler(fh)
    return logger


def info(string, logger):
    print(string)
    logger.info(string)
