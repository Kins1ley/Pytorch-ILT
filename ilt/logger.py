import logging

def get_logger(name, file_name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        # logger.propagate = 0
        console = logging.FileHandler(file_name)
        logger.addHandler(console)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        console.setFormatter(formatter)
        # logger.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    return logger


if __name__ == "__main__":
    file_name = "test.txt"
    logger = get_logger(__name__, file_name)
    # logger.setLevel(logging.DEBUG)
    logger.info("info")
    logger.debug("debug")