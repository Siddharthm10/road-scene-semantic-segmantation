import os
import logging

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training.log")
        logging.basicConfig(
            filename=self.log_file,
            filemode='a',
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

    def info(self, message):
        print(message)
        self.logger.info(message)
