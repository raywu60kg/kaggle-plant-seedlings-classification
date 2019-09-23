import os
import logging
from training import training

def main():
    training()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
    os._exit(0)

