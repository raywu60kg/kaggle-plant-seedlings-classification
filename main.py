import os
import logging
import kaggle-plant-seedlings-classification.training

def main():
    training()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
    os._exit(0)

