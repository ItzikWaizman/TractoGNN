import os
import argparse
import logging
from config import Parameters
from trainers.trainer import TractoGNNTrainer

if __name__ == '__main__':
     # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", required=False, help='Whether to start training phase')
    parser.add_argument('--track', action="store_true", required=False, help='Whether to start inference phase')
    args = parser.parse_args()

    # Set logger
    abs_path = os.path.abspath(__file__)
    dname = os.path.dirname(abs_path)
    log_path = os.path.join(dname, '.log')
    
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # Get parameters 
    params = Parameters().params

    #if args.train:
    if True:
        logger.info("Staring training setups")
        trainer = TractoGNNTrainer(logger=logger, params=params)
        train_stats, val_stats = trainer.train()

    #if args.tarck:
    #    logger.error("Tracking is not supported yet.")
    #    pass