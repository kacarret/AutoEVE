import logging
import subprocess
#import the needed things from the pre trainers
#from pre_retrainer import train as pretrain
#import the needed things from the post trainers
#from post_retrainer import train as posttrain

#add file name to log if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

#run the pre trainer and the post trainer and check the results from early stopping for each one and ensure they are less than 1.3
pre_es_num = 100
post_es_num = 100
counter = 1
endtraining = False

ask_user = input("Do you want to do FULL training? [with pre and post] (y/n): ").lower().strip() == 'y'

if ask_user:
    subprocess.call(["python", "EVE\Trainers\pretrain.py"])
    subprocess.call(["python", "EVE\Trainers\post_trainer.py"])
else:
    logging.info("Skipping full training... continuing to split training...")

def is_odd(num):
    return num % 2 == 1

def main(pre_es_num, post_es_num, counter):
    global endtraining
    #run the pre and post trainers and check early stopping
    if is_odd(counter):
        pre_es_num = pretrain()
        counter += 1
    elif not is_odd(counter):
        post_es_num = posttrain()
        counter += 1

    #check the early stopping numbers
    if pre_es_num <= 1.3 and post_es_num <= 1.3:
        logging.info(f"Training completed successfully, pre_es_num: {pre_es_num}, post_es_num: {post_es_num}")
        endtraining = True
    elif pre_es_num <= 1.3 or post_es_num <= 1.3:
        logging.info(f"Training split, pre_es_num: {pre_es_num}, post_es_num: {post_es_num}")
    elif pre_es_num > 1.3 or post_es_num > 1.3:
        logging.info(f"Training incomplete, pre_es_num: {pre_es_num}, post_es_num: {post_es_num}")
    else:
        logging.info(f"Unkown error, pre_es_num: {pre_es_num}, post_es_num: {post_es_num}")

while not endtraining:
    main(pre_es_num=pre_es_num, post_es_num=post_es_num, counter=counter)