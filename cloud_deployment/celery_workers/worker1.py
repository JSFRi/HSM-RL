from celery import Celery
import os

# Celery configuration
CELERY_BROKER_URL = 'amqp://test:123456@192.168.1.12:5672/'
CELERY_RESULT_BACKEND = 'rpc://'
# Initialize Celery
celery = Celery('worker', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

@celery.task()
def add_nums(a, b):
   return a + b

@celery.task()
def fast_move_file(i):
   os.system('mv /home/appuser/fast_tier/%d.txt /home/appuser/nfs/'%i)

@celery.task()
def fast_get_file(i):
   os.syetem('mv /home/appuser/nfs/* /home/appuser/fast_tier/')
