import os

def wandb_save_online(is_online):
    if not is_online:
        os.environ['WANDB_SILENT']='true'
        os.environ['WANDB_MODE'] = 'dryrun'