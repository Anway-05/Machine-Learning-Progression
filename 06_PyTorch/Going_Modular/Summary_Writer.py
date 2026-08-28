import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name,model_name,extra=None):

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir=os.path.join("runs",timestamp,experiment_name,model_name,extra)
    else:
        log_dir=os.path.join("runs",timestamp,experiment_name,model_name)

    print(f"Created SummaryWriter at: {log_dir}")

    return SummaryWriter(log_dir=log_dir)

    