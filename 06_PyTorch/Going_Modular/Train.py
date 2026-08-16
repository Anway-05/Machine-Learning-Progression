from timeit import default_timer as timer
from tqdm.auto import tqdm
from .Training_Pipeline import training_pipeline

def train(model,
          train_dataloader,
          optimizer,
          loss_func,
          epochs):

    start_time=timer()

    for epoch in tqdm(range(epochs)):
        print(f"-----Epoch {epoch+1}-----")
        training_pipeline(model,train_dataloader,loss_func,optimizer)

    end_time=timer()

    print(f"Time taken for training is {end_time-start_time}.")