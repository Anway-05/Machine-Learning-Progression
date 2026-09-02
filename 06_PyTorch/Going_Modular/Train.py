import torch
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from tqdm.auto import tqdm
from .Training_Pipeline import training_pipeline
from .Test import testing_pipeline

def train(model,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_func,
          epochs,
          writer=None):

    results={
        "training_loss":[],
        "training_accuracy":[],
        "test_loss":[],
        "test_accuracy":[]
    }
    
    start_time=timer()

    if writer:
        writer.add_graph(
            model=model,
            input_to_model=torch.randn(1,3,224,224)
        )

    for epoch in tqdm(range(epochs)):
        print(f"-----Epoch {epoch+1}-----")
        training_loss,training_accuracy=training_pipeline(model,train_dataloader,loss_func,optimizer)

        test_results=testing_pipeline(model,test_dataloader,loss_func)

        results["training_loss"].append(training_loss)
        results["training_accuracy"].append(training_accuracy)
        results["test_loss"].append(test_results["model_loss"])
        results["test_accuracy"].append(test_results["model_acc"])

        if writer:
            writer.add_scalars(
                "Loss",
                {
                    "train":training_loss,
                    "test":test_results["model_loss"]
                },
                epoch
            )
            writer.add_scalars(
                "Accuracy",
                {
                    "train":training_accuracy,
                    "test":test_results["model_acc"]
                },
                epoch
            )

    end_time=timer()

    print(f"Time taken for training is {end_time-start_time}.")

    if writer:
        writer.close()

    return results