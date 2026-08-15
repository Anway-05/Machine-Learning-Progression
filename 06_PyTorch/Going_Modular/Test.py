import torch
from sklearn.metrics import accuracy_score

def testing_pipeline(model,data_loader,loss_func):
    loss,accuracy=0,0
    model.eval()
    with torch.inference_mode():
        for x,y in data_loader:
            y_logit=model(x)
            loss+=loss_func(y_logit,y).item()
            accuracy+=accuracy_score(y,y_logit.argmax(dim=1))
        loss/=len(data_loader)
        accuracy/=len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss,
            "model_acc": accuracy}