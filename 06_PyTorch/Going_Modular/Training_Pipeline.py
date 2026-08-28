from sklearn.metrics import accuracy_score

def training_pipeline(model,data_loader,loss_func,optimizer):
    training_loss,training_accuracy=0,0
    model.train()
    for x,y in data_loader:
        y_logit=model(x)
        loss=loss_func(y_logit,y)
        training_loss+=loss.item()
        training_accuracy+=accuracy_score(y,y_logit.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_loss/=len(data_loader)
    training_accuracy/=len(data_loader)
    print(f"Training loss is {training_loss}")
    print(f"Training accuracy is {training_accuracy}")
    return training_loss,training_accuracy