import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader,model,num_batches,device):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i,(input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
    
    return total_loss/num_batches

def evaluate_model(train_loader,val_loader,model,eval_freq,device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,eval_freq,device)
        val_loss = calc_loss_loader(val_loader,model,eval_freq,device)
    model.train()
    return train_loss,val_loss
    

    