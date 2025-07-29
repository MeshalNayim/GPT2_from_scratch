from generate.generate import generate_and_print_sample
from train_and_test.loss_evaluate import calc_loss_batch,evaluate_model


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen+= input_batch.numel() # total number of elements seen upto to current batch
            global_step+=1
            
            if global_step%eval_freq==0: # display status every _ number of batches
                train_loss, val_loss = evaluate_model(train_loader,val_loader,model,eval_freq,device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step}): Train Loss {train_loss:.3f} Val Loss {val_loss:.3f}")
            
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
                
            