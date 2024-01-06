from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_lr 

train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

def train(model, device, train_loader, optimizer, epoch, scheduler, criterion): #adding scheduler and criterion
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)
    lrs.append(get_lr(optimizer))                           #adding extra line

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()                                        #adding extra line

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

                                                            #adding get_lr function below
    pbar.set_description(desc= f'Loss={loss.item()} LR={get_lr(optimizer)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader, criterion):            #added criterion here
    model.eval()
    test_loss = 0
    correct = 0
    # misclassified_images = []  # List to store misclassified images and labels
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim = True)  # Remove keepdim=True; now shape: [batch_size]
            correct += pred.eq(target.view_as(pred)).sum().item()        #modified this line

            # Find misclassified indices
            # misclassified_idxs = (pred != target).nonzero(as_tuple=False).squeeze()
            # for idx in misclassified_idxs:
            #     if len(misclassified_images) < 10:  # Collect only 10 images
            #         img = data[idx].cpu()
            #         actual_label = target[idx].item()
            #         predicted_label = pred[idx].item()
            #         misclassified_images.append((img, predicted_label, actual_label))
            #     else:
            #         break

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    Accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    return Accuracy