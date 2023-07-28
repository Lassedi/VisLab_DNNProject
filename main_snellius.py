import os
from torchsummary import summary
import torchvision
from torchvision import datasets,transforms
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
# import torch.optim.lr_scheduler as lr_scheduler

#dealing with input
parser = argparse.ArgumentParser(description="DNN_project")
parser.add_argument("--inp_dir", help="Directory of /Data is located", required=True)
parser.add_argument("--nepoch", type=int, required=True)
parser.add_argument("--dataset",choices=["mnist","celeba","places365","combi"],
                    help="Choices: mnist, celeba, places365, combi", type=str,required=True)

args = vars(parser.parse_args())
input_dir = args["inp_dir"]
data_dir = input_dir + "/Data"

EPOCHS = args["nepoch"]
dataset_choice = args["dataset"]

#######
# Model
#######

# from custom.mnist_net import FreeConvNetwork
# from custom.model3x_HighDimOutCh import FreeConvNetwork
if dataset_choice == "mnist":
    from custom.mnist_net import FreeConvNetwork
    sum_dim = (3,28,28)
    model = FreeConvNetwork()
elif dataset_choice == "celeba":
    from custom.celeba_net import FreeConvNetwork
    sum_dim = (3,218,178)
    model = FreeConvNetwork()
elif dataset_choice == "places365":
    ...
elif dataset_choice == "combi":
    ...

# Testing & summarizeing the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ndevices = torch.cuda.device_count()
ncpu = os.cpu_count()
print(device, ndevices, ncpu)



if ndevices > 1:
    model = nn.DataParallel(model)
model.to(device)
summary(model, sum_dim)


######
# Data
######

# Mnist for setting up snellius
if dataset_choice == "mnist":
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(data_dir, train = True, download=False, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=72, num_workers =  18, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=72, num_workers =  18, pin_memory=True)

    #report split sizes
    print("Training set size: {}".format(len(train_dataset)))
    print("Testing set size: {}".format(len(test_dataset)))

#CelebA Pytorch data set
elif dataset_choice == "celeba":
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CelebA(data_dir, "train",target_type="identity", download=False, transform=transform)
    test_dataset = datasets.CelebA(data_dir, "test", target_type="identity", download=False, transform=transform)
    val_dataset = datasets.CelebA(data_dir, "valid", target_type="identity",download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 32, num_workers =  18, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 200, num_workers =  18, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 200, num_workers =  18, pin_memory=True)

    #report split sizes
    print("Training set size: {}".format(len(train_dataset)))
    print("Testing set size: {}".format(len(test_dataset)))
    print("Validation set size: {}".format(len(val_dataset)))
else:
    raise ValueError("Other datasets still missing")


##########
# Training
##########

#defining the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)#), weight_decay=0.1) #weight decay = regularization to keep weights small
loss_fn = nn.CrossEntropyLoss()
# scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01,total_iters=100)

def train_one_epoch(epoch_index, tb_writer):

    running_loss = 0.
    last_loss = 0.
    correct = 0.
    total = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # moving the training data batch by batch to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # [a,b,c,d,f,e,g] = model(inputs) #debugging for NAN output
        
        # Make predictions for this batch
        outputs = model(inputs)
        # check data parallelism
        # print("Outise: Input size=", inputs.size(),"Output size=", outputs.size())

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Adjust the Learning rate
        # scheduler.step()
        # print(optimizer.param_groups[0]["lr"])
        
        # Gather data and report
        running_loss += loss.item()

        # calculating accuracy
        _, predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).sum()
        total += labels.size(0)

        if i % 100 == 99:
            last_loss = running_loss / 100 # avg loss over x batches
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.


    accuracy = correct/total * 100

    return last_loss, accuracy

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter((input_dir + '/runs/fashion_trainer_{}'.format(timestamp)))
epoch_number = 0

#visualize training data & corresponding labels & add them to summary
dataiter = iter(train_loader)
images, labels = next(dataiter)

# print("Labels: ",labels.numpy()[0:4])

images = images[:4,:,:,:]
img_grid = torchvision.utils.make_grid(images)

# plt.imshow(img_grid.permute(1,2,0))
# plt.show()

writer.add_image("train_data_firstFour; Labels: {}".format(
    labels.numpy()[0:4]), img_grid)



best_vloss = 1_000_000.


if not os.path.isdir((input_dir + "/model")):
    os.makedirs((input_dir + "/model"))

for epoch in range(EPOCHS):
    total = 0
    correct = 0
    # print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, tacc = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
         
        # moving the test data batch by batch to the GPU
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        
        # validating the model
        voutputs = model(vinputs)

        # calculating accuracy
        _, vpredicted = torch.max(voutputs.data,1)

        total += vlabels.size(0)
        correct += (vpredicted == vlabels).sum()
        

        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.detach().item() # wihtout .detach I get an out of memory error on GPU
    
    # log average accuracy
    vacc = (correct/total) * 100
    writer.add_scalars("Average Accuracy", 
                      {"Training" : tacc, "Validation" : vacc},
                      epoch_number + 1)
    

    # Log the running loss averaged per batch
    # for both training and validation
    avg_vloss = running_vloss / (i + 1)

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # print('LOSS train {} valid {}'.format(round(avg_loss,2), round(avg_vloss,2)))
    # print("Accuracy_training: {} Accuracy_validation {}".format(round(tacc.item(),2), round(vacc.item(),2)))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = (input_dir + '/model/model_{}_{}'.format(timestamp, epoch_number))
        if ndevices > 1:
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

    epoch_number += 1

############
# Validation (unseen data)
############
if not dataset_choice == "mnist":
    vcorrect = 0
    vtotal = 0

    with torch.no_grad():
        model.train(False)
        for i, data in enumerate(val_loader):
            vdata, vlabels = data
            vdata, vlabels = vdata.to(device), vlabels.to(device)

            voutput = model(vdata)

            vloss = loss_fn(voutput, vlabels)

            # validation accuracy unseen data
            _, vpred = torch.max(voutput.data,1)
            vcorrect =+ (vpred == vlabels).sum()
            vtotal =+ vlabels.size(0)

        vacc = vcorrect / vtotal * 100

        print("Validation Accuracy on unseen data is {}%".format(vacc))
