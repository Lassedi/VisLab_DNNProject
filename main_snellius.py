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
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

#dealing with input
parser = argparse.ArgumentParser(description="DNN_project")
parser.add_argument("--inp_dir", help="Directory of /Data is located", required=True)
parser.add_argument("--nepoch", type=int, required=True)
parser.add_argument("--dataset",choices=["mnist","celeba", "celeba_smallVal","places365","combi"],
                    help="Choices: mnist, celeba, celeba_smallVal, places365, combi", type=str,required=True)

args = vars(parser.parse_args())
input_dir = args["inp_dir"]
data_dir = input_dir + "/Data/celeba"
print(data_dir)

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
elif dataset_choice == "celeba" or dataset_choice == "celeba_smallVal":
    from custom.celeba_FCN import FreeConvNetwork
    sum_dim = (3,224,224)
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
    #CelebA Pytorch data set
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.CelebA("./Data", "all",target_type="identity", download=True, transform=transform) # load the whole datset


    # split identity indicies for face recognition 80% of each class training 10% each for validation and testing
    tind_list, vind_list, ttind_list = [], [], []
    for ind in range(len(torch.unique(dataset.identity))):
        inds, _ = torch.where(dataset.identity == ind + 1) # get all indices where identity matches the specific label
        train_size = int(len(inds) * 0.8)
        val_size = int((len(inds) -  train_size)/2)
        test_size = int((len(inds) - train_size)/2)
        if sum((train_size, val_size, test_size)) != len(inds):
            train_size += len(inds) - sum((train_size, val_size, test_size))

        # check if the individual classes are split nicely
        # print(train_size, val_size, test_size, sum([train_size, val_size, test_size]))
        # print(len(inds))

        tind, vind, ttind = torch.split(inds, [train_size, val_size, test_size]) # split class indices based on train/val/test split
        tind_list.append(tind)
        vind_list.append(vind)
        ttind_list.append(ttind)
        
    # conv lists to tensors
    tind_list = torch.cat(tind_list)
    vind_list = torch.cat(vind_list)
    ttind_list = torch.cat(ttind_list)

    # create train/val/test sets
    train_dataset = torch.utils.data.Subset(dataset, tind_list)
    val_dataset = torch.utils.data.Subset(dataset, vind_list[torch.randperm(len(vind_list))])
    test_dataset = torch.utils.data.Subset(dataset, ttind_list[torch.randperm(len(ttind_list))])

    #verify that all classes in val data are present in train data
    # print(torch.where(torch.isin(torch.unique(dataset.identity[vind_list]), dataset.identity[tind_list]))[0].shape[0] == 
    #       len(torch.unique(dataset.identity[vind_list])))
    # create loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 32, shuffle=True, num_workers = 18, pin_memory=True) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=18, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=18, pin_memory=True)
elif dataset_choice == "celeba_smallVal":
    #CelebA Pytorch data set without an extra test dataset and .1 validation size
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CelebA(data_dir, "all",target_type="identity", download=False, transform=transform) # load the whole datset


    # split identity indicies for face recognition 80% of each class training 10% each for validation and testing
    tind_list, vind_list = [], []
    exclusion_count = 0
    total_count = 0

    for ind in range(len(torch.unique(dataset.identity))):
        #get all images belowing to an identity
        id_ind = dataset.identity == ind + 1
        
        if total_count >= 500:
            continue
        elif torch.sum(id_ind).item() < 30: #if an identity has less than x imgs then exclude it
            exclusion_count += 1
            continue
        total_count += 1

        inds, _ = torch.where(id_ind) # get all indices where identity matches the specific label

        
        #split data into train and validation set per identity
        train_size = int(len(inds) * 0.9)
        val_size = int((len(inds) -  train_size))
        
        if sum((train_size, val_size)) != len(inds):
            train_size += len(inds) - sum((train_size, val_size))

        # check if the individual classes are split nicely
        # print(train_size, val_size, test_size, sum([train_size, val_size, test_size]))
        # print(len(inds))

        tind, vind = torch.split(inds, [train_size, val_size]) # split class indices based on train/val/test split
        tind_list.append(tind)
        vind_list.append(vind)

    # conv lists to tensors
    tind_list = torch.cat(tind_list)
    vind_list = torch.cat(vind_list)

    # create train/val sets
    train_dataset = torch.utils.data.Subset(dataset, tind_list)
    val_dataset = torch.utils.data.Subset(dataset, vind_list[torch.randperm(len(vind_list))])

    #verify that all classes in val data are present in train data
    # print(torch.where(torch.isin(torch.unique(dataset.identity[vind_list]), dataset.identity[tind_list]))[0].shape[0] ==
    #       len(torch.unique(dataset.identity[vind_list])))
    
    # create loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 32, shuffle=True, num_workers = 18, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=18, pin_memory=True)
    
    print("Excluded identities", exclusion_count)
    print("Total number of identitities: ", total_count)
    print("Remaining identities", len(torch.unique(dataset.identity)) - exclusion_count, "Original", len(torch.unique(dataset.identity)))
    print("Test Size", len(train_dataset), "Validation Size", len(val_dataset))
else:
    raise ValueError("Other datasets still missing")

#data augmentation on gpu
data_aug_all = nn.Sequential(
    transforms.Resize((224,224), antialias=True),
)
data_aug_all.to(device)
data_aug = nn.Sequential(
    transforms.RandomAffine(15,(0.2,0.2)),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.RandomHorizontalFlip(p=.5),
    #transforms.RandomVerticalFlip(p=.5),
)
data_aug.to(device)

##########
# Training
##########

#defining the loss function and optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)#), weight_decay=0.1) #weight decay = regularization to keep weights small
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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

        # Augment data
        if torch.rand([1,1]) < 0.6:
            inputs = data_aug_all(inputs)
            inputs = data_aug(inputs)
        else:
            inputs = data_aug_all(inputs)

        #remap labels to values from 1:newHighesLabelAfterExclusion
        x = torch.unique(dataset.identity[tind_list]) # get the unique identities which remain after exclusion
        y = torch.unique(torch.tensor(np.arange(0,len(x)))) # make a vector from 1:remaining identities
        x, y = x.to(device), y.to(device)

        labels_remap = torch.cat((torch.reshape(x, [1,-1]), torch.reshape(y, [1,-1]))) #cat them both together shape=[2:...]
        
        index_remap = torch.empty([1,len(labels)], dtype=torch.int32) #make index
        for lab in range(0, len(labels)):
            #print(lab, labels[lab].numpy())
            ind = torch.where(labels_remap[0,:] == labels[lab])
            #print(ind, labels_remap[0,ind])
            index_remap[:,lab] = ind[0]
        
        #print((labels == labels_remap[0,index_remap[0].numpy()]).sum())
        labels = labels_remap[1,index_remap[0].numpy()] # new labels

        

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
        
        #augment data 
        vinputs = data_aug_all(vinputs)
        
        #remap labels to values from 1:newHighesLabelAfterIdentityExclusion
        x = torch.unique(dataset.identity[vind_list]) # get the unique identities which remain after exclusion
        y = torch.unique(torch.tensor(np.arange(0,len(x)))) # make a vector from 1:remaining identities
        x, y = x.to(device), y.to(device)

        labels_remap = torch.cat((torch.reshape(x, [1,-1]), torch.reshape(y, [1,-1]))) #cat them both together shape=[2:...]
        
        index_remap = torch.empty([1,len(vlabels)], dtype=torch.int32) #make index
        for lab in range(0, len(vlabels)):
            ind = torch.where(labels_remap[0,:] == vlabels[lab])
            index_remap[:,lab] = ind[0]
        
        vlabels = labels_remap[1,index_remap[0].numpy()] # new labels

        
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
if dataset_choice == "celeba":
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
