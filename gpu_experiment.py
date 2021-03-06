import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ResNetCifar10
import torchvision.transforms as transforms

model_state_path="./Adamacclearn_decay.checkpoint.pth.tar"
def save_checkpoint(state, filename=model_state_path):
    torch.save(state, filename)

def plot_history(history):

    plt.subplot(1, 2, 1)
    train_loss = np.array(history['train_losses'])
    plt.semilogy(np.arange(train_loss.shape[0]), train_loss, label='Training cross-entropy')
    plt.legend()

    plt.subplot(1, 2, 2)
    test_accuracies = np.array(history['test_accuracies'])
    plt.plot(np.arange(test_accuracies.shape[0]), test_accuracies, label='Test set accuracy', color='g')
    plt.legend()
    plt.savefig('plots_Adam_acc_learn_decay.png')
    plt.show()


def param_update(epoch, scenario, optimizer, trainloader, update_every, factor, train_data):
    if scenario == 0:
        if not ((epoch + 1) % update_every):
            optimizer.param_groups[0]['lr'] /= factor

    elif scenario ==1:
        if (epoch + 1) >= 2 * update_every:
            optimizer.param_groups[0]['lr'] /= factor
        elif not ((epoch + 1) % update_every):
            new_batch=trainloader.batch_size
            new_batch *= factor
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=new_batch,
                                                      shuffle=True, num_workers=1)

    elif scenario == 2:
        if not ((epoch + 1) % update_every):
            new_batch = int(trainloader.batch_size * factor)
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=new_batch,
                                                      shuffle=True, num_workers=1)

    return optimizer, trainloader


def train(epoch, scenario, optimizer, trainloader,net,weight_decay,
          acceptable_batch_size,update_every,scenarios,factor,criterion,stat_every,history,train_data, para_list):
    net.train()

    optimizer, trainloader = param_update(epoch, scenario, optimizer, trainloader, update_every, factor, train_data)
    print("Epoch: {0}      batch size: {1}      learning rate: {2}      weight decay: {3}".format(epoch + 1,
                                                                                                  trainloader.batch_size,
                                                                                                  optimizer.param_groups[
                                                                                                      0]['lr'],
                                                                                     weight_decay))
    total = 0
    errors = 0
    train_loss = 0
    processed = 0
    s=0
    print(trainloader)
    for data in trainloader:
        s=s+1
        processed += int(data[1].size()[0])
        optimizer.zero_grad()  # zero the parameter gradients

        # ---Here is the tirck to artificially increase the batch size without additional RAM available:---
        #acc_loops = ((int(data[1].size()[0]) - 1) // acceptable_batch_size) + 1
        #print(acc_loops)
        #if (epoch + 1) >= update_every and scenario != scenarios['learn_decay']:
            #acc_loops = factor * ((epoch + 1) // update_every)  # how many parts we want to split our big batch
        #batch_max_size = ((int(data[1].size()[0]) - 1) // acceptable_batch_size) + 1  # how big the parts should be
        #for loop in range(acc_loops):
        inputs = data[0]  # get partial images
        labels = data[1]  # get partial labels
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()  # wrap them in Variable
        outputs = net.forward(inputs)  # forward + backward + optimize
        loss = criterion(outputs, labels).cuda()
        loss.backward()

        optimizer.step()
        # -------------------------------------------------------------------------------------------------
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        errors += (predicted != labels.data).sum()
        train_loss += loss.item()
        if (processed // stat_every):  # print loss statistics every `stat_every` processed elements
            print('\tLoss: %.3f' % (loss.item()))
            processed -= stat_every

    print(s)
    correct = total - errors
    train_accuracy = 100.0 * correct / total
    history['train_losses'].append(train_loss/s)
    print('\tAccuracy for training epoch: %.2f%%' % (train_accuracy))


    para_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(trainable_num)
    para_list.append(int(para_num*s/1000))

    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scenario': scenario,
        'trainloader': trainloader,
        'history': history,
        'para_num':para_list,
    })
    print("\tCheckpoint saved!")
    return optimizer, trainloader


def test(epoch, testloader,net,criterion,history,classes):
    net.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    test_loss = 0

    for data in testloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        loss = criterion(outputs, labels).cuda()

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        if not ((epoch + 1) % 5):
            c = (predicted == labels.data).squeeze()
            for i in range(int(data[1].size()[0])):
                label = labels[i]
                class_correct[label.item()] += c[i]
                class_total[label.item()] += 1

    test_accuracy = 100.0 * correct / total
    history['test_accuracies'].append(test_accuracy)
    print('\tAccuracy of the network on the 10000 test images: %.2f%%' % (test_accuracy))
    if not ((epoch + 1) % 5):
        for i in range(10):
            print('\t\tFor %5s : %2d %%' % (
                classes[i], 100.0 * class_correct[i] / class_total[i]))



def main():
    net = ResNetCifar10()
    net.cuda()
    N = 50000
    batch_size = 512
    weight_decay = 0.0005
    momentum = 0.9
    lr = 0.1
    batch_scaling_coef = batch_size / lr  # for future use
    stat_every = 10000
    epoch = 0
    max_epochs = 200
    acceptable_batch_size = 1024

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr, momentum, weight_decay, nesterov = True)
    optimizer = optim.Adam(net.parameters(), lr=lr/100, betas=(momentum,0.99), weight_decay=weight_decay)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=0, nesterov=True)
    history = {'train_losses': [], 'test_accuracies': []}
    para_list=[]
    scenarios = {'learn_decay': 0, 'hybrid': 1, 'batch_increase': 2}
    #original_update_every = 60
    update_every = 30
    factor = 2
    #update_asif_original_factor = 4
    #update_factor = 1 / (1.0 / update_asif_original_factor) ** (1.0 / (original_update_every / update_every))

    transform = transforms.Compose([
                 transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])

    transform_test = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   shuffle=False, num_workers=1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=acceptable_batch_size,
                                                  shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    scenario = scenarios['learn_decay']
    epochs_num = 100
    for e in range(200 - epoch):
        if epoch == epochs_num:
            break
        optimizer, trainloader = train(epoch, scenario, optimizer, trainloader,net, weight_decay, acceptable_batch_size, update_every, scenarios, factor ,criterion, stat_every, history, train_data, para_list)
        test(epoch, testloader,net,criterion,history, classes)
        epoch += 1

    plot_history(history)


if __name__ == "__main__":
    main()
