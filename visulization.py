import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ResNetCifar10

decay_path = './Adamacclearn_decay.checkpoint.pth.tar'
hybrid_path = './Adamacchybrid.checkpoint.pth.tar'
batch_path = './Adamaccbatch_increase.checkpoint.pth.tar'

if os.path.isfile(decay_path):
    checkpoint = torch.load(decay_path)
    decay_history = checkpoint['history']
    decay_para_list=checkpoint['para_num']
    print(decay_para_list)
    print("Decay checkpoint successfully loaded from '{}'!".format(decay_path))

if os.path.isfile(hybrid_path):
    checkpoint = torch.load(hybrid_path)
    hybrid_history = checkpoint['history']
    hybrid_para_list = checkpoint['para_num']
    print("Hybrid checkpoint successfully loaded from '{}'!".format(hybrid_path))

if os.path.isfile(batch_path):
    checkpoint = torch.load(batch_path)
    batch_history = checkpoint['history']
    batch_para_list = checkpoint['para_num']
    print("Batch checkpoint successfully loaded from '{}'!".format(batch_path))

plt.figure(figsize=(10,8))
plt.title('Adam_plots')

plt.subplot(2, 2, 1)
decay_train_loss = np.array(decay_history['train_losses'])
hybrid_train_loss = np.array(hybrid_history['train_losses'])
batch_train_loss = np.array(batch_history['train_losses'])

plt.semilogy(np.arange(decay_train_loss.shape[0]), decay_train_loss, label='Decaying learning rate')
plt.semilogy(np.arange(hybrid_train_loss.shape[0]), hybrid_train_loss, label='Hybrid')
plt.semilogy(np.arange(batch_train_loss.shape[0]), batch_train_loss, label='Increasing batch size')
plt.ylabel('Training cross-entropy')
plt.xlabel('Number of epochs')

plt.legend()

plt.subplot(2, 2, 2)
decay_para_num = np.cumsum(np.array(decay_para_list))
hybrid_para_num = np.cumsum(np.array(hybrid_para_list))
batch_para_num = np.cumsum(np.array(batch_para_list))

plt.semilogy(decay_para_num, decay_train_loss, label='Decaying learning rate')
plt.semilogy(hybrid_para_num, hybrid_train_loss, label='Hybrid')
plt.semilogy(batch_para_num, batch_train_loss, label='Increasing batch size')
plt.xlabel('Number of parameters')

plt.legend()

plt.subplot(2, 2, 3)
decay_test_accuracies = np.array(decay_history['test_accuracies'])
hybrid_test_accuracies = np.array(hybrid_history['test_accuracies'])
batch_test_accuracies = np.array(batch_history['test_accuracies'])

plt.plot(np.arange(decay_test_accuracies.shape[0]), decay_test_accuracies, label='Decaying learning rate')
plt.plot(np.arange(hybrid_test_accuracies.shape[0]), hybrid_test_accuracies, label='Hybrid')
plt.plot(np.arange(batch_test_accuracies.shape[0]), batch_test_accuracies, label='Increasing batch size')
plt.ylabel('Test set accuracy')
plt.xlabel('Number of epochs')
plt.legend()

plt.subplot(2, 2, 4)

plt.semilogy(decay_para_num[0:99], decay_test_accuracies, label='Decaying learning rate')
plt.semilogy(hybrid_para_num[0:99], hybrid_test_accuracies, label='Hybrid')
plt.semilogy(batch_para_num[0:99], batch_test_accuracies, label='Increasing batch size')
plt.xlabel('Number of parameters')

plt.legend()

plt.savefig('Adam_plots.png')
plt.show()
