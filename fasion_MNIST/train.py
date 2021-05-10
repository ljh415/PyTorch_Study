import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch._C import device
import torch.nn as nn
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from dataset import FashionDataset
from utils import *
from model import *

class Train():
    def __init__(self, vis_mode, save_fig, num_workers):
        self.vis_mode = vis_mode
        self.save_fig = save_fig
        self.num_workers = num_workers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        '''
        Two ways to load the Fashion MNIST dataset.
        1. Load csv and then inherite Pytorch Dataset class
        2. Use Pytorch module torchvision.datasets. It has many popular datasets like MNIST,
            Fasion MNIST, CIFAR10 etc.
        '''

        # 1. Using Dataset class
        self.train_csv = pd.read_csv("/media/jaeho/HDD/datasets/fashion_mnist/fashion-mnist_train.csv")
        self.test_csv = pd.read_csv("/media/jaeho/HDD/datasets/fashion_mnist/fashion-mnist_test.csv")

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.loss_list = []
        self.iteration_list = []
        self.accuracy_list = []

        self.prediction_list = []
        self.labels_list = []

    def prepare(self) :
        self.train_set = FashionDataset(self.train_csv, transform=self.transform)
        self.test_set = FashionDataset(self.test_csv, transform=self.transform)

        self.train_loader = DataLoader(self.train_set, batch_size=100, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.train_set, batch_size=100, num_workers=self.num_workers)

        self.model = FashionCNN()
        self.model.to(self.device)

    def check_data(self):
        # check data
        a = next(iter(self.train_loader))
        print(a[0].size())
        print(len(self.train_set))

        image,label = next(iter(self.train_set))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'{label}, {output_label(label)}')
        plt.show()
    
    def check_batch_data(self, grid_flag=False):

        # check batch data
        demo_loader = torch.utils.data.DataLoader(self.train_set, batch_size=10)
        batch = next(iter(demo_loader))
        images, labels = batch
        print(type(images), type(labels))
        print(images.shape, labels.shape)

        if grid_flag:
            # Visualize data
            grid = torchvision.utils.make_grid(images, nrow=10)
            plt.figure(figsize=(15, 20))
            plt.imshow(np.transpose(grid, (1,2,0)))
            print("labels: ", end=" ")
            for i, label in enumerate(labels):
                print(output_label(label), end=", ")
            plt.show()

    def train_model(self):

        error = nn.CrossEntropyLoss()

        lr = 0.001
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        num_epochs = 5
        count = 0

        for epoch in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                train = Variable(images.view(100, 1, 28, 28))
                labels = Variable(labels)

                outputs = self.model(train)
                loss = error(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count += 1

                if not (count % 50):
                    total = 0
                    correct = 0

                    for images, labels in self.test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        self.labels_list.append(labels)

                        test = Variable(images.view(100, 1, 28, 28))

                        outputs = self.model(test)

                        predictions = torch.max(outputs, 1)[1]
                        self.prediction_list.append(predictions)
                        correct += (predictions == labels).sum()

                        total += len(labels)

                    accuracy = correct * 100 / total
                    self.loss_list.append(loss.data.cpu())
                    self.iteration_list.append(count)
                    self.accuracy_list.append(accuracy.cpu())

                if not (count % 500):
                    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

    def visualize(self, mode=None, save_fig=False):
        if mode == 'loss':
            plt.plot(self.iteration_list, self.loss_list)
            plt.xlabel("num of Iteration")
            plt.ylabel("Loss")
            plt.title("Iterations vs Loss")
            plt.show()
            if save_fig:
                plt.savefig('./loss.png')

        elif mode == 'acc':
            plt.plot(self.iteration_list, self.accuracy_list)
            plt.xlabel("num of Iteration")
            plt.ylabel("Accuracy")
            plt.title("Iteration vs Accuracy")
            plt.show()
            if save_fig:
                plt.savefig('./acc.png')
        
        elif mode == 'all':
            plt.subplot(1,2,1)
            plt.plot(self.iteration_list, self.loss_list)
            plt.xlabel("num of Iteration")
            plt.ylabel("Loss")
            plt.title("Iterations vs Loss")

            plt.subplot(1,2,2)
            plt.plot(self.iteration_list, self.accuracy_list)
            plt.xlabel("num of Iteration")
            plt.ylabel("Accuracy")
            plt.title("Iteration vs Accuracy")
            if save_fig:
                print(f'saved_path : /home/jaeho/Documents/fasion_MNIST/loss_acc.png')
                plt.savefig('./loss_acc.png')
            plt.show()

        else :
            raise ValueError("invalid mode")

    def evaluation(self):
        class_correct = [0. for _ in range(10)]
        total_correct = [0. for _ in range(10)]

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                test = Variable(images)
                outputs = self.model(test)
                predicted = torch.max(outputs, 1)[1]
                c = (predicted == labels).squeeze()

                for i in range(100):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    total_correct[label] += 1
        
        for i in range(10):
            print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i]*100/total_correct[i]))

    def process(self, eval):
        print('initalizing...\n')
        self.prepare()

        print('training...')
        self.train_model()
        print('\n\n')

        self.visualize(mode=self.vis_mode, save_fig=self.save_fig)

        if eval:
            print('evaluating...')
            self.evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fashion MNIST Classifier")
    parser.add_argument('--vis', type=str, default=None)
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    t = Train(args.vis, args.save_fig, args.num_workers)
    t.process(args.eval)