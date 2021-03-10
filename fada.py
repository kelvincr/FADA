from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def train_stage_1(args, model, device, train_loader, optimizer, epoch):
    
for i in range(num_epochs1):
    total = 0.0
    correct = 0.0
    for data,labels in dataloaders_1['train']:
        data=data.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        pred = classifier(encoder(data))
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss=loss_fn(pred,labels)
        loss.backward()
        optimizer.step()
    accuracy2 = correct / total
    scheduler.step()
    acc = 0
    total = 0.0
    correct = 0.0
    for data, labels in dataloaders_1['val']:
        with torch.no_grad():
            data=data.to(device)
            labels=labels.to(device)
            outputs=classifier(encoder(data))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #acc += (torch.max(y_test_pred,1)[1]==labels).float().mean().item()
    #accuracy=round(acc / float(len(dataloaders_1['val'])), 3)
    accuracy = correct / total
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))
    if(accuracy>best_acc):
        best_acc = accuracy
        torch.save(encoder.state_dict(),'result/encoder_fada_extra.pth')
        torch.save(classifier.state_dict(), 'result/classifier_fada_extra.pth')
        #classifier.save()
    print("Phase 1 ---- Epoch {} accuracy: {} / {}".format((i+1), accuracy, accuracy2))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FADA Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    data_dir = '/../dataset/herbarium'
    data_dir2 = '/../dataset/photo'
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    classifier = models.ClassifierPro()
    encoder = tmodels.resnet50(pretrained=True)
    encoder.fc = nn.Sequential()

        print("||||| Stage 1 |||||")
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(ssnet.parameters())+list(genusnet.parameters())+list(familynet.parameters()), lr = 0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_stage_1(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
