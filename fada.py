from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models as tmodels
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
import os
import os.path
import pathlib
from PIL import Image
import models
import data_loader
import transformations


def train_stage_1(config, model, encoder, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    correct, total = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(encoder(data))
        _, predicted = torch.max(pred.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss=loss_fn(pred,target)
        loss.backward()
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            wandb.log({"train batch loss": loss.item()})
            if config.dry_run:
                print('dry-run')
                break
    accuracy = 100. * correct / total
    return accuracy

def test_stage_1(model, encoder, device, test_loader):
    model.eval()
    correct, total = 0, 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            outputs= model(encoder(data))
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100. * correct / total
    print('Accuracy test images: %d %%' % (accuracy))
    return accuracy

def stage_1(config, device, image_datasets, classifier, encoder, ssnet, genusnet, familynet, train_kwargs, test_kwargs):
    train_loader = torch.utils.data.DataLoader(image_datasets['train'],**train_kwargs)
    test_loader = torch.utils.data.DataLoader(image_datasets['val'], **test_kwargs)

    opt_params = list(encoder.parameters())+list(classifier.parameters())+list(ssnet.parameters())+list(genusnet.parameters())+list(familynet.parameters())
    optimizer = torch.optim.Adam(opt_params, lr = config.learning_rate)

    loss_fn=torch.nn.CrossEntropyLoss()
    wandb.watch(classifier)
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=config.gamma)
    val_accuracy, best_acc = 0.0, 0.0
    for epoch in tqdm(range(config.epochs)):
        train_accuracy = train_stage_1(config, classifier, encoder, device, train_loader, loss_fn, optimizer, epoch)
        val_accuracy = test_stage_1(classifier, encoder, device, test_loader)
        scheduler.step()
        wandb.log({"val accuracy": val_accuracy, 'Step': epoch})
        wandb.log({"train accuracy": train_accuracy, 'Step': epoch})
        if(val_accuracy>best_acc):
            best_acc = val_accuracy
            torch.save(encoder.state_dict(),os.path.join(config.result_path, 'encoder_fada_extra.pth'))
            torch.save(classifier.state_dict(), os.path.join(config.result_path, 'classifier_fada_extra.pth'))
        wandb.log({"epoch": epoch})



def train_stage_2(config, model, encoder, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    correct, total = 0, 0
    confusion_matrix = torch.zeros(4, 4)
    for batch_idx, (X1, X2, idx1, idx2, ground_truths, op1, op2, same, img0_ss, label0_ss, img1_ss, label1_ss) in enumerate(train_loader):
        X1,X2=X1.to(device),X2.to(device)
        same,ground_truths = same.to(device), ground_truths.to(device)
        optimizer.zero_grad()
        X_cat = torch.cat([encoder(X1),encoder(X2)],1)
        y_pred = model(X_cat.detach())
        acc += (torch.max(y_pred,1)[1]==ground_truths).float().mean().item()
        pred = torch.max(y_pred,1)[1]
        for t, p in zip(ground_truths.view(-1), pred.view(-1)):
            total += 1.0
            if(t == p):
                correct += 1
            confusion_matrix[t.long(), p.long()] += 1
        ground_truths = ground_truths.view(-1)
        loss=loss_fn(y_pred,ground_truths)
        loss.backward()
        if batch_idx % config.log_interval == 0:
            wandb.log({"discriminator train batch loss": loss.item()})
            if config.dry_run:
                print('dry-run')
                break
    accuracy = 100. * correct / total
    return accuracy

def stage_2(config, device, dataset, discriminator, encoder, ssnet, genusnet, familynet, train_kwargs, test_kwargs):

    data_loader = DataLoader(dataset, **train_kwargs)

    opt_params = list(discriminator.parameters())+list(discriminator_genus.parameters())+list(discriminator_family.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=config.learning_rate)

    loss_fn=torch.nn.CrossEntropyLoss()
    wandb.watch(discriminator)
    #scheduler = MultiStepLR(optimizer, milestones=[30], gamma=config.gamma)
    val_accuracy, best_acc = 0.0, 0.0
    for epoch in tqdm(range(config.epochs * 2)):
        train_accuracy = train_stage_2(config, discriminator, encoder, device, train_loader, loss_fn, optimizer, epoch)
        optimizer.step()
        wandb.log({"discriminator accuracy": val_accuracy, 'Step': epoch})
        if(val_accuracy>best_acc):
            best_acc = val_accuracy
            torch.save(discriminator.state_dict(),os.path.join(config.result_path, 'discriminator_fada_extra.pth'))
        wandb.log({"epoch": epoch})

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FADA')
    parser.add_argument('--batch-size', type=int, default=60, metavar='N',
                        help='input batch size for training (default: 60)')
    parser.add_argument('--test-batch-size', type=int, default=60, metavar='N',
                        help='input batch size for testing (default: 60)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
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
    parser.add_argument('--result-path', type=pathlib.Path, default="../result/",
                        help='Path for Saving the current Model')
    parser.add_argument('--stage', type=int, default=0, metavar='N',
                        help='stage of model 0 all, 1 classifier, 2 discriminator, 3 final')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    config = wandb.config
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.gamma = args.gamma
    config.epochs = args.epochs
    config.test_batch_size = args.test_batch_size
    config.log_interval = args.log_interval
    config.image_size = 224
    config.dry_run = args.dry_run
    config.num_workers = 6
    config.stage = args.stage

    wandb.init(project='FADA', config=config, tags=list("baseline"))

    result_path = os.path.join(args.result_path, wandb.run.name)
    config.result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': config.batch_size }
    test_kwargs = {'batch_size': config.test_batch_size }
    if use_cuda:
        cuda_kwargs = {'num_workers': config.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transformations.TileHerb(),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val': transforms.Compose([
        transformations.CropField(),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val_photo': transforms.Compose([
        transformations.CropField(),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ])
}
    
    data_dir = '/dev/shm/dataset'
    herbarium = os.path.join(data_dir, 'herbarium')
    photo = os.path.join(data_dir, 'photo_split')

    classifier = models.ClassifierPro().to(device)
    encoder = tmodels.resnet50(pretrained=True).to(device)
    encoder.fc = nn.Sequential()
    ssnet = models.TaxonNet(64).to(device)
    genusnet = models.TaxonNet(510).to(device)
    familynet = models.TaxonNet(151).to(device)

    discriminator = models.DCDPro().to(device)
    discriminator_genus = models.DCDPro().to(device)
    discriminator_family = models.DCDPro().to(device)

    image_datasets = {x: datasets.ImageFolder(os.path.join(herbarium, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
    
    base_mapping = image_datasets['train'].class_to_idx
    class_name_to_id = image_datasets['train'].class_to_idx
    id_to_class_name = {v: k for k, v in class_name_to_id.items()}
    
    # siamese_dataset = data_loader.FADADatasetSSTaxons(herbarium,
    #                                 photo,
    #                                 'train',
    #                                 image_datasets['train'].class_to_idx,
    #                                 class_name_to_id,
    #                                 id_to_class_name,
    #                                 config.image_size
    #                                 )

    if(config.stage == 1):
        stage_1(config, device, image_datasets, classifier, encoder, ssnet, genusnet, familynet, train_kwargs, test_kwargs)
    # elif(config.stage == 2):
    #     encoder.load_state_dict(torch.load('best/encoder_fada_extra.pth'))
    #     classifier.load_state_dict(torch.load('best/classifier_fada_extra.pth'))
    #     stage_2(config, device, siamese_dataset, classifier, encoder, ssnet, genusnet, familynet, train_kwargs, test_kwargs)
    

if __name__ == '__main__':
    main()