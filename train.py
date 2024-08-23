import argparse
import time
import os
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import sys
from net_models import *
from SGD_Dropback import *



def build_model(args):
    print('==> Building model..')

    no_of_class = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'svhn': 10,
        'imagenet': 1000
    }[args.dataset]
    net = {       
            'ResNet18': ResNet18,
            'ResNet50': ResNet50,
            'ResNet34': ResNet34,
            'ResNet101': ResNet101,
            'PreActResNet18': PreActResNet18,
            'GoogLeNet' : GoogLeNet,
            'DenseNet121' : DenseNet121,
            'ResNeXt29_2x64d' : ResNeXt29_2x64d,
            'MobileNet' : MobileNet,
            'MobileNetV2' : MobileNetV2,
            'DPN92' : DPN92,
            'ShuffleNetG2': ShuffleNetG2,
            'SENet18' : SENet18,
            'ShuffleNetV2': ShuffleNetV2,
            'EfficientNetB0' : EfficientNetB0,
            'RegNetX_200MF' : RegNetX_200MF,
            'SimpleDLA':SimpleDLA  ,
            'SwinTransformer':torchvision.models.swin_v2_t,
            'convNext':torchvision.models.convnext_tiny,
            'convNextv2_tiny':torchvision.models.convnext_tiny
        }[args.model](num_classes=no_of_class)
    return net


def build_dataset(args):
    num_workers=32
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.dataset == 'cifar10':
            dataset = torchvision.datasets.CIFAR10
            TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
            TRAIN_STD = (0.2023, 0.1994, 0.2010)
        else:
            dataset = torchvision.datasets.CIFAR100
            TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.Resize((args.image_res,args.image_res)),
            transforms.RandomCrop(args.image_res, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((args.image_res,args.image_res)),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])

        trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=num_workers)

        testset = dataset(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch, shuffle=False, num_workers=num_workers)
        
    if args.dataset == 'imagenet':
        path = args.imagenet_path
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.ImageFolder(traindir,transform =  transforms.Compose([
                                                        
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(15),
                                                                    transforms.Resize((args.image_res,args.image_res)),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=num_workers, pin_memory=True,shuffle=True)
        val_dateset = torchvision.datasets.ImageFolder(valdir,transform =  transforms.Compose([transforms.Resize((args.image_res,args.image_res)),
                                                        transforms.ToTensor(),
                                                        normalize]))
        val_loader = torch.utils.data.DataLoader(val_dateset,
                                                    batch_size=args.batch, shuffle=True,
                                                    num_workers=num_workers, pin_memory=False)
        return train_loader , val_loader

    return trainloader, testloader


def train_model(model, device, train_loader, optimizer, criterion):
    model.train(True)
    train_loss = 0
    train_total = 0
    train_correct = 0

    start_t = time.process_time()
    drop_prob = []
    for batch_idx, (data, target) in enumerate(train_loader, start=0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 100), target)
        train_loss += loss.item()
        scores, predictions = torch.max(output.view(-1, 100).data, 1)
        train_total += target.size(0)
        train_correct += int(sum(predictions == target))
        loss.backward()
        optimizer.step()

    end_t = time.process_time() - start_t
    acc = round((train_correct / train_total) * 100, 2)
    if drop_prob == []:
        drop_prob = 0

    train_data={'train_loss': round(train_loss / train_total,6),'train_accuracy': acc, 'end_t':end_t }

    return train_data


def test_model(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    test_total = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output  = model(data)
            test_loss += criterion(output.view(-1, 100), target).item()
            scores, predictions = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += int(sum(predictions == target))

    acc = round((test_correct / test_total) * 100, 2)
    test_data={'test_loss': round(test_loss / test_total,6),'test_acc': acc}
    return test_data

# Fixed parameters
class Config():
    def __init__(self,dataset="cifar10", model="resnet18", epochs=164,batch_size=512,dropback_rate_final=0.2,
                 dropback_rate_channels=0.3, skip_init_layers=30,dropback_rate_init=0.01, 
                 lr_dropback=0.03, gamma_dropback=0.05, image_res=224,imagenet_path=""):
        
        self.batch                  = batch_size
        self.weight_decay           = 5e-4
        self.momentum               = 0.9
        
        # gamma
        self.gamma_baseline         = 0.05
        self.gamma_dropback         = gamma_dropback
        
        # lr
        self.lr_baseline            = 0.03
        self.lr_dropback            = lr_dropback
        
        # dropback rate
        self.dropback_rate_init     = dropback_rate_init
        self.dropback_rate_final    = dropback_rate_final
        self.dropback_rate_channels = dropback_rate_channels  # prob. to drop an entire channel 
        
        # skip layers
        self.skip_init_layers       = skip_init_layers  
        
        self.model                  = model
        self.dataset                = dataset
        self.imagenet_path          = imagenet_path
        self.epochs                 = epochs
        self.milestones             = [round(self.epochs * 0.3), round(self.epochs * 0.6), round(self.epochs * 0.85)]
        self.image_res              = image_res
        self.expr_name              = '{}-{}-{}-{}-[{}]-{}'.format( self.dataset, 
                                                          self.model, 
                                                          self.epochs,
                                                          self.dropback_rate_final,
                                                          self.dropback_rate_channels,
                                                          self.skip_init_layers,
                                                          time.strftime("%Y_%m_%d_%H_%M_%S"))
        
def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropback_net = build_model(args)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dropback_net = nn.DataParallel(dropback_net)


    dropback_net = dropback_net.to(device)
    
    trainloader, testloader = build_dataset(args)

    dropback_criterion = nn.CrossEntropyLoss()

    dropback_optimizer      = SGD_Dropback(dropback_net, args.lr_dropback, momentum=args.momentum,
                                      weight_decay=args.weight_decay, nesterov=True,dropback=True,skip_init_layers=args.skip_init_layers)
  
    
    dropback_lr_scheduler   = optim.lr_scheduler.MultiStepLR(dropback_optimizer, milestones=args.milestones, gamma=args.gamma_dropback)
    dropback_rate_schedular = StepLR_Dropback(dropback_optimizer,args.milestones,
                                              dropback_rate_init     =args.dropback_rate_init,
                                              dropback_rate_final    =args.dropback_rate_final,
                                              dropback_rate_channels =args.dropback_rate_channels)

    dropback_total_time = 0
    dropback_max_accuracy = 0
    baseline_max_accuracy = 0

    print("Device: " + str(device))
    print("Total model parameters:{:,}".format(sum(param.numel() for param in dropback_net.parameters())))
    print(f"Experiment Name: ",args.expr_name)
    print(f"┏━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓")
    print(f"┃     ┃       ┃       ┃Drop ┃Drop ┃Drop ┃ Drop┃     ┃")
    print(f"┃Epoch┃Max_Acc┃Acc    ┃Rate ┃Param┃Loss ┃ L.R ┃Time ┃")
    print(f"┠─────╂───────╂───────╂─────╂─────╂─────╂─────╂─────┃")

    for epoch in range(args.epochs):

        ###dropback training
        train_info_dropback      = train_model(dropback_net, device, trainloader, dropback_optimizer, dropback_criterion)
        end_t                    = train_info_dropback['end_t']     
        dropback_total_time     += end_t
        test_info_dropback       = test_model(dropback_net, device, testloader, dropback_criterion)
        dropback_lr_scheduler.step()
        dropback_rate_schedular.step()

        ###baseline training
        train_info_baseline= train_info_dropback
        test_info_baseline=test_info_dropback

        # get max accuracy and current lr
        if test_info_dropback['test_acc'] > dropback_max_accuracy:
            dropback_max_accuracy = test_info_dropback['test_acc']
            dropback_max_epoch    = epoch

        if test_info_baseline['test_acc'] > baseline_max_accuracy:
            baseline_max_accuracy = test_info_baseline['test_acc']
            baseline_max_epoch    = epoch

        # Get the actual drop rate and lr for this epoch
        lr_dropback         = dropback_optimizer.param_groups[0]["lr"]
        lr_baseline         = lr_dropback 
        drop_rate_per_epoch = dropback_optimizer._get_dropback_prob_mean_()
        drop_rate_optimizer = dropback_optimizer._get_dropback_rate_()
        baseline_total_time=1
        #### print
        print("┃{:4d} ┃{:7.2f}┃{:7.2f}┃{:1.3f}┃{:1.3f}┃{:1.3f}┃{:1.3f}┃{:02.0f}:{:02.0f}┃".format(
            epoch,dropback_max_accuracy,test_info_dropback['test_acc'],
            round(drop_rate_optimizer,3), round(drop_rate_per_epoch,3), test_info_dropback['test_loss'],
            lr_dropback,lr_baseline,
            round((baseline_total_time/(epoch+1))//60,0),round((baseline_total_time/(epoch+1))%60,0)),flush=True)

    print("Total time is  {:02.0f}:{:02.0f}".format(round((dropback_total_time+baseline_total_time)//60,0),round((dropback_total_time+baseline_total_time)%60,0)))
    print('Finished Training')

#19612968
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--model', default='ResNet34', type=str, help='model')
    parser.add_argument('--dataset', type=str, default="cifar100", help='dataset', choices=['cifar10', 'cifar100', 'mnist', 'svhn','imagenet'])
    parser.add_argument('--imagenet_path', default="", type=str, help='path to imagenet if used')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate gamma')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--dropback_rate_final', default=0.3, type=float, help='dropback_rate_final')
    parser.add_argument('--dropback_rate_channels', default=0.3, type=float, help='dropback_rate_channels')
    parser.add_argument('--dropback_rate_init', default=0.03, type=float, help='dropback_rate_init')
    parser.add_argument('--skip_init_layers', default=3, type=int, help='skip_init_layers')
    parser.add_argument('--image_res', default=32, type=int, help='image resolution')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = get_parser()
    args=Config(dropback_rate_final=args.dropback_rate_final, dropback_rate_init=args.dropback_rate_init, dropback_rate_channels=args.dropback_rate_channels, skip_init_layers=args.skip_init_layers,
                    lr_dropback=args.lr, gamma_dropback=args.lr_gamma,batch_size=args.batch_size,
                    epochs=args.epochs,dataset=args.dataset,model=args.model, image_res=args.image_res, imagenet_path=args.imagenet_path)
    main(args)


