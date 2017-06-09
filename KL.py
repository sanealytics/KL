import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

class FineTuneModel(nn.Module):                                                 
    """ Finetunes just the last layer                                           
                                                                                
    This freezes the weights of all layers except the last one.                 
    You should also look into finetuning previous layers, but slowly            
    Ideally, do this first, then unfreeze all layers and tune further with small lr
                                                                                
    Arguments:                                                                  
        original_model: Model to finetune                                       
        arch: Name of model architecture                                        
        num_classes: Number of classes to tune for                              
                                                                                
    """                                                                         
    def __init__(self, original_model, arch, num_classes):                      
        super(FineTuneModel, self).__init__()                                   
        self.archname = original_model.__class__.__name__
        
        if arch.startswith('alexnet') or arch.startswith('vgg'):                
            self.features = original_model.features                             
            self.fc = nn.Sequential(*list(original_model.classifier.children())[:-1])
            self.classifier = nn.Sequential(                                    
                nn.Linear(4096, num_classes)                                    
            )                                                                   
        elif arch.startswith('resnet') :                                        
            # Everything except the last linear layer                           
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(                                    
                nn.Linear(512, num_classes)                                     
            )                                                                   
        elif arch.startswith('inception') :                                     
            # Everything except the last linear layer                           
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(                                    
                nn.Linear(2048, num_classes)                                    
            )                                                                   
        else :                                                                  
            raise("Finetuning not supported on this architecture yet. Feel free to add")
                                                                                
        self.unfreeze(False) # Freeze weights except last layer                 
                                                                                
    def unfreeze(self, unfreeze):
        # Freeze those weights                                                  
        for p in self.features.parameters():                                    
            p.requires_grad = unfreeze                                          
        if hasattr(self, 'fc'):                                                 
            for p in self.fc.parameters():                                      
                p.requires_grad = unfreeze                                      
                                                                                
    def forward(self, x):                                                       
        f = self.features(x)                                                    
        if hasattr(self, 'fc'):                                                 
            f = f.view(f.size(0), -1)                                           
            f = self.fc(f)                                                      
        f = f.view(f.size(0), -1)                                               
        y = self.classifier(f) 
        y = F.log_softmax(y) # Needed for our loss
        return y           


num_classes = 8
resume = False

import torch.nn.parallel

#resnet18 = models.resnet18(pretrained=True)
#model = FineTuneModel(resnet18, 'resnet18', num_classes)
#alexnet = models.alexnet(pretrained=True)
#model = FineTuneModel(alexnet, 'alexnet', num_classes)
vgg11 = models.vgg11(pretrained=True)
model = FineTuneModel(vgg11, 'vgg11', num_classes)
# For alexnet and vgg
model.features = torch.nn.DataParallel(model.features)                          


if resume:
    #model = torch.load('KL_model.2.tar')

    checkpoint = torch.load('checkpoint.AlexNet.20.tar')                                
    print('Loaded epoch ', checkpoint['epoch'])                              
    model.load_state_dict(checkpoint['state_dict'])                     
    #optimizer.load_state_dict(checkpoint['optimizer']) 

print(model)

import torch.backends.cudnn as cudnn

# for alexnet and vgg
print(model.cuda())


img_dir = '/opt/styles'

from style_loader import StyleLoader
import torchvision.transforms as transforms
from PIL import ImageEnhance

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train = pd.read_csv('/home/saurabh/Downloads/product_tags_data.csv')
train.set_index('style')
classes = train.columns.tolist()[1:]
num_classes = len(classes)

# Take out bad data
tots = train.sum(1)
m = tots == 8 # All 1s
train = train[-tots.isin([8])]

# Split the data
np.random.seed(2246)
coin_toss = np.random.rand(len(train))
#train_idx = (coin_toss < 0.6)
#val_idx = np.logical_and(coin_toss >= 0.6, coin_toss < 0.8)
#test_idx = (coin_toss >= 0.8)

np.random.seed(224610)
train_idx = (coin_toss < 0.8)
val_idx = (coin_toss >= 0.8)

#batch_size=512 # for Alexnet
batch_size=32 # For vgg11
#batch_size=2 # For resnet18

train_style_loader = StyleLoader(img_dir, train[train_idx], transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.Lambda(lambda image: # Random transforms
                ImageEnhance.Contrast(image).enhance(np.random.random())),
            transforms.Lambda(lambda image: 
                ImageEnhance.Sharpness(image).enhance(np.random.random())),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

train_loader = torch.utils.data.DataLoader(
        train_style_loader,
        batch_size=batch_size, shuffle=True,
        num_workers=5, pin_memory=False)

val_style_loader = StyleLoader(img_dir, train[val_idx], transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

val_loader = torch.utils.data.DataLoader(
        val_style_loader,
        batch_size=batch_size, shuffle=True,
        num_workers=5, pin_memory=False)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Freeze weights
model.unfreeze(False)

print('send criterion to cuda')
criterion = nn.KLDivLoss(size_average=False)

criterion.cuda()

test_one_element = True
if test_one_element:
    print('Get an element to test')
    x, y = next(iter(train_loader))
    y = y.cuda(async=True)
    xvar, yvar = Variable(x, volatile=True), Variable(y, volatile=True)
    print('Sending vars to cuda')
    print('running through model')
    op = model(xvar)

    print('got back output')

    print(op)
    print('Compare to output')
    print(yvar)

    print('Get loss')
    loss = criterion(op, yvar)
    print(loss[0])

class AverageMeter(object):                                                     
    """Computes and stores the average and current value"""                     
    def __init__(self):                                                         
        self.reset()                                                            
        self.history = []
        self.prev_epoch = -1
                                                                                
    def reset(self):                                                            
        self.val = 0                                                            
        self.avg = 0                                                            
        self.sum = 0                                                            
        self.count = 1                                                          
                                                                                
    def update(self, epoch, val, n=1):                                                 
        if epoch != self.prev_epoch:
            self.history.append(self.avg)
            self.reset()
        self.val = val                                                          
        self.sum += val * n                                                     
        self.count += n                                                         
        self.avg = self.sum / self.count            
        self.prev_epoch = epoch

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    _, actuals = target.topk(maxk, 1, True, True)
    
    correct = actuals.eq(pred)
    correct = correct.t()
    
    res = []
    for k in topk:
        correct_k = correct[:k].sum()
        res.append(correct_k * 100.0 / batch_size)
    return res


print('Running the whole batch')

def checkpoint_name(prefix, state):                                             
    return "{}.{}.{}.tar".format(prefix, state['arch'], state['epoch'])         
                                                                                
def save_checkpoint(state, is_best):                                            
    filename = checkpoint_name("checkpoint", state)                        
    torch.save(state, filename)                                                 
    if is_best:                                                                 
        shutil.copyfile(filename, checkpoint_name('model_best', state))  

train_loss = AverageMeter()
val_loss = AverageMeter()
train_prec1 = AverageMeter()
train_prec2 = AverageMeter()
val_prec1 = AverageMeter()
val_prec2 = AverageMeter()
epochs = 500
base_lr = 1e-3

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=base_lr)


for epoch in range(epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch)
    for x, y in train_loader:
        y = y.cuda(async=True)
        xvar, yvar = Variable(x), Variable(y)
        #for instance, label in data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()
        #optimizer.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a Variable
        # as an integer.  For example, if the target is SPANISH, then we wrap the integer
        # 0.  The loss function then knows that the 0th element of the log probabilities is
        # the log probability corresponding to SPANISH
        #bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        #target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3. Run our forward pass.
        output = model(xvar)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        loss = criterion(output, yvar)
        train_loss.update(epoch, loss.data[0], x.size(0))
        prec1, prec2 = accuracy(output.data, y, topk=(1, 2))
        train_prec1.update(epoch, prec1, x.size(0))
        train_prec2.update(epoch, prec2, x.size(0))
                
        loss.backward()
        optimizer.step()

    # Check on validation set
    model.eval()
    for val_x, val_y in val_loader:
        val_y = val_y.cuda(async=True)
        val_x_var, val_y_var = Variable(val_x, volatile=True), Variable(val_y, volatile=True)
        val_output = model(val_x_var)
        epoch_val_loss = criterion(val_output, val_y_var)
        val_loss.update(epoch, epoch_val_loss.data[0], val_y.size(0))
        prec1, prec2 = accuracy(val_output.data, val_y, topk=(1, 2))
        val_prec1.update(epoch, prec1, val_y.size(0))
        val_prec2.update(epoch, prec2, val_y.size(0))

    print('Epoch: [{0}/{1}]\t'                                     
        'Train Loss {loss.val:.4f} ({loss.avg:.4f})\t'                      
        'Train Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'                    
        'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(              
        epoch, epochs, loss=train_loss, top1=train_prec1, top2=train_prec2))   

    print('Epoch: [{0}/{1}]\t'                                     
        'Val Loss {loss.val:.4f} ({loss.avg:.4f})\t'                      
        'Val Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'                    
        'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\n'.format(              
        epoch, epochs, loss=val_loss, top1=val_prec1, top2=val_prec2))   


    # Save checkpoint
    save_checkpoint({                                                       
        'epoch': epoch + 1,                                                 
        'arch' : model.archname,
        'state_dict': model.state_dict(),                                   
        'optimizer' : optimizer.state_dict(),                               
        'train_loss': train_loss,
        'val_loss'  : val_loss,
        'train_prec1'     : train_prec1,
        'train_prec2'     : train_prec2,
        'val_prec1'       : val_prec1,
        'val_prec2'       : val_prec2
    }, False)

    if epoch == 100:
        print('Un-freezing last layers')
        model.unfreeze(True)

print('Losses')
print(train_loss.history)
print(val_loss.history)

