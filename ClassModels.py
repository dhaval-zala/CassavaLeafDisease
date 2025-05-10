# import torch 
# import torch.nn as nn 
# import torch.nn.functional as F # import convolution functions like Relu
# import torch.optim as optim # optimzer
# import torchvision
# from torchvision import models
# import cv2 
# import numpy as np
# import pandas as pd
# import os 

# trainImagesPath = []
# for i in os.listdir("/home/dhavalsinh/Desktop/research/age_gender/dataset/train"):
#     path = "/home/dhavalsinh/Desktop/research/age_gender/dataset/train/"+i
#     for j in os.listdir(path):
#         trainImagesPath.append(path+"/"+j)
# # print(trainImagesPath[0].split("/")[-1].split(".")[0])


# BATCH_SIZE = 32
# IMG_SIZE = 112
# DEVICE = "cuda"

# class Dataset(torch.utils.data.Dataset):

#     def __init__(self, trainImagesPath):
#         self.trainImagesPath = trainImagesPath

#     def __len__(self):
#         return len(self.trainImagesPath)
    
#     def __getitem__(self, idx):

#         imagePath = self.trainImagesPath[idx]
#         label = int(imagePath.split("/")[-1].split(".")[0])
      
#         # frame = torchvision.io.read_image(imagePath)
#         # frame = torchvision.transforms.Resize(IMG_SIZE)(frame)
#         frame = cv2.imread(imagePath)
#         frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE),interpolation = cv2.INTER_LINEAR)
#         frame = np.moveaxis(frame, 2, 0)
#         frame = frame/255. 
#         frame = torch.from_numpy(frame)
#         frame = frame.to(DEVICE)
#         # label = torch.from_numpy(np.array(label))
#         # label = label.to(DEVICE)
#         return frame, label*1.0

# trainset = Dataset(trainImagesPath)
# print(f"Total examples in the trainset : {len(trainset)}")

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# print(f"Total no batches in trainloader: {len(trainloader)}")


# for array, label in trainloader:
#     break

# print(f"Shape of one batch images : {array.shape}")


# MODELS = {
# 	"vgg16": models.vgg16(pretrained=True),
# 	# "vgg19": models.vgg19(pretrained=True),
# 	# "inception": models.inception_v3(pretrained=True),
# 	# "densenet": models.densenet121(pretrained=True),
# 	# "resnet": models.resnet50(pretrained=True)
# }

# # model = MODELS["vgg16"].to(DEVICE)
# # model.eval()

# class Net(nn.Module):
#     ''' Models a simple Convolutional Neural Network'''
	
#     def __init__(self):
#         super(Net, self).__init__()
#         self.backbone = MODELS["vgg16"].to(DEVICE)
#         self.dense1 = torch.nn.Linear(in_features=400, out_features=1)
#         self.dense2 = torch.nn.Linear(in_features=10, out_features=1)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return x

# net = Net().to(DEVICE)
# print(net)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# # start = torch.cuda.Event(enable_timing=True)
# # end = torch.cuda.Event(enable_timing=True)



# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0



# # Waits for everything to finish running
# torch.cuda.synchronize()

# print('Finished Training')





import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14, ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models.convnext import CNBlock
from torchvision.models.swin_transformer import SwinTransformerBlock
# from dataset import AgeGenderDataset, AgeDataset, GenderDataset

class ModelVIT(nn.Module):

    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()

        # Init param
        self.num_classes = num_classes

        # Load the Vision transformer backbone.
        if pretrained:
            self.vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit_model = vit_b_16()
        
        # Change the head to suit to the task at hand.
        self.vit_model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=self.num_classes)
        )

        # Freeze all the layers except the last layer
        if pretrained:
            for param in self.vit_model.parameters():
                param.requires_grad = False
            for param in self.vit_model.heads.parameters():
                param.requires_grad = True

    def forward(self, x):
        
        return self.vit_model(x)
    
class MobileNetv3ClassificationModel(nn.Module):

    def __init__(self, num_classes=10, pretrained=True, keep_n_layers=1):
        super().__init__()

        # Init param
        self.num_classes = num_classes
        self.keep_n_layers = keep_n_layers

        # Load theMobilenetv3 model.
        if pretrained:
            self.mobilenetv3_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

            # Change the last Linear module to output given no. of classes.
            self.mobilenetv3_model.classifier[3] = nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)

            # Freeze all parameters initially
            for param in self.mobilenetv3_model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last N blocks if N is greater than one.
            if keep_n_layers > 0:
                for block in self.mobilenetv3_model.features[-self.keep_n_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True

            # Make sure the classifier is unfrozen as well
            for param in self.mobilenetv3_model.classifier.parameters():
                param.requires_grad = True

        else:
            self.mobilenetv3_model = mobilenet_v3_large()

            # Change the last Linear module to output given no. of classes.
            self.mobilenetv3_model.classifier[3] = nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)

    def forward(self, x):
        
        return self.mobilenetv3_model(x)
    
    
class SwinTransformerClassificationModel(nn.Module):

    def __init__(self, num_classes=10, pretrained=True, keep_n_layers=1):
        super().__init__()

        # Init param
        self.num_classes = num_classes
        self.keep_n_layers = keep_n_layers

        # Load theMobilenetv3 model.
        if pretrained:
            self.swin_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

            # Change the last Linear module to output given no. of classes.
            self.swin_model.head = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

            # Freeze all parameters initially
            for param in self.swin_model.parameters():
                param.requires_grad = False
            
            if keep_n_layers > 0:
                block_count = 0
                for layer in reversed(self.swin_model.features):
                    if isinstance(layer, nn.Sequential):
                        for sublayer in reversed(layer):
                            if isinstance(sublayer, SwinTransformerBlock):
                                block_count += 1
                                if block_count <= keep_n_layers:
                                    for param in sublayer.parameters():
                                        param.requires_grad = True

            # Make sure the classifier is unfrozen as well
            for param in self.swin_model.head.parameters():
                param.requires_grad = True

        else:
            self.swin_model = swin_t()

            # Change the last Linear module to output given no. of classes.
            self.swin_model.head = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

    def forward(self, x):
        
        return self.swin_model(x)

class ConvNextClassificationModel(nn.Module):

    def __init__(self, num_classes=10, pretrained=True, keep_n_layers=1, freeze_layers=True):
        super().__init__()

        # Init param
        self.num_classes = num_classes
        self.keep_n_layers = keep_n_layers
        self.freeze_layers = freeze_layers

        # Load the ConvNeXt model.
        if pretrained:
            self.convnext_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

            # Change the last Linear module to output given no. of classes.
            self.convnext_model.classifier[2] = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

            if self.freeze_layers:
                # Freeze all parameters initially
                for param in self.convnext_model.parameters():
                    param.requires_grad = False

            if keep_n_layers > 0:
                # Initialize a counter for CNBlocks
                cn_block_count = 0
                # Iterate in reverse order to count CNBlocks
                for layer in reversed(self.convnext_model.features):
                    if isinstance(layer, nn.Sequential):
                        for sublayer in layer:
                            if isinstance(sublayer, CNBlock):
                                cn_block_count += 1
                                if cn_block_count <= keep_n_layers:
                                    for param in sublayer.parameters():
                                        param.requires_grad = True

            # Make sure the classifier is unfrozen as well
            for param in self.convnext_model.classifier.parameters():
                param.requires_grad = True

        else:
            self.convnext_model = convnext_tiny()

            # Change the last Linear module to output given no. of classes.
            self.convnext_model.classifier[2] = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

    def forward(self, x):
        
        return self.convnext_model(x)

    
class ResNeXtClassificationModel(nn.Module):

    def __init__(self, num_classes=10, pretrained=True, keep_n_layers=1):
        super().__init__()

        # Init param
        self.num_classes = num_classes
        self.keep_n_layers = keep_n_layers

        # Load theMobilenetv3 model.
        if pretrained:
            self.resnext_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

            # Change the last Linear module to output given no. of classes.
            self.resnext_model.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

            # Freeze all parameters initially
            for param in self.resnext_model.parameters():
                param.requires_grad = False
            
            if self.keep_n_layers > 0:
                layers_to_unfreeze = ['layer4']  # Add more layer names if needed
                # Unfreeze the specified layers
                for name, child in self.resnext_model.named_children():
                    if name in layers_to_unfreeze:
                        for param in child.parameters():
                            param.requires_grad = True

                # Unfreeze the last N Bottleneck blocks in layer4
                for i, block in enumerate(reversed(self.resnext_model.layer4)):
                    if i < self.keep_n_layers:
                        for param in block.parameters():
                            param.requires_grad = True

            # Make sure the classifier is unfrozen as well
            for param in self.resnext_model.fc.parameters():
                param.requires_grad = True

        else:
            self.resnext_model = resnext50_32x4d()

            # Change the last Linear module to output given no. of classes.
            self.resnext_model.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

    def forward(self, x):
        
        return self.resnext_model(x)

# if __name__ == "__main__":

#     # VIT
#     dataset = AgeGenderDataset(batch_size=1)
#     model = ModelVIT(num_classes=10)

#     train_loader, test_loader = dataset.get_loaders()

#     for x,y in train_loader:
#         print(model(x))
#         break
#     #

#     # MobileNetv3
#     dataset = AgeDataset(batch_size=1)
#     model = MobileNetv3ClassificationModel(num_classes=5, pretrained=True, keep_n_layers=3)

#     train_loader, test_loader = dataset.get_loaders()
#     for x,y in train_loader:
#         print(model(x))
#         break

#     # SwinTransformer
#     dataset = AgeDataset(batch_size=1)
#     model = SwinTransformerClassificationModel(num_classes=5, pretrained=False, keep_n_layers=0)
#     print(f'SwinModel:{model}')

#     train_loader, test_loader = dataset.get_loaders()
#     for x,y in train_loader:
#         print(model(x))
#         break

#     # ConvNext
#     dataset = AgeDataset(batch_size=1)
#     model = ConvNextClassificationModel(num_classes=5, pretrained=False, keep_n_layers=0)

#     train_loader, test_loader = dataset.get_loaders()
#     for x,y in train_loader:
#         print(model(x))
#         break

#     # ResNext
#     dataset = AgeDataset(batch_size=1)
#     model = ResNeXtClassificationModel(num_classes=5, pretrained=True, keep_n_layers=2)
#     print(f'ResNext:{model}')

#     train_loader, test_loader = dataset.get_loaders()
#     for x,y in train_loader:
#         print(model(x))
#         break