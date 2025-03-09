import torch
#import torchvision
import nets
import triple_transforms

import os
import os.path

import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from torchvision import transforms

#Overrides 

batch_size = 20
training = 'student-post' #student-pre #'student-post #teacher
epochs = 30 #20 -- 100 -- 30

teacher_lr = 5e-4
student_pre_lr = 5e-1
student_post_lr = 5e-2

teach_lr_update = teacher_lr/(epochs-1)
student_pre_lr_update = student_pre_lr/(epochs-1)
student_post_lr_update = student_post_lr/(epochs-1)
#hardcoding for blocks atm
def make_dataset(root, is_train):
    if is_train:
        print('train')

        input = open(os.path.join(root, 'data/airsimnh/train.txt'))
        #ground_t = open(os.path.join(root, 'data/train_gt.txt'))
        depth_t = open(os.path.join(root, 'data/airsimnh/train.txt'))
        image = [(os.path.join(root, 'data/airsimnh/image/', img_name.strip('\n'))) for img_name in
                 input]
        #gt = [(os.path.join(root, 'image', img_name.strip('\n'))) for img_name in
        #         ground_t]
        depth = [(os.path.join(root, 'data/airsimnh/depth/', img_name.strip('\n'))) for img_name in
             depth_t]

        input.close()
        #ground_t.close()
        #depth_t.close()

        return [[image[i], depth[i]] for i in range(len(image))] #[[image[i], gt[i], depth[i]]for i in range(len(image))]

    else:
        print('test')

        input = open(os.path.join(root, 'data/airsimnh/test.txt'))
        #ground_t = open(os.path.join(root, 'data/test_gt.txt'))
        depth_t = open(os.path.join(root, 'data/airsimnh/test.txt'))

        image = [(os.path.join(root, 'data/airsimnh/image/', img_name.strip('\n'))) for img_name in
                 input]
        # gt = [(os.path.join(root, 'image', img_name.strip('\n'))) for img_name in
        #       ground_t]
        depth = [(os.path.join(root, 'data/airsimnh/depth/', img_name.strip('\n'))) for img_name in
                 depth_t]

        input.close()
        #ground_t.close()
        #depth_t.close()

        return [[image[i], depth[i]]  for i in range(len(image))] #[[image[i], gt[i], depth[i]]for i in range(len(image))]



class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, depth_path = self.imgs[index]
        #print(img_path)
        #print(gt_path)
        #print(depth_path)
        img = Image.open(img_path)
        # target = Image.open(gt_path)
        depth = Image.open(depth_path)
        depth = ImageOps.grayscale(depth)
        if self.triple_transform is not None:
            img, depth = self.triple_transform(img, depth) #img, target, depth = self.triple_transform(img, target, depth)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            #target = self.target_transform(target)
            depth = self.target_transform(depth)

        return img.cuda(), depth.cuda() #img, target, depth

    def __len__(self):
        return len(self.imgs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 144))
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((256, 144)),
    #triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

train_set = ImageFolder(os.getcwd(), transform=transform, target_transform=transform, triple_transform=triple_transform, is_train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = ImageFolder(os.getcwd(), transform=transform, target_transform=transform, is_train=False)
test_loader = DataLoader(test_set, batch_size=batch_size)
print(len(train_loader))
print(len(test_loader))

teacher_criterion = torch.nn.L1Loss()
eval_criterion = torch.nn.MSELoss()

if training == 'student-pre':
    student_head = nets.DGNLNet_v1_Head_Split(bn_channel=48).cuda()
    student_tail = nets.DGNLNet_v1_Tail_Split(bn_channel=48).cuda()

    teacher_model = nets.DGNLNet_v1().cuda()
    teacher_model.load_state_dict(torch.load('./Ians_Models/Teacher.pth'))
    for param in teacher_model.parameters():
        param.requires_grad = False 

    student_tail.conv6.requires_grad = False
    student_tail.conv7.requires_grad = False
    student_tail.conv8.requires_grad = False
    student_tail.conv9.requires_grad = False
    student_tail.conv10.requires_grad = False
    student_tail.depth_pred.requires_grad = False

    student_criterion = torch.nn.MSELoss() #will use for all things
    mu = 0.5

    optimizer = torch.optim.Adam((list(student_head.parameters()) + list(student_tail.decoder.parameters())), lr=student_pre_lr)
elif training == 'student-post':
    student_head = nets.DGNLNet_v1_Head_Split(bn_channel=48).cuda()
    student_tail = nets.DGNLNet_v1_Tail_Split(bn_channel=48).cuda()

    student_head.load_state_dict(torch.load('./Ians_Models/student001_Head_bn48.pth'))
    student_tail.load_state_dict(torch.load('./Ians_Models/student001_Tail_bn48.pth'))
    
    for param in student_head.parameters():
        param.requires_grad = False

    student_tail.decoder_pre.requires_grad = False
    student_tail.decoder.requires_grad = False

    student_criterion = torch.nn.MSELoss() #will use for all things

    optimizer = torch.optim.Adam((list(student_tail.conv6.parameters()) + list(student_tail.conv7.parameters()) + list(student_tail.conv8.parameters())
                                  + list(student_tail.conv9.parameters()) + list(student_tail.conv10.parameters()) + list(student_tail.depth_pred.parameters())), lr=student_post_lr)
else:
    teacher_model = nets.DGNLNet_v1().cuda()

    # #for testing
    # teacher_model.load_state_dict(torch.load('./Ians_Models/Teacher.pth'))

    lr_decay = teacher_lr/(epochs-1)
    optimizer = torch.optim.Adam(list(teacher_model.parameters()), lr=teacher_lr)


if training == 'teacher':
    for epoch in range(epochs):
        # print("===========Training:", epoch, "===========")
        # result_avg = 0
        # loss = 0

        # teacher_model.train()
        # for i, data_batch in enumerate(train_loader):
        #     if (i+1) % int(len(train_loader)/4) == 0:
        #         print('At batch:', i+1, '\t----\tPreformace:', result_avg/(i+1))

        #     input, target = data_batch
        #     output = teacher_model(input)

        #     loss = teacher_criterion(output, target)
            
        #     result_avg += loss.item()

        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()

        # print("FINAL TRAIN: ", result_avg/(len(train_loader)))


        print("===========Eval:", epoch, "===========")
        result_avg = 0
        loss = 0

        teacher_model.eval()
        for i, data_batch in enumerate(test_loader):
            if (i+1) % int(len(test_loader)/2) == 0:
                print('At batch:', i+1, '\t----\tPreformace:', result_avg/(i+1))

            input, target = data_batch
            output = teacher_model(input)

            loss = torch.mean(torch.abs(torch.sum((target-output))/(torch.numel(target)))) #teacher_criterion(output, target)
            
            result_avg += loss.item()
            
        print("FINAL EVAL: ", result_avg/(len(test_loader)))

        torch.save(teacher_model.state_dict(), 'Ians_Models/Teacher-2.pth')

        for g in optimizer.param_groups:
            g['lr'] -= teach_lr_update

elif training == 'student-pre':
    for epoch in range(epochs):
        print("===========Training:", epoch, "===========")
        result_avg = 0
        loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i+1) % int(len(train_loader)/4) == 0:
                print('At batch:', i+1, '\t----\tPreformace:', result_avg/(i+1))

            input, target = data_batch

            teacher_out = teacher_model(input)

            bn_out = student_head(input)
            student_out = student_tail(bn_out)

            loss = mu * student_criterion(student_tail.d_f5, teacher_model.d_f5)

            loss += (1-mu) * student_criterion(student_tail.d_f6, teacher_model.d_f6)/2
            loss += (1-mu) * student_criterion(student_tail.d_f7, teacher_model.d_f7)/2
            loss += (1-mu) * student_criterion(student_tail.d_f8, teacher_model.d_f8)/2
            loss += (1-mu) * student_criterion(student_tail.d_f9, teacher_model.d_f9)/2
            loss += (1-mu) * student_criterion(student_tail.d_f10, teacher_model.d_f10)/2
            loss += (1-mu) * student_criterion(student_tail.depth_pred_out, teacher_model.depth_pred_out)/2
            
            loss += teacher_criterion(student_out, target)#torch.mean(torch.abs(torch.sum((target-output))/(torch.numel(target))))#

            result_avg += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("FINAL TRAIN: ", result_avg/(len(train_loader)))

        print("===========Eval:", epoch, "===========")
        result_avg = 0
        loss = 0

        #student_head.eval()
        #student_tail.eval()
        for i, data_batch in enumerate(test_loader):
            if (i+1) % int(len(test_loader)/2) == 0:
                print('At batch:', i+1, '\t----\tPreformace:', result_avg/(i+1))

            input, target = data_batch
            output = student_head(input)
            #print(output.shape)
            output = student_tail(output)

            #print(input.shape, output.shape, target.shape)
            loss = torch.mean(torch.abs(torch.sum((target-output))/(torch.numel(target))))#teacher_criterion(output, target)
            
            result_avg += loss.item()
            
        print("FINAL EVAL: ", result_avg/(len(test_loader)))

        torch.save(student_head.state_dict(), 'Ians_Models/student001_Head_bn48.pth')
        torch.save(student_tail.state_dict(), 'Ians_Models/student001_Tail_bn48.pth')

        for g in optimizer.param_groups:
            g['lr'] -= student_pre_lr_update

elif training == 'student-post':
    for epoch in range(epochs):
        print("===========Training:", epoch, "===========")
        result_avg = 0
        loss = 0

        #student_tail.train()
        for i, data_batch in enumerate(train_loader):
            if (i+1) % int(len(train_loader)/4) == 0:
                print('At batch:', i+1, '\t----\tPreformace:', result_avg/(i+1))

            input, target = data_batch
            output = student_head(input)
            #print(output.shape, input.shape)
            output = student_tail(output)
            #print(output.shape)

            loss = teacher_criterion(output, target)#torch.mean(torch.abs(torch.sum((target-output))/(torch.numel(target))))#
            
            result_avg += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("FINAL TRAIN: ", result_avg/(len(train_loader)))


        print("===========Eval:", epoch, "===========")
        result_avg = 0
        loss = 0

        #student_tail.eval()
        for i, data_batch in enumerate(test_loader):
            if (i+1) % int(len(test_loader)/2) == 0:
                print('At batch:', i+1, '\t----\tPreformace:', result_avg/(i+1))

            input, target = data_batch
            output = student_head(input)
            #print(output.shape)
            output = student_tail(output)

            #print(input.shape, output.shape, target.shape)
            loss = torch.mean(torch.abs(torch.sum((target-output))/(torch.numel(target))))#teacher_criterion(output, target)
            
            result_avg += loss.item()
            
        print("FINAL EVAL: ", result_avg/(len(test_loader)))

        torch.save(student_head.state_dict(), 'Ians_Models/student001_Head-2_bn48.pth')
        torch.save(student_tail.state_dict(), 'Ians_Models/student001_Tail-2_bn48.pth')

        for g in optimizer.param_groups:
            g['lr'] -= student_post_lr_update