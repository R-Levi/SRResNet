import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import configargparse
from V_0.model import SRResNet
from V_0.DataSet import DataProcess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file = True,help='config file path')
    parser.add_argument("--img_path", default="set5\\Urban100\\",help="dataset path")
    #超参数
    parser.add_argument("--batchsize", default=32, help = "size of batch")
    parser.add_argument("--epoches", default=30, help = "nums_epoch")
    parser.add_argument("--lr",default=0.001, help="learning rate")
    parser.add_argument("--history",default=True,help="is record history data")

    return parser

def train():

    parser = config_parser()
    args = parser.parse_args()
    img_path = args.img_path
    dataset = DataProcess(imgPath=img_path)
    net = SRResNet().to(device)
    train_data = DataLoader(dataset,batch_size=args.batchsize)
    trainer = optim.Adam(net.parameters(),lr=args.lr)
    loss_Fun = nn.MSELoss().to(device)

    print('[INFO] Training on ',device)
    for epoch in range(args.epoches):
        net.train()
        total_loss = 0.
        history = []
        for i,(cropImg,sourceImg) in tqdm(enumerate(train_data)):
            cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)
            trainer.zero_grad()

            pre = net(cropImg)
            loss = loss_Fun(pre,sourceImg)
            loss.backward()
            trainer.step()
            total_loss +=loss.item()

        average_loss = total_loss/(i+1)
        if args.history:
            history.append(average_loss)
        total_loss = 0.

        print('[INFO] Epoch %d loss: %.3f' % (epoch + 1, average_loss))
    """
    torch.save(model.state_dict(), '\parameter.pkl')
    # 加载
    model = SRResNet()
    model.load_state_dict(torch.load('\parameter.pkl'))
    """
    torch.save(net,"model_V0.pkl")
    print('[INFO] Finished Training')

def pre_imshow(path,net):
    """展示结果"""
    preTransform = transforms.Compose([transforms.ToTensor()])
    orign_img = cv2.imread(path)
    img = preTransform(orign_img).unsqueeze(0)

    # 使用cpu就行
    net.cpu()
    SR_img = net(img)[0, :, :, :]
    SR_img = SR_img.cpu().detach().numpy()  # 转为numpy
    SR_img = SR_img.transpose((1, 2, 0))  # 切换形状
    SR_img = np.clip(SR_img, 0, 1)  # 修正图片
    img = np.uint8(SR_img * 255)

    img = cv2.resize(src=img,dsize=orign_img.shape[:-1],interpolation=cv2.INTER_LINEAR)

    plt.title("orign")
    plt.imshow(orign_img)
    plt.show()
    plt.title("SR_Res")
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    #train()
    net = torch.load("model_V0.pkl")
    pre_imshow(path="",net=net)

