import torch
from torch import nn
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

class Trainer(nn.Module):
    def __init__(self, number_of_epoch, train_dataloader, valid_dataloader, model, learning_rate):
        super(Trainer, self).__init__()
        self.number_of_epoch = number_of_epoch
        self.train_dataloader = train_dataloader
        self.valid_dataloader =valid_dataloader
        self.model = model
        self.learning_rate = learning_rate
        #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #                         lr=learning_rate)


    def train(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=self.learning_rate)
        loss_function = nn.L1Loss()
        if torch.cuda.is_available():
            loss_function = loss_function.cuda()

        mse_list = []
        psnr_list = []
        l1_list = []
        train_loss = []

        for epoch in range(self.number_of_epoch):
            i = 0
            loss_total = 0
            for _, batch in enumerate(self.train_dataloader):
                low_res, high_res = batch
                if torch.cuda.is_available():
                    low_res = low_res.cuda()
                if torch.cuda.is_available():
                    high_res = high_res.cuda()
                # reset the gradient
                optimizer.zero_grad()
                # forward pass through the model
                # print(low_res.shape)
                #print(high_res)
                high_res_prediction = self.model(low_res)
                #print(high_res_prediction)
                # compute the loss
                loss = loss_function(high_res_prediction, high_res)
                # backpropagation
                loss.backward()
                # update the model parameters
                optimizer.step()
                # log the metrics, images, etc
                #running_loss += loss.item()
                #if i % 10 == 0:    # print every 2000 mini-batches
                #        print('[%d, %5d] loss: %.3f' %
                #        (epoch + 1, i + 1, loss))
                        #running_loss = 0.0
                i = i + 1
                #print(i)
                #print('-----------------------')
                loss_total = loss_total + loss
            loss_total = loss_total/i
            train_loss.append(loss_total)
            print('[Epoch %d Training] Loss: %.3f' %
                        (epoch + 1, loss_total))
            if ((epoch+1)%100 == 0):
                print("Saving model ...")
                torch.save(self.model.state_dict(), str(epoch)+ '_'+str(self.learning_rate)+'.pth')
            if (epoch%10 == 0):
                plt.cla()
                plt.plot(train_loss)
                plt.xlabel('epoch')
                plt.ylabel('Train Loss')
                plt.savefig("train_loss"+ '_'+str(self.learning_rate)+".png", dpi=120)

            mse_total = 0
            psnr_total = 0
            l1_total = 0
            i = 0
            with torch.no_grad():
                for _, batch in enumerate(self.valid_dataloader):
                    low_res, high_res = batch
                    if torch.cuda.is_available():
                        low_res = low_res.cuda()
                    if torch.cuda.is_available():
                        high_res = high_res.cuda()
               
                    high_res_prediction = self.model(low_res)

                    ssim_val = ssim(high_res, high_res_prediction, data_range=1, size_average=False)
                    loss_mse = nn.MSELoss()
                    if torch.cuda.is_available():
                        loss_mse = loss_mse.cuda()
                    mse = loss_mse(high_res,high_res_prediction)
                    psnr = 10*torch.log10(1/mse)

                    mse_total = mse_total + mse
                    psnr_total = psnr_total + psnr

                    l1_loss = nn.L1Loss()
                    if torch.cuda.is_available():
                        l1_loss = l1_loss.cuda()
                    l1 = l1_loss(high_res,high_res_prediction)
                    l1_total = l1_total + l1

                    i = i + 1

                mse_total = mse_total/i
                psnr_total = psnr_total/i
                l1_total = l1_total/i
                mse_list.append(mse_total)
                psnr_list.append(psnr_total)
                l1_list.append(l1_total)
                print('[Epoch %d Evaluation] L1 Loss: %.3f MSE Loss: %.3f PSNR Loss: %.3f' %
                            (epoch + 1, l1_total, mse_total, psnr_total))
                if (epoch%10 == 0):
                    #print("Saving model ...")
                    #torch.save(self.model.state_dict(), str(epoch)+'.pth')
                    print("Plotting loss ...")
                    plt.cla()
                    plt.plot(l1_list)
                    plt.xlabel('epoch')
                    plt.ylabel('L1 Loss')
                    plt.savefig("l1_loss"+ '_'+str(self.learning_rate)+".png", dpi=120)
                    plt.cla()
                    plt.plot(mse_list)
                    plt.xlabel('epoch')
                    plt.ylabel('MSE Loss')
                    plt.savefig("mse_loss"+ '_'+str(self.learning_rate)+".png", dpi=120)
                    plt.cla()
                    plt.plot(psnr_list)
                    plt.xlabel('epoch')
                    plt.ylabel('PSNR Loss')
                    plt.savefig("psnr_loss"+ '_'+str(self.learning_rate)+".png", dpi=120)




