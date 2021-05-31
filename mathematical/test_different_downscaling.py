import torch
from torch import nn
from pytorch_msssim import ssim
import torchvision
from torchvision import transforms

class Tester(nn.Module):
    def __init__(self, model, test_dataloader):
        super(Tester, self).__init__()
        self.test_dataloader = test_dataloader
        self.model = model
        #self.learning_rate = learning_rate
        #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #                         lr=learning_rate)


    def test(self):
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                         lr=self.learning_rate)
        #loss_function = nn.L1Loss()

        #for epoch in range(self.number_of_epoch):
        #    i = 0
        i = 0
        mse_total = 0
        psnr_total = 0
        with torch.no_grad():
            for _, batch in enumerate(self.test_dataloader):
                low_res, high_res = batch
                # reset the gradient
                #optimizer.zero_grad()
            
                #print(low_res.shape)
                #low_res = low_res.unsqueeze(0)
                high_res = high_res.unsqueeze(0)
                trans = transforms.Resize(size=(32,32),interpolation=InterpolationMode.NEAREST)
                low_res = trans(high_res)
                # low_res = high_res.resize((32,32), Image.NEAREST)
                #print(low_res.shape)
                high_res_prediction = self.model(low_res)

                if (i==0):
                    torchvision.io.write_png(high_res[0, ...].mul(255).byte(), "high_res_downscale.png")
                    torchvision.io.write_png(high_res_prediction[0, ...].mul(255).byte(), "high_prediction_downscale.png")
                    torchvision.io.write_png(low_res[0, ...].mul(255).byte(), "low_res_downscale.png")


                #loss = loss_function(high_res_prediction, high_res)
                ssim_val = ssim(high_res, high_res_prediction, data_range=1, size_average=False)
                loss_mse = nn.MSELoss()
                mse = loss_mse(high_res,high_res_prediction)
                psnr = 10*torch.log10(1/mse)
                #loss = torch.sum((ssim_val+psnr)/2)
                mse_total = mse_total+mse
                psnr_total = psnr_total+psnr
                i = i + 1



                #loss.backward()
                #optimizer.step()
                if i % 1 == 0:    # print every 2000 mini-batches
                    print('[%5d] mse_loss: %.3f psnr_loss: %.3f ' %
                    (i , mse, psnr))
                    #running_loss = 0.0
                #i = i + 1
            mse_total = mse_total/i
            psnr_total = psnr_total/i
            print("Final Loss: ")
            print("MSE: "+str(mse_total.data))
            print("PSNR: "+str(psnr_total.data))
 