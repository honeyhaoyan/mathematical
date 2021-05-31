import torch
from dataset import SRDataset
import torchvision
from train import Trainer
from model import BasicSRModel
from test_different_downscaling import Tester
import sys

# You create an object from the dataset class:
# For example: train_dataset = SRDataset(data_path)
# Assuming the train_dataset object is properly initialized
# and that the __getitem__ returns a tuple
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(’Using {} device’.format(device))

def main(learning_rate):
    
    
    test_dataset = SRDataset("data/eval")
    test_dataloader = torch.utils.data.dataloader.DataLoader(
                            test_dataset,
                            batch_size=2,
                            shuffle=True,
                            num_workers=2,
                            drop_last=True,
                            pin_memory=True)
    model_path = '199'+ '_'+str(learning_rate)+'.pth'
    model = BasicSRModel(3)
    model.load_state_dict(torch.load(model_path))
    tester = Tester(model,test_dataset)
    tester.test()

if __name__ == "__main__":
    #print("Current Learning Rate is: " + str(float(sys.argv[1])))
    main(1e-4)
