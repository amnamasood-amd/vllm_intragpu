import torch
from torch import nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #x = self.linear1(x)
        #x = self.activation(x)
        #x = self.linear2(x)
        #x = self.softmax(x)
        #return x
        return self.linear2(self.activation(self.linear1(x)))


@torch.inference_mode
def main():
    torch.cuda.set_device(0)
    myModel=TinyModel().to("cuda:0")
    x=torch.ones((1,100),device="cuda:0",dtype=torch.float32)
    
    out=myModel(x)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out=myModel(x)

    y=g.replay()
    z=myModel.softmax(y)
    print(z)

if __name__ == "__main__":
    main()


