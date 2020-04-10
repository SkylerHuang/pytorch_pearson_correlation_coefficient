"""
Created on 04 10 2020
@author: SkylerHuang
"""
import torch

def compute_pearson(input):
    """
    input: tensor [b,c,n] or [b,c,h,w]
    return: pearson correlation coefficient [c,c]
    """
    batch, channel,_ = input.shape
    input = input.view(batch, channel,-1)
    mean = torch.mean(input,dim=0).unsqueeze(0)
    cov = torch.matmul(input-mean,(input-mean).permute(0,2,1))
    diag = torch.sum(torch.eye(channel).unsqueeze(0) * cov,dim=2).view(batch,channel,-1)
    stddev = torch.sqrt(torch.matmul(diag,diag.permute(0,2,1)))
    pearson = torch.div(cov,stddev)
    return pearson

if __name__ == '__main__':
    a = torch.randn(16,512,196)
    print(compute_pearson(a))
