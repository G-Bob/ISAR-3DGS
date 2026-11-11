import torch,sys
ck=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
print(list(ck.keys())); print({k:tuple(v.shape) for k,v in ck["model"].items()})
z=ck["model"]["support_logits"]; q=torch.sigmoid(z/0.35)
print("N",len(q),"q030",(q>=0.3).sum().item(),"q050",(q>=0.5).sum().item())
print("quant",[round(x,4) for x in torch.quantile(q,torch.tensor([.01,.05,.1,.25,.33,.4,.5,.75,.9,.99])).tolist()])
print("hist",torch.histc(q,bins=10,min=0,max=1).int().tolist())
