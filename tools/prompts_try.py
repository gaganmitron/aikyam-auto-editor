import numpy as np, open_clip, torch
E=np.load("tools/eval_out/emb.npy",allow_pickle=True).item()
name="hf-hub:timm/ViT-B-16-SigLIP"
m,_,_=open_clip.create_model_and_transforms(name); tok=open_clip.get_tokenizer(name); m.eval()
P={
 "deity":["a photo of a Hindu deity idol","a decorated idol of a Hindu god in a temple shrine"],
 "priest":["a photo of a Hindu priest","a priest performing a ritual in front of a deity"],
 "devotees":["a photo of devotees praying in a temple","a crowd of people sitting on the floor singing devotional songs","a group of people sitting in a temple hall"],
 "flowers":["a photo of marigold flowers","a plate of flowers","flower garlands"],
 "lamps":["a photo of a burning lamp","a flame in a temple","a lit oil lamp"],
 "aarti":["a photo of aarti","a priest waving a burning lamp in front of a deity","fire flame in front of a deity idol"],
 "abhishekam":["a photo of abhishekam, pouring milk over an idol","a deity idol being bathed with water"],
 "decorations":["a photo of festival decorations","temple decorated with lights and garlands"],
}
flat=[(k,p) for k,ps in P.items() for p in ps]; own=[k for k,_ in flat]
with torch.no_grad(): t=m.encode_text(tok([p for _,p in flat])); t=(t/t.norm(dim=-1,keepdim=True)).numpy()
sc,b=float(m.logit_scale.exp()),float(m.logit_bias)
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print(list(P))
for n,e in E.items():
    p=1/(1+np.exp(-(sc*(e@t.T)+b))); print("\n",n)
    print(np.array([[max(p[f,i] for i,o in enumerate(own) if o==k) for k in P] for f in range(len(e))]))
