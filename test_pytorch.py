import torch

a=torch.Tensor([[1,2],[3,4]])#直接赋值初始化
print(a) 
print(a.type())

a=torch.Tensor(2,3,4)#按照形状初始化
print(a)

#定义全1tensor
a=torch.ones(2,2)
print(a)
b=torch.ones_like(a)#定义一个同a一样的全1tensor
print("b=",a)


# 对角线为1
c=torch.eye(3,3)
print(c)

b=torch.ones_like(c)#这里应该用eye_like()，但目前的结果显示原理不知道
print("b=",a)

"""随机  分布"""
a=torch.rand(2,3,4)#参数为形状,随机结果在0-1之间
print(a)
 
a5=torch.normal(mean=0.0,std=torch.rand(5))#输入均值和标准差，这里标准差随机5个，所以会有5个结果
print(a5)

#生成均匀分布
a=torch.Tensor(2,2).uniform_(-1,1)
print(a)
"""序列"""

a=torch.arange(0,10,2)#0-9 步长为2 [0,10)
print(a)

a=torch.linspace(2,10,4) #等间隔的4个数字[2,10]
print(a)
a=torch.randperm(10)#0-9顺序打乱
print(a)

#定义稀疏张量
i=torch.tensor([[0,1,2],[0,1,2]])
v=torch.tensor([1,2,3])
a=torch.sparse_coo_tensor(i,v,(4,4))#加 .to_dense()转为稠密张量
print(a)
a=torch.sparse_coo_tensor(i,v,(4,4)).to_dense()#转为稠密张量
print(a)
##  运算
## * 点乘
a=torch.tensor([1,2,3,4,8])
b=torch.tensor([3,2,1,4,5])
print(a*b)
## 字典
dict={'name':'xiaozhang','age':23,2:1}
