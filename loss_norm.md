# 总结论文中遇到的损失函数

## focal loss
Focal Loss处理正负极样本不平衡
![alt text](image-81.png)
![alt text](image-80.png)

## GIoU loss
IoU的一个改进版本，使用了两个框的并矩形框，效果会更好一些
https://blog.csdn.net/weixin_45377629/article/details/124915296


## 空间一致性损失 Lspa
- [总结论文中遇到的损失函数](#总结论文中遇到的损失函数)
  - [focal loss](#focal-loss)
  - [GIoU loss](#giou-loss)
  - [空间一致性损失 Lspa](#空间一致性损失-lspa)
  - [Ltv — Illumination Smoothness Loss 照明平滑度损失](#ltv--illumination-smoothness-loss-照明平滑度损失)
- [归一化方法](#归一化方法)
  - [softmax](#softmax)
- [标准化方法](#标准化方法)
- [激活函数](#激活函数)
  - [SiLU 激活函数(Swish 激活函数)](#silu-激活函数swish-激活函数)
  - [gelu](#gelu)


![alt text](image-82.png)

## Ltv — Illumination Smoothness Loss 照明平滑度损失
![alt text](image-83.png)





# 归一化方法
## softmax 
输出值总和为1  范围0-1 完美符合多分类问题
![alt text](image-84.png)
![alt text](image-88.png)


# 标准化方法


# 激活函数

## SiLU 激活函数(Swish 激活函数)
x*softmax

![alt text](image-85.png)
![alt text](image-86.png)
比较
![alt text](image-87.png)

## gelu
![alt text](image-89.png)
![alt text](image-90.png)