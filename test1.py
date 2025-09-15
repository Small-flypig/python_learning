"""getattr()"""
class Test(object):
    x = "nihao"
a = Test()
print(getattr(a, 'x'))  # 获取属性 x 值
print(getattr(a, 'y', 'None'))  # 获取属性 y 值不存在，但设置了默认值
# print(getattr(a, 'y'))    # AttributeError: 'Test' object has no attribute 'y'
print(a.x)  # 效果等同于上面

"""__next__()"""
string="hello"
ite=iter(string)
print(ite.__next__())#从0开始
print(ite.__iter__())#返回迭代器本身，类似指针
#print(ite(1))#错误用法

for i in string: 
    print(i)

for i in iter(string):
    print(i)

"""切片"""
lst = [10,20,30,40,50,60,70,80]
#切片为：start=2 , stop=6 , step=1
lst2 = lst[2:6:2]
print(lst2)    #[30, 40, 50, 60]

"""enumerate"""
b = [1,2,3,4,5,6]
for i  in enumerate(b):
    print(i)

x={1,2,3}
x=[x];
if isinstance(x,list):
    print("true")





import torch
import debugpy
debugpy.listen(5779)
print('wait debugger')
debugpy.wait_for_client()
print("Debugger Attached")
def compute_locations_per_level(h, w, stride):
    shifts_x = torch.arange(  #从0-》w*stride ,步长为step的数组
        0, w * stride, step=stride,# w * stride
        dtype=torch.float32
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))#建立网格
    shift_x = shift_x.reshape(-1)#每列相同  变成一维
    shift_y = shift_y.reshape(-1)#每行相同
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride
    locations = torch.stack((shift_x, shift_y), dim=1) + 32  #alex:48 // 32
    return locations


compute_locations_per_level(5,5,1,)