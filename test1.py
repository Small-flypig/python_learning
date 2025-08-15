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