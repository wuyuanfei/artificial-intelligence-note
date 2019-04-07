### Python装饰器

- **@property**：广泛应用在类的定义中，可以让调用者写出**简短**的代码，同时保证**对参数进行必要的检查**，这样，程序运行时就减少了出错的可能性，[参考](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001386820062641f3bcc60a4b164f8d91df476445697b9e000)。

```python
class Student(object):
	# 把一个getter方法变成属性，只需要加上@property就可以了，此时，@property本身又创建了另一个装
    # 饰器@score.setter，负责把一个setter方法变成属性赋值
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
    
    @property
    def name(self):
        return self.name
```

> 只读属性，只定义getter()方法，不定义setter()方法就是一个只读属性
>
> score可读写，而name仅可读。



python之"_xxx"，"__xxx"，"__xxx__"

- 使用_one_underline来表示该方法或属性是私有的，不属于API；
- 当创建一个用于python调用或一些特殊情况时，使用__two_underline__；
- 使用__just_to_underlines，来避免子类的重写！