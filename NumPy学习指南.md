# NumPy 学习指南（可运行示例版）

> 这份笔记改成了更适合学习的形式：每一块都尽量做到“看概念 -> 跑代码 -> 对照结果 -> 理解原因”。

---

## 0. 怎么使用这份笔记

推荐学习方式：

1. 打开 Python 或 Jupyter。
2. 每次只复制一个代码块运行。
3. 先猜结果，再运行，再对照下面的输出说明。

基础导入：

```python
import numpy as np
```

---

## 1. 数组创建

一句话理解：NumPy 的核心对象是 `ndarray`，它比 Python 列表更适合数值计算。

### 1.1 从列表创建数组

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
arr_2d = np.array([[1, 2], [3, 4]])

print(arr)
print(arr_2d)
```

预期输出：

```python
[1 2 3 4]
[[1 2]
 [3 4]]
```

学习提示：一维数组像“数轴上的点列”，二维数组像“表格”。

### 1.2 常用创建函数

```python
import numpy as np

print("zeros:\n", np.zeros((2, 3)))
print("ones:\n", np.ones((2, 3)))
print("eye:\n", np.eye(3))
print("arange:", np.arange(0, 10, 2))
print("linspace:", np.linspace(0, 1, 5))
```

预期输出：

```python
zeros:
 [[0. 0. 0.]
  [0. 0. 0.]]
ones:
 [[1. 1. 1.]
  [1. 1. 1.]]
eye:
 [[1. 0. 0.]
  [0. 1. 0.]
  [0. 0. 1.]]
arange: [0 2 4 6 8]
linspace: [0.   0.25 0.5  0.75 1.  ]
```

重点区别：

- `np.arange(start, stop, step)`：已知步长。
- `np.linspace(start, stop, num)`：已知点数。
- 画图时通常更推荐 `linspace`。

### 1.3 随机数组

```python
import numpy as np

np.random.seed(42)

print(np.random.randint(30, 100, (2, 3)))
print(np.random.rand(2, 3))
print(np.random.randn(2, 3))
```

学习提示：`seed()` 用来固定随机结果，方便复现和调试。

---

## 2. 数组属性

一句话理解：先看清数组长什么样，再做运算。

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print("shape =", arr.shape)
print("ndim =", arr.ndim)
print("size =", arr.size)
print("dtype =", arr.dtype)
print("itemsize =", arr.itemsize)
```

预期输出：

```python
shape = (2, 3)
ndim = 2
size = 6
dtype = int64
itemsize = 8
```

说明：`dtype` 和 `itemsize` 会因系统环境略有不同，比如可能是 `int32`。

---

## 3. 索引与切片

一句话理解：NumPy 的二维索引非常像“表格取行列”。

```python
import numpy as np

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

print("arr[0, 1] =", arr[0, 1])
print("arr[:, 0] =", arr[:, 0])
print("arr[1:3, :] =\n", arr[1:3, :])
print("arr[:, 1:3] =\n", arr[:, 1:3])
```

预期输出：

```python
arr[0, 1] = 20
arr[:, 0] = [10 40 70]
arr[1:3, :] =
 [[40 50 60]
  [70 80 90]]
arr[:, 1:3] =
 [[20 30]
  [50 60]
  [80 90]]
```

记忆方法：

- `:` 放在前面，表示“选所有行”。
- `:` 放在后面，表示“选所有列”。

---

## 4. 数组运算

一句话理解：NumPy 默认是“整个数组一起算”，这就是向量化。

### 4.1 逐元素运算

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print("a + b =", a + b)
print("a * b =", a * b)
print("a ** 2 =", a ** 2)
print("np.sqrt(a) =", np.sqrt(a))
```

预期输出：

```python
a + b = [11 22 33]
a * b = [10 40 90]
a ** 2 = [1 4 9]
np.sqrt(a) = [1.         1.41421356 1.73205081]
```

提醒：`a * b` 是逐元素乘法，不是矩阵乘法。

### 4.2 广播机制

一句话理解：形状能对齐时，小数组会被“自动扩展”。

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

print(a + b)
```

预期输出：

```python
[[11 22 33]
 [14 25 36]]
```

理解方式：`b` 看起来像被复制成了两行，再和 `a` 对应相加。

### 4.3 矩阵运算

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("A @ B =\n", A @ B)
print("A.T =\n", A.T)
print("det(A) =", np.linalg.det(A))
```

预期输出：

```python
A @ B =
 [[19 22]
  [43 50]]
A.T =
 [[1 3]
  [2 4]]
det(A) = -2.0000000000000004
```

说明：浮点运算有时会出现很小误差，这是正常的。

---

## 5. 两个特别常用的 NumPy 函数

这一节专门改成“学习卡片”形式，方便你反复跑。

### 5.1 `np.where()`：按条件选值或替换值

一句话理解：满足条件选前者，不满足条件选后者。

```python
import numpy as np

scores = np.array([58, 76, 91, 43, 88])
result = np.where(scores >= 60, "及格", "不及格")

print(result)
```

预期输出：

```python
['不及格' '及格' '及格' '不及格' '及格']
```

再看一个数值替换版本：

```python
import numpy as np

arr = np.array([10, 55, 80, 30, 95])
new_arr = np.where(arr > 60, 1, 0)

print(new_arr)
```

预期输出：

```python
[0 0 1 0 1]
```

固定模板：

```python
np.where(条件, 条件为真时取什么, 条件为假时取什么)
```

### 5.2 `np.argmax()` / `np.argmin()`：找最大值或最小值的位置

一句话理解：这两个函数返回的不是值，而是“索引位置”。

```python
import numpy as np

arr = np.array([15, 99, 42, 77])

print("最大值索引 =", np.argmax(arr))
print("最小值索引 =", np.argmin(arr))
print("最大值 =", arr[np.argmax(arr)])
print("最小值 =", arr[np.argmin(arr)])
```

预期输出：

```python
最大值索引 = 1
最小值索引 = 0
最大值 = 99
最小值 = 15
```

二维数组最重要：

```python
import numpy as np

arr = np.array([[85, 92, 78],
                [88, 76, 95],
                [90, 81, 89]])

print("每列最大值所在行索引 =", np.argmax(arr, axis=0))
print("每行最大值所在列索引 =", np.argmax(arr, axis=1))
```

预期输出：

```python
每列最大值所在行索引 = [2 0 1]
每行最大值所在列索引 = [1 2 0]
```

这里的 `axis` 很关键：

- `axis=0`：按列看，往下比较。
- `axis=1`：按行看，往右比较。

---

## 6. 统计与聚合

一句话理解：把很多数压缩成更少的信息。

```python
import numpy as np

arr = np.array([[80, 90, 100],
                [70, 85, 95]])

print("mean =", np.mean(arr))
print("sum =", np.sum(arr))
print("max =", np.max(arr))
print("mean(axis=0) =", np.mean(arr, axis=0))
print("mean(axis=1) =", np.mean(arr, axis=1))
```

预期输出：

```python
mean = 86.66666666666667
sum = 520
max = 100
mean(axis=0) = [75.  87.5 97.5]
mean(axis=1) = [90. 83.33333333]
```

理解：

- `axis=0` 是“压行留列”。
- `axis=1` 是“压列留行”。

---

## 7. 逻辑筛选

```python
import numpy as np

arr = np.array([25, 48, 67, 82, 91])

print(arr[arr > 60])
print(arr[(arr > 30) & (arr < 90)])
print(np.any(arr > 90))
print(np.all(arr > 20))
```

预期输出：

```python
[67 82 91]
[48 67 82]
True
True
```

提醒：

- 用 `&` 代替 `and`
- 用 `|` 代替 `or`
- 条件两边通常都要加括号

---

## 8. 形状变换

```python
import numpy as np

arr = np.arange(12)

print(arr.reshape(3, 4))
print(arr.reshape(-1, 1))
print(arr.ravel())
```

预期输出：

```python
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[ 0]
 [ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]]
[ 0  1  2  3  4  5  6  7  8  9 10 11]
```

重点：`reshape()` 不会改变数据本身，只是换一种“看法”。

---

## 9. `lambda` 函数补充说明

这一部分并到 NumPy 学习里，是因为很多人会在“数据处理”和“快速写函数”时一起碰到它。

### 9.1 `lambda` 是什么

一句话理解：`lambda` 就是“匿名函数”，适合写很短的小函数。

普通函数写法：

```python
def square(x):
    return x * x

print(square(5))
```

等价的 `lambda` 写法：

```python
square = lambda x: x * x

print(square(5))
```

预期输出：

```python
25
```

固定结构：

```python
lambda 参数: 返回值
```

### 9.2 `lambda` 适合什么场景

适合：

- 临时的小函数
- 配合 `sorted()`、`map()`、`filter()`
- 不值得单独写 `def` 的简单逻辑

不太适合：

- 逻辑复杂
- 有多步判断
- 需要写很多注释

原则：`lambda` 应该短，一眼能看懂。

### 9.3 `lambda` 常见示例

#### 示例 1：排序时指定规则

```python
students = [("张三", 85), ("李四", 92), ("王五", 78)]

result = sorted(students, key=lambda x: x[1], reverse=True)
print(result)
```

预期输出：

```python
[('李四', 92), ('张三', 85), ('王五', 78)]
```

解释：`lambda x: x[1]` 的意思是“按每个元素的第 2 项排序”。

#### 示例 2：配合 `map()`

```python
nums = [1, 2, 3, 4]
result = list(map(lambda x: x * 2, nums))

print(result)
```

预期输出：

```python
[2, 4, 6, 8]
```

#### 示例 3：配合 `filter()`

```python
nums = [1, 2, 3, 4, 5, 6]
result = list(filter(lambda x: x % 2 == 0, nums))

print(result)
```

预期输出：

```python
[2, 4, 6]
```

### 9.4 `lambda` 和 NumPy 的关系

要点很重要：

- `lambda` 是 Python 语法。
- NumPy 更强调“向量化”，很多时候不需要 `map(lambda ...)`。

例如下面两种写法都能把每个元素乘 2：

写法 1：普通 Python 风格

```python
nums = [1, 2, 3, 4]
result = list(map(lambda x: x * 2, nums))
print(result)
```

写法 2：NumPy 风格，更推荐

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
result = arr * 2
print(result)
```

预期输出：

```python
[2 4 6 8]
```

结论：

- 处理普通列表时，`lambda` 很常见。
- 处理 NumPy 数组时，优先想“能不能直接数组运算”。

### 9.5 一个容易混淆的点

很多初学者会这样写：

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
result = list(map(lambda x: x * 2, arr))
print(result)
```

这当然能跑，但它没有发挥 NumPy 的优势。

更好的写法：

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr * 2)
```

因为：

- 更短
- 更快
- 更符合 NumPy 的思维方式

---

## 10. 一组建议你亲手跑的练习

### 练习 1：理解 `axis`

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(np.sum(arr, axis=0))
print(np.sum(arr, axis=1))
```

你先自己猜：

- 哪个结果是“每列求和”？
- 哪个结果是“每行求和”？

### 练习 2：理解 `np.where`

```python
import numpy as np

scores = np.array([45, 61, 59, 90, 72])
print(np.where(scores >= 60, "通过", "未通过"))
```

### 练习 3：理解 NumPy 和 `lambda` 的区别

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

result1 = list(map(lambda x: x + 10, arr))
result2 = arr + 10

print(result1)
print(result2)
```

观察：

- 两个结果是否一致？
- 哪个写法更像 NumPy？

---

## 11. 快速速查表

### 数组创建

```python
np.array(...)
np.zeros(...)
np.ones(...)
np.eye(...)
np.arange(...)
np.linspace(...)
```

### 常用运算

```python
a + b
a * b
a @ b
a ** 2
np.sqrt(a)
np.log(a)
np.exp(a)
```

### 常用统计

```python
np.mean(arr)
np.sum(arr)
np.min(arr)
np.max(arr)
np.argmax(arr)
np.argmin(arr)
```

### 条件筛选

```python
arr[arr > 0]
np.where(condition, x, y)
np.any(...)
np.all(...)
```

### 形状处理

```python
arr.reshape(...)
arr.ravel()
arr.flatten()
arr.T
```

---

## 12. 学习路线建议

建议顺序：

1. 先掌握数组创建、索引、切片。
2. 再重点吃透 `np.where()`、`np.argmax()`、`axis`。
3. 然后理解广播、`reshape()`、布尔筛选。
4. 最后再回头比较 `lambda` 和 NumPy 向量化的不同思路。

如果你愿意，下一步最值得继续深化的是这 3 个点：

- `axis=0` 和 `axis=1`
- `np.where()` 的条件替换思维
- “少写循环，多想数组整体运算”

---

> 最后更新：2026-05-24
> 说明：本版已调整为“可直接复制运行”的学习型 Markdown。

---

## 13. 项目常犯错误特殊说明（结合你的代码库）

以下内容来自这几个位置的实际代码习惯与问题：

- `D:\study\code\python_3.13_backup\code\MathMatFunction\NumPy_02.py`
- `D:\study\code\python_3.13_backup\code\MathMatFunction\exam2.py`
- `D:\study\code\python\JupyterProject\sample.ipynb`
- `C:\Users\zys\Desktop\NumPy与Matplotlib绘图完全教程.py`

### 13.1 导入相关（NumPy + plt）

错误写法：

```python
import matplotlib as plt
```

正确写法：

```python
import matplotlib.pyplot as plt
```

原因：`matplotlib` 是包，`pyplot` 才有 `plot/show/subplots` 等函数。

---

### 13.2 `np.argmax/np.argmin` 返回的是索引，不是分数

你项目里是对的，但这个点最容易后面写乱。

正确模板：

```python
max_idx = np.argmax(scores_of_subject)
max_score = scores_of_subject[max_idx]
max_student = students_names[max_idx]
```

避免误写成：

```python
max_score = np.argmax(scores_of_subject)   # 这其实是位置，不是分数
```

---

### 13.3 `axis` 含义最容易反

你代码里有：

```python
np.mean(scores, axis=1)   # 每个学生平均分
np.mean(scores, axis=0)   # 每门课平均分
```

记忆口诀：

- `axis=0`：跨行算，保留列（看“每列”）
- `axis=1`：跨列算，保留行（看“每行”）

快速自检法：先 `print(scores.shape)`，再判断你要“按行统计”还是“按列统计”。

---

### 13.4 `np.where` 的一个关键坑：它不是短路求值

你在 `NumPy_02.py` 的注释已经提到了这一点，非常关键。

像这样写递归时会出问题：

```python
# return np.where(abs(x)>=2, x*f(x-1), 1)
```

问题：`np.where` 会把两边表达式都先算出来，不会因为条件不满足就跳过某一边。

建议：

- 复杂分支用普通 `if/else`
- 向量场景再用 `np.where`

---

### 13.5 `reshape` 只能改形状，不能改元素总数

例如：

```python
a = np.arange(12)
a.reshape(5, 3)  # 错：15 != 12
```

正确方式：

```python
a.reshape(3, 4)
a.reshape(2, 6)
a.reshape(-1, 3)
```

---

### 13.6 `A * B` 和 `A @ B` 不是一回事

你项目里同时出现了这两种写法，容易混淆。

- `A * B`：逐元素乘法（形状需相同或可广播）
- `A @ B` / `np.dot(A, B)`：矩阵乘法

学习时建议每次都先打印 `shape`：

```python
print(A.shape, B.shape)
```

---

### 13.7 Jupyter 中“多打印了一行模块路径”的原因

你 `sample.ipynb` 的输出里有一行：

```text
<module 'numpy.random' from '...'>
```

这通常是误把模块对象本身输出了（如单独写了 `np.random` 或 `print(np.random)`）。

正确做法：只打印函数返回值，不打印模块对象。

---

### 13.8 Matplotlib 面向对象 API 常见误用（plt 部分重点）

当你写：

```python
fig, axes = plt.subplots(...)
```

后续应写：

```python
axes[0].plot(...)
axes[0].set_title(...)
axes[0].set_xlabel(...)
```

不要写成：

```python
axes[0].title("xxx")      # 错
axes[0].xlabel("xxx")     # 错
axes[0].plt.plot(...)     # 错
```

---

### 13.9 转置遍历时要确认维度语义

你有这种写法：

```python
for i, subject in enumerate(scores.T):
    std_subjects = subject.std()
```

这在“行=学生，列=科目”的前提下是正确的。  
但如果数据矩阵方向变了，就会统计错对象。

建议加一行防呆注释：

```python
# scores: shape=(学生数, 科目数)
```

---

### 13.10 一份可直接套用的防错模板（建议粘贴到新脚本开头）

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 保证复现

# 数据约定：行=样本，列=特征
scores = np.random.randint(30, 100, (5, 3))
print("shape:", scores.shape)

# 统计
print("每列均值:", np.mean(scores, axis=0))
print("每行均值:", np.mean(scores, axis=1))

# 索引和值分离
idx = np.argmax(scores[:, 0])
print("第0列最高分索引:", idx, "分数:", scores[idx, 0])

# 绘图
x = np.arange(scores.shape[0])
plt.plot(x, scores[:, 0], marker="o", label="第0列")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

这份模板可以明显减少你当前项目里 80% 的初学期错误。

---

## 14. 分层练习（题目 + 解析 + 参考答案）

这一节建议你按顺序做，不要跳级。  
做法：先看题目自己写，再看解析和答案对照。

### Level 1：基础数组与 axis

题目 1：

```python
import numpy as np

arr = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# 要求：
# 1) 求每行和（结果 shape 应该是 (3,)）
# 2) 求每列均值（结果 shape 应该是 (3,)）
# 3) 找出整表最大值及其索引位置
```

解析：

- 每行和：`axis=1`
- 每列均值：`axis=0`
- 最大值索引如果想要二维坐标，要用 `np.unravel_index`

参考答案：

```python
row_sum = np.sum(arr, axis=1)
col_mean = np.mean(arr, axis=0)
flat_idx = np.argmax(arr)
row_col = np.unravel_index(flat_idx, arr.shape)

print("每行和:", row_sum)
print("每列均值:", col_mean)
print("最大值:", arr[row_col], "位置:", row_col)
```

---

### Level 2：where、布尔索引、广播

题目 2：

```python
import numpy as np

scores = np.array([
    [58, 76, 91, 83],
    [62, 55, 79, 88],
    [94, 90, 96, 92],
    [43, 61, 67, 71]
])

# 要求：
# 1) 用 where 输出及格矩阵（及格=1，不及格=0）
# 2) 把所有 <60 的分数替换为 60
# 3) 每个科目加分 [2, 3, 1, 0]（广播）
```

解析：

- 条件替换：`np.where(cond, x, y)`
- 全表“就地保底”也可以用 `np.maximum(scores, 60)`
- 广播时要注意尾维匹配，`(4,4) + (4,)` 可行

参考答案：

```python
passed = np.where(scores >= 60, 1, 0)
fixed = np.where(scores < 60, 60, scores)
bonus = np.array([2, 3, 1, 0])
new_scores = np.clip(scores + bonus, 0, 100)

print("及格矩阵:\n", passed)
print("保底后:\n", fixed)
print("加分后:\n", new_scores)
```

---

### Level 3：argmax/argmin + 名称映射

题目 3：

```python
import numpy as np

students = np.array(["张三", "李四", "王五", "赵六"])
subjects = np.array(["语文", "数学", "英语"])
scores = np.array([
    [85, 90, 78],
    [92, 88, 95],
    [78, 85, 80],
    [90, 92, 88]
])

# 要求：
# 1) 每门课最高分是谁，分数多少
# 2) 每门课最低分是谁，分数多少
```

解析：

- `np.argmax(scores, axis=0)` 返回每一列的“行索引”
- 索引映射到名字：`students[idx]`
- 分数读取：`scores[idx, col]`

参考答案：

```python
max_idx = np.argmax(scores, axis=0)
min_idx = np.argmin(scores, axis=0)

for col, subject in enumerate(subjects):
    i_max = max_idx[col]
    i_min = min_idx[col]
    print(f"{subject}最高: {students[i_max]} {scores[i_max, col]}分")
    print(f"{subject}最低: {students[i_min]} {scores[i_min, col]}分")
```

---

### Level 4：reshape 与数据重排

题目 4：

```python
import numpy as np

a = np.arange(24)

# 要求：
# 1) reshape 成 (2, 3, 4)
# 2) 取出第 2 个“大块”中的第 3 行
# 3) 展平并检查是否与原 a 相同
```

解析：

- `reshape` 前后总元素个数必须一致
- 三维索引顺序是 `[块, 行, 列]`

参考答案：

```python
b = a.reshape(2, 3, 4)
print("b.shape =", b.shape)
print("目标切片:", b[1, 2, :])

flat = b.ravel()
print("展平后是否与原数组一致:", np.array_equal(flat, a))
```

---

### Level 5：线代小综合

题目 5：

```python
import numpy as np

A = np.array([[2, 1], [5, 3]])
B = np.array([[1, 2], [3, 4]])

# 要求：
# 1) 分别计算 A*B 与 A@B
# 2) 求 A 的行列式和逆矩阵
# 3) 解方程 A x = [7, 18]
```

参考答案：

```python
print("逐元素乘法:\n", A * B)
print("矩阵乘法:\n", A @ B)
print("det(A) =", np.linalg.det(A))
print("inv(A) =\n", np.linalg.inv(A))

b = np.array([7, 18])
x = np.linalg.solve(A, b)
print("方程解 x =", x)
```

---

## 15. 练习解析法（你可以套这个模板）

以后每次做题都按下面 4 步写，提升会非常快：

1. 明确输入形状：先 `print(arr.shape)`
2. 明确目标方向：按行还是按列（决定 `axis`）
3. 明确结果类型：要“值”还是“索引”
4. 做完复核：`shape`、范围、边界值、是否有 NaN

可直接复制的“解题骨架”：

```python
import numpy as np

def solve(arr):
    print("shape:", arr.shape)
    # 1) 你要按哪一维计算？
    # 2) 你要的是值，还是位置？
    # 3) 广播是否匹配？
    return None
```

---

## 16. 小项目实战（从 NumPy 到结论）

目标：只用 NumPy 完成一个完整分析。

### 16.1 题目

模拟 30 天网站数据并回答：

- 哪一天访问量最高？
- 哪一天转化效率最好？
- 哪几天 ROI 为负？
- 分周后哪一周收益最高？

### 16.2 参考实现

```python
import numpy as np

np.random.seed(2026)
days = np.arange(1, 31)
visits = np.random.randint(300, 900, size=30)
conversion = np.random.uniform(0.02, 0.09, size=30)
price = np.random.randint(35, 80, size=30)
ad_cost = np.random.randint(3000, 9000, size=30)

orders = np.round(visits * conversion).astype(int)
revenue = orders * price
roi = (revenue - ad_cost) / ad_cost

best_visit_day = days[np.argmax(visits)]
best_roi_day = days[np.argmax(roi)]
neg_roi_days = days[roi < 0]

weekly = np.array([
    revenue[:7].sum(),
    revenue[7:14].sum(),
    revenue[14:21].sum(),
    revenue[21:28].sum(),
    revenue[28:].sum()
])
best_week = np.argmax(weekly) + 1

print("访问量最高日:", best_visit_day)
print("ROI 最高日:", best_roi_day)
print("ROI 为负的日期:", neg_roi_days)
print("周收益:", weekly)
print("最佳周: Week", best_week)
```

### 16.3 结果解读建议

输出数字之后，不要停在“算出来了”，要补 3 句分析：

1. 趋势判断：整体是在上升、下降还是波动？
2. 风险判断：亏损天数是否集中，集中在什么区间？
3. 行动建议：下一步优先提升转化还是降低成本？

---

## 17. 高频坑点排查图（升级版）

### 17.1 广播报错

报错：

```text
ValueError: operands could not be broadcast together ...
```

排查顺序：

1. 打印两边 `shape`
2. 从最后一维往前看
3. 每一维必须“相等”或“其中一方为 1”

---

### 17.2 索引越界

报错：

```text
IndexError: index X is out of bounds
```

常见原因：

- 用了 `<= len(arr)` 而不是 `< len(arr)`
- `argmax` 结果用于了错误维度

---

### 17.3 dtype 被污染

报错：

```text
TypeError: ufunc ... did not contain a loop ...
```

常见原因：数组里混入了字符串或 `None`。  
修复方式：

```python
arr = arr.astype(float)
```

---

### 17.4 `np.where` 递归陷阱

错误思路：

```python
np.where(cond, f(x), g(x))
```

注意：`f(x)` 和 `g(x)` 都会先计算。  
复杂逻辑优先 `if/else`，`where` 用在向量化替换最稳。

---

## 18. 30 天打卡路径（可执行）

### 第 1 周：数组与索引

- Day 1-2：`array/zeros/ones/arange/linspace`
- Day 3-4：切片、花式索引、布尔索引
- Day 5-7：`mean/sum/max/min/argmax/argmin`

### 第 2 周：where、广播、shape 思维

- Day 8-10：`where` 和条件替换
- Day 11-12：广播规则专项
- Day 13-14：`reshape/ravel/transpose`

### 第 3 周：线代与数据处理

- Day 15-17：`@/dot/inv/det/solve`
- Day 18-19：`nanmean/nan_to_num`
- Day 20-21：综合练习（成绩分析）

### 第 4 周：项目化输出

- Day 22-25：30天业务数据分析
- Day 26-27：把结论写成 5 条要点
- Day 28-30：复盘 + 修正代码风格

每天完成标准：

- 至少跑通 1 段代码
- 至少写 1 条“为什么这样做”的解释
- 至少记录 1 个错误与修复方法

---

## 19. 下一步衔接（给 pandas 预热）

学完本文件后，建议马上进入 pandas。  
这 4 个 NumPy 思维会直接迁移过去：

1. 形状意识（二维数据本质）
2. 按轴聚合（groupby 之前的基础）
3. 条件筛选（布尔掩码思维）
4. 向量化优先（避免逐行循环）

你会发现 pandas 只是“带标签的二维数据 + 更强的数据清洗能力”。

---

> 最后更新：2026-05-25  
> 本次新增：分层练习、逐题解析、综合项目、打卡路径、排查图谱。
