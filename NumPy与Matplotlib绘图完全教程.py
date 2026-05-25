"""
NumPy 与 Matplotlib.pyplot 联合绘图完全教程
=============================================
运行环境：Python 3.13  +  numpy  +  matplotlib
运行方式：python "NumPy与Matplotlib绘图完全教程.py"

本文件既是教程，也是可直接运行的代码。
每个示例都封装为独立的 demo_*() 函数，取消注释即可运行对应示例。
建议：从第一部分顺序阅读，遇到感兴趣的示例取消注释运行看效果。

作者备注：本文件合并优化了多份学习资料，修正了所有已知问题，
        补充了详尽说明与错误分析。
"""

# ================================================================
# 准备工作：正确导入（最容易犯的错从这里开始）
# ================================================================

# --- 标准导入约定（记死！）---
import numpy as np                    # 别名 np，全行业通用
import matplotlib.pyplot as plt       # 别名 plt，全行业通用
# 3D 绘图现在无需额外导入，直接用 projection='3d' 即可
# (matplotlib 3.4+ 自动注册了 3D 投影)

"""
常见导入错误对照：
  错误写法                            后果
  ─────────────────────────────────────────────────────────
  import matplotlib as plt            AttributeError: module 'matplotlib' has no attribute 'plot'
                                      原因：matplotlib 是包，pyplot 才是绘图子模块
  import numpy                        繁琐，之后每次写 numpy.array(...) 多打6个字
  from matplotlib.pyplot import *     污染命名空间，不推荐
  import numpy as np 写在函数内部      每次调用都要重新导入，浪费资源
  from mpl_toolkits.mplot3d import Axes3D  多余！matplotlib 3.4+ 自动处理
"""


# ================================================================
# 第一部分：为什么需要 NumPy
# ================================================================

# --- 1.1 列表 vs NumPy 数组的直观对比 ---
def demo_1_1_list_vs_numpy():
    """对比 Python 原生列表和 NumPy 数组的行为差异"""
    import math
    import numpy as np

    print("=== Python 列表 vs NumPy 数组 ===\n")

    # Python 列表
    a_list = [1, 2, 3]
    print("列表 [1,2,3] * 2   =", a_list * 2)       # 复制！[1,2,3,1,2,3]

    # NumPy 数组
    a_arr = np.array([1, 2, 3])
    print("数组 [1,2,3] * 2   =", a_arr * 2)        # 逐元素！[2,4,6]

    # 尝试数学运算
    print("\n列表 + 10:", end=" ")
    try:
        print(a_list + 10)
    except TypeError as e:
        print(f"报错！{e}")

    print("数组 + 10 =", a_arr + 10)                 # 广播 [11,12,13]

    # 尝试取三角函数
    print("\nmath.sin(列表):", end=" ")
    try:
        print(math.sin(a_list))
    except TypeError as e:
        print(f"报错！{e}")

    print("np.sin(数组)  =", np.sin(a_arr))

    print("\n结论：NumPy 让 '一堆数字' 的行为像 '一个数学对象'。")

# demo_1_1_list_vs_numpy()


# ================================================================
# 第二部分：NumPy 核心操作
# ================================================================

# --- 2.1 八大创建方法 ---
"""
方法                 用途                    记忆点
───────────────────────────────────────────────────────────
np.array(列表)        从已有数据创建           最通用
np.arange(起,止,步)   固定步长序列             不包含 stop（和 range 一样）
np.linspace(起,止,N)  固定点数（包含stop）     画平滑曲线首选！
np.zeros(N)           全零数组                 初始化占位
np.ones(N)            全一数组                 初始权重
np.eye(N)             单位矩阵（对角=1）       矩阵运算
np.random.rand(N)     [0,1) 均匀随机          统计模拟
np.random.randn(N)    标准正态随机             噪声生成
np.full(N, 值)        全部填同一个值           常数数组
"""
def demo_2_1_create():
    import numpy as np
    print("=== 2.1 数组创建方法 ===")
    print("arange(0, 10, 2) :", np.arange(0, 10, 2))       # [0 2 4 6 8]
    print("linspace(0, 1, 5):", np.linspace(0, 1, 5))       # [0. 0.25 0.5 0.75 1.0]
    print("zeros(3)         :", np.zeros(3))                 # [0. 0. 0.]
    print("ones(3)          :", np.ones(3))                  # [1. 1. 1.]
    print("eye(3):\n", np.eye(3))                            # 3x3 单位矩阵
    print("rand(4)          :", np.random.rand(4))           # [0,1) 均匀
    print("randn(4)         :", np.random.randn(4))          # 标准正态
    print("full(5, 7)       :", np.full([5,4], 7))               # [7 7 7 7 7]
# print(demo_2_1_create())


# --- 2.2 arange vs linspace：最容易选错的函数 ---
"""
arange(0, 10, 0.3)   → 知道「步长」，不包含10，点数不可控
linspace(0, 10, 100)  → 知道「点数」，包含10，步长自动算

画图时永远优先用 linspace！原因：
  - 点数精确可控，保证曲线平滑度
  - 浮点累加误差不会让你少一个点（arange 的通病）

实锤对比：
  np.arange(0, 1, 0.1) → 可能停在 0.9（浮点误差导致 0+10*0.1 < 1.0）
  np.linspace(0, 1, 11) → 稳稳的 [0, 0.1, 0.2, ..., 1.0]
"""


# --- 2.3 数据类型（dtype）的坑 ---
"""
创建数组时不要混入字符串，否则整个数组全部变成字符串类型：

  np.array([1, 2, 3])       → dtype=int32       正常
  np.array([1, 2, 3.0])     → dtype=float64     混入浮点，提升为浮点（还好）
  np.array([1, 2, '3'])     → dtype='<U1'       混入字符串，全变字符串！（灾难）
  np.zeros(3)               → dtype=float64     默认浮点
  np.zeros(3, dtype=int)    → dtype=int32       显式指定类型

如果后续要对数组做数学运算而数组变成了字符串类型，会得到：
  TypeError: ufunc 'add' did not contain a loop with signature matching types
"""


# --- 2.4 向量化运算：不需要 for 循环 ---
"""
NumPy 的"向量化"：一次操作应用到数组所有元素，无需 for 循环。
比 Python 原生循环快 10~100 倍，这是 NumPy 存在的根本意义。
"""
def demo_2_4_vectorize():
    import numpy as np
    a = np.array([1, 2, 3, 4, 5])
    print("=== 2.4 向量化运算 ===")
    print("a + 10     =", a + 10)
    print("a * 3      =", a * 3)
    print("a ** 2     =", a ** 2)
    print("np.sqrt(a) =", np.sqrt(a))
    print("np.sin(a)  =", np.sin(a))
    print("a > 3      =", a > 3)           # 布尔数组
    print("a[a > 3]  =", a[a > 3])         # 布尔索引
# demo_2_4_vectorize()


# --- 2.5 广播（broadcasting）：最强大也最易错的特性 ---
"""
广播规则（从最后一个维度往前对齐）：
  (3, 4) 与 (4,)   → 可行，尾维度 4=4
  (3, 4) 与 (3,)   → 可行（第二维3=3，第一维补1→扩展为(1,3)→(3,3)）
  (3, 4) 与 (5,)   → 报错！尾维度 4≠5
"""
def demo_2_5_broadcast():
    import numpy as np
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])           # shape (2, 3)
    print("=== 2.5 广播 ===")
    print("a + [10, 20, 30]:\n", a + np.array([10, 20, 30]))
    print("a + [[10], [20]]:\n", a + np.array([[10], [20]]))
    # 下面这行会报错，取消注释试试：
    # d = np.array([10, 20])            # shape (2,)
    # print(a + d)                      # 报错：(2,3) + (2,) → 尾维度 3≠2
# demo_2_5_broadcast()


# --- 2.6 reshape：改变看数据的角度 ---
"""
a = np.arange(12)       → [0..11]  shape (12,)
a.reshape(3, 4)          → 3行4列
a.reshape(3, -1)         → 3行，列自动算（=4）
a.reshape(-1, 1)         → 12行1列（列向量）

⚠️ 总元素数不能变！12个元素只能 reshape 为乘积=12 的形状。
   否则：ValueError: cannot reshape array of size 12 into shape (5,3)
"""


# ================================================================
# 第三部分：Matplotlib 快速入门
# ================================================================

# --- 3.1 你的第一张图：最小示例 ---
def demo_3_1_first_plot():
    """最小可运行的绘图示例"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
    plt.title('y = sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
# demo_3_1_first_plot()


# --- 3.2 plot() 参数详解 ---
"""
plt.plot(x, y, '格式串', **关键字参数)

格式串 (fmt) = 颜色 + 标记 + 线型（顺序可互换）：
  颜色: 'b'蓝 'r'红 'g'绿 'k'黑 'm'品红 'c'青 'y'黄 'w'白
  标记: 'o'圆 '.'点 's'方 '^'三角 '+'十字 '*'星 'd'菱形
  线型: '-'实线 '--'虚线 '-.'点划线 ':'点线 ''无线

  示例：'b-' 蓝色实线, 'ro' 红色圆点不连线, 'g--' 绿色虚线, 'ks-' 黑色方块实线

关键字参数（比 fmt 更精确）：
  color='red' / '#FF0000' / (1.0, 0.0, 0.0)
  linewidth=2.0       线宽
  linestyle='--'      线型
  marker='o'          标记类型
  markersize=8        标记大小
  markerfacecolor='white' / markeredgecolor='blue'
  alpha=0.7           透明度(0~1)
  label='图例文本'
"""


# --- 3.3 坐标系控制 ---
def demo_3_3_axes_control():
    """坐标轴范围、等比例、多线叠加"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 150)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 左图：多线叠加
    axes[0].plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
    axes[0].plot(x, np.cos(x), 'r--', label='cos(x)', linewidth=2)
    axes[0].set_title('多线叠加')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 中图：限制坐标范围
    axes[1].plot(x, np.sin(x))
    axes[1].set_xlim(0, np.pi)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title('xlim=0~pi, ylim=0~1.2')
    axes[1].grid(True, alpha=0.3)

    # 右图：等比例坐标轴（画圆必备）
    t = np.linspace(0, 2 * np.pi, 200)
    axes[2].plot(np.cos(t), np.sin(t), 'm-', linewidth=2)
    axes[2].set_aspect('equal')   # [重要] 保证圆形不变椭圆
    axes[2].set_title('set_aspect("equal")')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
# demo_3_3_axes_control()


# ================================================================
# 第四部分：Figure 与 Axes（面向对象方式）
# ================================================================

"""
两种编程方式：
  方式A（pyplot 状态机）：简单快捷，适合探索
    plt.plot(x, y)       → 画在"当前"坐标系
    plt.title('xxx')     → 设置"当前"标题

  方式B（面向对象）：精确控制，适合复杂布局 [推荐]
    fig, ax = plt.subplots()   → 拿到 Figure 和 Axes 对象
    ax.plot(x, y)              → 画在指定坐标系
    ax.set_title('xxx')        → 注意是 set_title，不是 title！
    ax.set_xlabel('xxx')       → 注意是 set_xlabel，不是 xlabel！

⚠️ 常见错误：在 Axes 对象上调用 plt.xxx()
  axes[0,0].plt.plot(...)   → 报错！Axes 对象没有 plt 属性
  axes[0,0].title('xxx')    → 报错！Axes 的方法是 set_title
"""


# --- 4.1 子图布局：2x2 示例 ---
def demo_4_1_subplots():
    """subplots 创建 2x2 布局"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 100)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('sin(x)')

    axes[0, 1].plot(x, np.cos(x), 'r-')
    axes[0, 1].set_title('cos(x)')

    axes[1, 0].plot(x, np.tan(x), 'g-')
    axes[1, 0].set_title('tan(x)')
    axes[1, 0].set_ylim(-3, 3)

    axes[1, 1].plot(x, np.sin(x) * np.cos(x), 'm-')
    axes[1, 1].set_title('sin(x) * cos(x)')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
# demo_4_1_subplots()


# ================================================================
# 第五部分：参数方程与几何曲线
# ================================================================

"""
绘制参数方程的要点：
  1. 理解周期，选对参数 t 的范围（太长会重复覆盖，太短会不完整）
  2. 别忘了 set_aspect('equal')，否则圆变椭圆、正方形变长方形
  3. meshgrid 和参数方程是两回事——这里只用一维 t 即可
"""

# --- 5.1 经典曲线合集 ---
def demo_5_1_parametric_curves():
    """四种经典参数曲线：圆、心形线、利萨如、三叶玫瑰线"""
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 2 * np.pi, 500)

    # ---- 圆 ----
    x_circle = np.cos(t)
    y_circle = np.sin(t)

    # ---- 心形线 (Cardioid) ----
    # 正确方程: r = 1 - cos(t),  x = r*cos(t), y = r*sin(t)
    x_cardioid = (1 - np.cos(t)) * np.cos(t)
    y_cardioid = (1 - np.cos(t)) * np.sin(t)

    # ---- 利萨如曲线 (Lissajous) ----
    # 核心：x和y的频率不同，这里 3:2 产生经典 8 字形
    #   如果频率相同（如 sin(3t), cos(3t)），画出来只是斜椭圆！
    t_liss = np.linspace(0, 2 * np.pi, 1000)
    x_liss = np.sin(3 * t_liss)
    y_liss = np.sin(2 * t_liss)

    # ---- 三叶玫瑰线 (Rose Curve) ----
    # r = cos(3*theta)，周期 2π/3，完整三瓣只需要 [0, π]
    #   用 [0, 2π] 会重复覆盖，线条来回重复
    theta = np.linspace(0, np.pi, 500)
    r = np.cos(3 * theta)
    x_rose = r * np.cos(theta)
    y_rose = r * np.sin(theta)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, (x, y, title, color) in zip(axes.flat, [
        (x_circle,  y_circle,  '圆',   'blue'),
        (x_cardioid, y_cardioid, '心形线', 'red'),
        (x_liss,    y_liss,    '利萨如 3:2', 'green'),
        (x_rose,    y_rose,    '三叶玫瑰线', 'purple'),
    ]):
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.set_aspect('equal')    # [关键] 没这行圆会变椭圆
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
# demo_5_1_parametric_curves()


# --- 5.2 参数方程常见错误演示 ---
def demo_5_2_parametric_errors():
    """参数方程常见错误对比：正确 vs 错误"""
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 2*np.pi, 500)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # (0,0) 心形线正确版
    axes[0,0].plot((1 - np.cos(t)) * np.cos(t),
                   (1 - np.cos(t)) * np.sin(t), 'r-', linewidth=1.5)
    axes[0,0].set_aspect('equal')
    axes[0,0].set_title('心形线 [正确] r = 1-cos(t)')

    # (0,1) 心形线错误版：误用 cos²
    axes[0,1].plot((1 - np.cos(t)**2) * np.cos(t),
                   (1 - np.cos(t)**2) * np.sin(t), 'orange', linewidth=1.5)
    axes[0,1].set_aspect('equal')
    axes[0,1].set_title('心形线 [错误] 误用 r=1-cos^2(t) → 水滴状')

    # (1,0) 利萨如正确版：频率不同 3:2
    t_l = np.linspace(0, 2*np.pi, 1000)
    axes[1,0].plot(np.sin(3*t_l), np.sin(2*t_l), 'g-', linewidth=1.5)
    axes[1,0].set_aspect('equal')
    axes[1,0].set_title('利萨如 [正确] sin(3t) vs sin(2t)')

    # (1,1) 利萨如错误版：频率相同 3:3
    axes[1,1].plot(np.sin(3*t_l), np.cos(3*t_l), 'olive', linewidth=1.5)
    axes[1,1].set_aspect('equal')
    axes[1,1].set_title('利萨如 [错误] sin(3t) vs cos(3t) → 斜椭圆')

    plt.tight_layout()
    plt.show()
# demo_5_2_parametric_errors()


# ================================================================
# 第六部分：meshgrid 与二维函数可视化
# ================================================================

"""
meshgrid 什么时候用？
  当你需要计算 z = f(x, y) 这种"关于两个变量的函数"时。

原理：
  x = [1, 2, 3]    一维
  y = [10, 20]     一维
  想算 z = x + y 在每一对 (x,y) 上的值

  X, Y = np.meshgrid(x, y) 返回：
    X = [[1, 2, 3],   Y = [[10, 10, 10],
         [1, 2, 3]]        [20, 20, 20]]

  然后 Z = X + Y 一次性算出整个二维网格的值。
  meshgrid 把"叉积关系"变成可直接向量化运算的形状。
"""
def demo_6_1_meshgrid():
    """meshgrid 原理演示"""
    import numpy as np
    x = np.array([1, 2, 3])
    y = np.array([10, 20])
    X, Y = np.meshgrid(x, y)
    print("x =", x)
    print("y =", y)
    print("X:\n", X)
    print("Y:\n", Y)
    print("X + Y:\n", X + Y)
# demo_6_1_meshgrid()


# --- 6.2 等高线图 ---
def demo_6_2_contour():
    """等高线图：contourf + contour"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X**2 + Y**2) / (X**2 + Y**2 + 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cf = axes[0].contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=axes[0])
    axes[0].set_title('contourf（填充等高线）')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    axes[1].contour(X, Y, Z, levels=15, colors='black', linewidths=0.6)
    axes[1].set_title('contour（线框等高线）')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    plt.tight_layout()
    plt.show()
# demo_6_2_contour()


# ================================================================
# 第七部分：3D 曲面绘图
# ================================================================

"""
3D 绘图要点：
  1. 创建子图时指定 projection='3d'（无需额外导入！）
  2. 必须先 meshgrid 生成二维网格
  3. 注意除零问题（如 sin(R)/R 在 R=0 处未定义）
  4. tight_layout() 对 3D 图形效果有限，用 subplots_adjust 手动调
  5. 保存时用 fig.savefig()，确保在 show() 之前调用
"""

# --- 7.1 3D 曲面：抛物面 + 高斯曲面 ---
def demo_7_1_3d_surfaces():
    """3D 曲面图"""
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 80)
    y = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(x, y)

    Z_para = X**2 + Y**2                               # 抛物面
    Z_gauss = np.exp(-(X**2 + Y**2) / 2)               # 高斯函数

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('3D 曲面图', fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_para, cmap='viridis', edgecolor='none')
    ax1.set_title('抛物面 z = x^2 + y^2')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_gauss, cmap='Blues', edgecolor='none')
    ax2.set_title('高斯函数 z = exp(-(x^2+y^2)/2)')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.subplots_adjust(wspace=0.15)    # 手动调整间距（3D图用这个比 tight_layout 好）
    plt.show()
# demo_7_1_3d_surfaces()


# --- 7.2 sinc 函数：正确处理 R=0 除零 ---
def demo_7_2_sinc():
    """波浪面 z = sin(R)/R 的除零处理（经典坑）"""
    import numpy as np
    import matplotlib.pyplot as plt

    R_range = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(R_range, R_range)
    R = np.sqrt(X**2 + Y**2)

    # [坑在此处] R==0 时 sin(0)/0 = 0/0 = NaN
    #   直接写 Z = np.sin(R)/R 会出 RuntimeWarning，且图上有个洞
    #   解决方法：先算，再用 sinc 极限值（=1）填洞

    # 错误做法（取消注释下面这行运行看看警告）：
    # Z_bad = np.sin(R) / R

    # 正确做法：利用 np.sinc 或手动填洞
    # np.sinc(x) = sin(pi*x)/(pi*x)，这里用安全写法
    Z = np.ones_like(R)                               # 默认填 1（sinc(0) 极限）
    mask = R != 0
    Z[mask] = np.sin(R[mask]) / R[mask]               # 只在 R≠0 时算

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
    ax1.set_title('sinc(R) [正确：R=0处填1]')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

    ax2 = fig.add_subplot(1, 2, 2)
    c = ax2.contourf(X, Y, Z, levels=30, cmap='coolwarm')
    plt.colorbar(c, ax=ax2)
    ax2.set_title('sinc(R) 等高线')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()
# demo_7_2_sinc()


# ================================================================
# 第八部分：常用图表类型
# ================================================================

# --- 8.1 折线图 plot() ---
def demo_8_1_line():
    import numpy as np
    import matplotlib.pyplot as plt
    days = np.arange(1, 31)
    prices = 100 + np.cumsum(np.random.randn(30))
    plt.plot(days, prices, 'b-o', markersize=4, linewidth=1.5,
             markerfacecolor='white')
    plt.fill_between(days, prices, alpha=0.2)
    plt.title('Simulated Stock Price (30 Days)')
    plt.xlabel('Day'); plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.show()
# demo_8_1_line()


# --- 8.2 散点图 scatter() ---
def demo_8_2_scatter():
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 0.6 * x + np.random.randn(n) * 0.5
    colors = np.sqrt(x**2 + y**2)
    sizes = np.abs(x + y) * 50 + 10
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    plt.colorbar(label='Distance from origin')
    plt.title('Scatter: Positive Correlation')
    plt.xlabel('x'); plt.ylabel('y')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, alpha=0.2)
    plt.show()
# demo_8_2_scatter()


# --- 8.3 柱状图 bar() ---
def demo_8_3_bar():
    import numpy as np
    import matplotlib.pyplot as plt
    cats = ['Python', 'C++', 'Java', 'JS', 'Go', 'Rust']
    vals = [95, 72, 68, 80, 55, 48]
    colors = ['#3776AB', '#00599C', '#ED8B00', '#F7DF1E', '#00ADD8', '#DEA584']
    bars = plt.bar(cats, vals, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', fontsize=10)
    plt.title('Programming Language Popularity')
    plt.ylabel('Score')
    plt.ylim(0, max(vals) + 10)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
# demo_8_3_bar()


# --- 8.4 直方图 hist() ---
def demo_8_4_hist():
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(42)
    data = np.random.randn(2000)
    plt.hist(data, bins=40, color='steelblue', edgecolor='white',
             alpha=0.8, density=True)
    x = np.linspace(-4, 4, 200)
    y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x, y, 'r--', linewidth=2, label='N(0,1) PDF')
    plt.title('Histogram of Normal Distribution (n=2000)')
    plt.xlabel('Value'); plt.ylabel('Probability Density')
    plt.legend(); plt.show()
# demo_8_4_hist()


# --- 8.5 饼图 pie() ---
def demo_8_5_pie():
    import matplotlib.pyplot as plt
    sizes = [35, 25, 20, 12, 8]
    labels = ['Python', 'JavaScript', 'Java', 'C++', 'Other']
    colors = ['#3776AB', '#F7DF1E', '#ED8B00', '#00599C', '#AAAAAA']
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.75,
            explode=(0.05, 0, 0, 0, 0))
    plt.title('Language Usage Share')
    plt.axis('equal')           # [关键] 正圆形
    plt.show()
# demo_8_5_pie()


# --- 8.6 热力图 imshow() ---
def demo_8_6_heatmap():
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(42)
    data = np.random.randn(10, 10)
    corr = data @ data.T
    corr /= np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    plt.colorbar(label='Correlation', shrink=0.8)
    plt.title('Correlation Heatmap')
    # 注意：imshow 坐标是 (y, x) → text 时列号在前 (j, i)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                     fontsize=8,
                     color='black' if abs(corr[i,j]) < 0.7 else 'white')
    plt.xticks(range(10), [f'V{i+1}' for i in range(10)], rotation=45)
    plt.yticks(range(10), [f'V{i+1}' for i in range(10)])
    plt.tight_layout(); plt.show()
# demo_8_6_heatmap()


# ================================================================
# 第九部分：样式、字体与图片保存
# ================================================================

"""
常用内置样式：
  'ggplot'          R语言风格
  'seaborn-v0_8'    seaborn 风格（推荐日常使用）
  'dark_background' 深色背景
  'fivethirtyeight' 数据新闻风格（粗线条大字）

查看全部：print(plt.style.available)

中文字体设置（Windows）：
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
"""

# --- 9.1 样式对比 ---
def demo_9_1_styles():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x = np.linspace(0, 10, 100)
    styles = ['default', 'ggplot', 'seaborn-v0_8', 'fivethirtyeight']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, style in zip(axes.flat, styles):
        with plt.style.context(style):
            ax.plot(x, np.sin(x), linewidth=2, label='sin(x)')
            ax.plot(x, 0.5 * np.cos(x), '--', linewidth=2, label='cos(x)')
        ax.set_title(f'Style: {style}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
# demo_9_1_styles()


# --- 9.2 保存图片到文件 ---
def demo_9_2_savefig():
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0, 2*np.pi, 100)
    plt.plot(x, np.sin(x), linewidth=2)
    plt.title('sin(x)')

    # [关键] savefig 必须在 show() 之前调用！
    #   show() 之后 Figure 会被清空，保存的就是空白图
    plt.savefig('output_plot.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    # dpi: 分辨率（打印用 300，屏幕看用 100-150）
    # bbox_inches='tight': 自动裁切多余白边
    # 支持格式: .png .jpg .svg .pdf（.svg/.pdf 是矢量图，可无限放大）

    print("Figure saved to output_plot.png")
    plt.close()                # 释放内存 [好习惯]
# demo_9_2_savefig()


# ================================================================
# 第十部分：常见错误大全（附可复现示例代码）
# ================================================================

"""
以下每个错误都包含：原因解释 + 错误代码演示 + 正确写法。
取消注释对应的错误块来看看它长什么样。
"""

# --- 10.1 x 和 y 长度不一致 ---
# def demo_error_1():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     x = np.linspace(0, 10, 100)          # 100个点
#     y = np.sin(np.linspace(0, 10, 99))   # 99个点！
#     plt.plot(x, y)
#     plt.show()
#     # 报错：ValueError: x and y must have same first dimension, but have shapes (100,) and (99,)


# --- 10.2 中文显示为方框 ---
# def demo_error_2():
#     import matplotlib.pyplot as plt
#     plt.title('没有设置中文字体')   # → 显示为 □□□□
#     plt.show()
#     # 解决：plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#     #       plt.rcParams['axes.unicode_minus'] = False


# --- 10.3 未导入就使用 plt ---
# def demo_error_3():
#     plt.plot([1,2,3], [1,2,3])   # NameError: name 'plt' is not defined
#     # 解决：import matplotlib.pyplot as plt


# --- 10.4 广播形状不兼容 ---
# def demo_error_4():
#     import numpy as np
#     a = np.ones((3, 4))          # shape (3, 4)
#     b = np.ones(5)               # shape (5,)
#     print(a + b)                 # ValueError: operands could not be broadcast
#     # 解决：打印 .shape，确保尾维度匹配 → b = np.ones(4)


# --- 10.5 热力图标注位置偏移 ---
# def demo_error_5():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     data = np.random.randn(5, 5)
#     plt.imshow(data, cmap='RdBu_r')
#     # 错误：plt.text(i, j, ...)  # i是行号，j是列号 → 标注"转置"了
#     # 正确：plt.text(j, i, ...)  # 列号在前！（imshow坐标是(y,x)）
#     for i in range(5):
#         for j in range(5):
#             plt.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=7)
#     plt.show()


# --- 10.6 子图标签重叠 ---
# def demo_error_6():
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(2, 3)        # 默认 figsize 太小！
#     for i, ax in enumerate(axes.flat):
#         ax.set_title(f'Subplot {i+1} with a very long title that overlaps')
#     # plt.tight_layout()  解决方案：增大 figsize + 加上这行
#     plt.show()


# --- 10.7 savefig 先 show 后 save → 空白图 ---
# def demo_error_7():
#     import matplotlib.pyplot as plt
#     plt.plot([1,2,3])
#     plt.show()                  # 先show → Figure被清空
#     plt.savefig('blank.png')    # 保存的是空白图！
#     # 解决：savefig 放 show 之前，或用 fig.savefig()


# --- 10.8 arange 浮点精度丢失 ---
# def demo_error_8():
#     import numpy as np
#     pts = np.arange(0, 1, 0.1)
#     print("arange(0,1,0.1):", pts)           # 可能少最后一个点
#     print("length:", len(pts))
#     # 解决：用 np.linspace(0, 1, 11) 代替


# --- 10.9 math.sin 不能处理 NumPy 数组 ---
# def demo_error_9():
#     import numpy as np
#     import math
#     x = np.linspace(0, np.pi, 10)
#     y = math.sin(x)  # TypeError: only length-1 arrays can be converted
#     # 解决：y = np.sin(x)  使用 NumPy 版本的函数


# --- 10.10 图例不显示 ---
# def demo_error_10():
#     import matplotlib.pyplot as plt
#     plt.plot([1,2,3], [1,2,3])    # 没有 label 参数
#     plt.plot([1,2,3], [3,2,1])
#     plt.legend()                   # 图例为空！
#     plt.show()
#     # 解决：plt.plot(..., label='线条1')


# --- 10.11 圆变椭圆 ---
# def demo_error_11():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     t = np.linspace(0, 2*np.pi, 200)
#     plt.plot(np.cos(t), np.sin(t))
#     # 忘了 set_aspect('equal') → 画出个椭圆！
#     plt.show()
#     # 解决：plt.gca().set_aspect('equal') 或 plt.axis('equal')


# --- 10.12 scatter 的 s 参数太小/太大 ---
# def demo_error_12():
#     import matplotlib.pyplot as plt
#     plt.scatter([1,2,3], [1,2,3], s=1)    # s 是面积，1太小了看不到
#     #       s=1000                          # 太大了全糊在一起
#     plt.show()
#     # 建议：s 取值通常在 10~500 之间


# --- 10.13 3D 图形上 tight_layout 不生效 ---
# 3D axes 的 tight_layout 支持很差，用 fig.subplots_adjust() 手动调

# --- 10.14 错误：在 Axes 上用 .plt 方法 ---
# axes[0,0].plt.plot(x, y)       # AttributeError: no attribute 'plt'
# axes[0,0].title('xxx')         # 应该是 set_title
# axes[0,0].xlabel('xxx')        # 应该是 set_xlabel
# axes[0,0].grid(True)           # grid 在 Axes 上直接可用，不需要 plt.grid


# ================================================================
# 第十一部分：综合练习 — 气象数据 Dashboard
# ================================================================

def demo_11_dashboard():
    """综合练习：一个月的气温和湿度数据，多视图展示"""
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # ---- 生成模拟数据 ----
    np.random.seed(42)
    days = np.arange(1, 31)
    temp = 20 + 8 * np.sin(np.linspace(0, np.pi, 30)) + np.random.randn(30) * 2
    humidity = 60 + 15 * np.cos(np.linspace(0, np.pi, 30)) + np.random.randn(30) * 5
    humidity = np.clip(humidity, 0, 100)

    # ---- 2x2 布局 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('May Weather Report', fontsize=16, fontweight='bold', y=0.98)

    # 图1: 折线图 — 气温变化
    ax = axes[0, 0]
    ax.plot(days, temp, 'r-o', linewidth=2, markersize=5,
            markerfacecolor='white', label='Daily Temp')
    ax.fill_between(days, temp, alpha=0.15, color='red')
    ax.axhline(np.mean(temp), color='gray', linestyle='--',
               linewidth=1, label=f'Mean: {np.mean(temp):.1f}C')
    ax.set_title('Temperature Trend'); ax.set_xlabel('Day'); ax.set_ylabel('Temp (C)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 图2: 散点图 — 温度 vs 湿度
    ax = axes[0, 1]
    sc = ax.scatter(temp, humidity, c=days, cmap='coolwarm',
                    s=100, edgecolors='white', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='Day')
    corr = np.corrcoef(temp, humidity)[0, 1]
    ax.set_title(f'Temperature vs Humidity (r = {corr:.3f})')
    ax.set_xlabel('Temp (C)'); ax.set_ylabel('Humidity (%)')
    ax.grid(True, alpha=0.3)

    # 图3: 柱状图 — 按周分组
    ax = axes[1, 0]
    weeks = ['Week1', 'Week2', 'Week3', 'Week4']
    means = [np.mean(temp[i*7:(i+1)*7]) for i in range(4)]
    if len(temp) > 28: means[-1] = np.mean(temp[21:])
    bars = ax.bar(weeks, means,
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                  edgecolor='white')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{m:.1f}C', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Weekly Average Temp'); ax.set_ylabel('Temp (C)')
    ax.set_ylim(0, max(means) * 1.15); ax.grid(axis='y', alpha=0.3)

    # 图4: 直方图 — 湿度分布
    ax = axes[1, 1]
    ax.hist(humidity, bins=10, color='steelblue', edgecolor='white',
            alpha=0.8, density=True)
    ax.axvline(np.mean(humidity), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(humidity):.1f}%')
    ax.set_title('Humidity Distribution'); ax.set_xlabel('Humidity (%)')
    ax.set_ylabel('Probability Density'); ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
# demo_11_dashboard()


# ================================================================
# 第十二部分：编写建议与最佳实践
# ================================================================

BEST_PRACTICES = """
============================================================
编写 NumPy + Matplotlib 代码的最佳实践
============================================================

1. 【命名规范】
   - 统一: import numpy as np, import matplotlib.pyplot as plt
   - 避免: from xxx import *（污染命名空间）
   - 3D绘图不需要额外导入，直接用 projection='3d'

2. 【数据准备】
   - 画曲线：用 np.linspace 生成 x（点数精确，包含端点）
   - 画曲面：先 np.linspace → meshgrid → 计算 Z
   - 向量化：永远不要写 for i in range(n) 来逐元素计算！用 np.sin(x) 一次解决

3. 【绘图流程】
   - 创建 figure/axes（推荐 plt.subplots）
   - 调用绘图方法画数据
   - 设置标题、标签、图例、网格
   - plt.savefig 保存（在 show 之前！）
   - plt.show 显示
   - plt.close 释放内存

4. 【子图操作】
   - 用 axes[i,j] 索引（0-based），axes.flat 遍历
   - 几何图形务必加 set_aspect('equal')
   - plt.tight_layout() 自动调间距（3D 图例外，用 subplots_adjust）

5. 【参数方程】
   - 确认参数范围（周期、定义域）
   - 心形线、利萨如、玫瑰线需查证正确方程再编码
   - 频率比相同 → 是椭圆不是利萨如！

6. 【3D 绘图】
   - projection='3d' 即可，不需要单独导入
   - plot_surface 时 edgecolor='none' 更美观
   - 处理除零（如 sinc 函数在 R=0 处）
   - 保存时用 fig.savefig()

7. 【调试技巧】
   - 先打印数据形状（.shape），确认维度匹配
   - 遇到 RuntimeWarning 说明有 NaN/Inf，追踪来源
   - 逐步构建：先画最简单的，再加细节

8. 【保证颜色一致性】
   - 散点图/热力图的 c 和 colorbar 的 cmap 要配套
   - 自定义色带：plt.cm.viridis / plt.cm.coolwarm / plt.cm.RdBu_r
"""


# ================================================================
# 速查表
# ================================================================

CHEAT_SHEET = """
============================================================
NumPy 常用函数
============================================================
np.array(list)          从列表创建
np.arange(start,stop,step) 固定步长（不含stop）
np.linspace(start,stop,N)  固定点数（含stop）[画图首选]
np.zeros/ones/eye       零/一/单位矩阵
np.random.rand(n)       [0,1)均匀随机
np.random.randn(n)      标准正态随机
np.full(n, val)         填充指定值
np.meshgrid(x, y)       网格化 [二维绘图必备]
np.sin/cos/sqrt/exp     逐元素数学运算
np.mean/std/sum/min/max 统计函数
np.clip(a, lo, hi)      限幅
np.corrcoef(a, b)       相关系数
a.reshape(r, c)         改变形状
a.ravel()               展平为一维
np.column_stack((a,b))  列拼接
np.where(cond, a, b)    条件选择

============================================================
Matplotlib.pyplot 常用函数
============================================================
plt.plot(x, y)          折线图
plt.scatter(x, y)       散点图
plt.bar(x, height)      柱状图
plt.hist(data, bins)    直方图
plt.pie(sizes)          饼图
plt.imshow(matrix)      热力图
plt.contour/contourf    等高线
plt.title/xlabel/ylabel 标题/轴标签
plt.xlim/ylim           坐标轴范围
plt.legend()            图例
plt.grid(True)          网格
plt.colorbar()          颜色条
plt.axhline/axvline     参考线
plt.fill_between        区域填充
plt.subplots(r,c)       创建多子图
plt.tight_layout()      自动调间距
plt.savefig(path)       保存图片
plt.show()              显示
plt.close()             关闭
plt.style.use(name)     设置样式
plt.rcParams            全局配置（字体等）
============================================================
"""


# ================================================================
# 第十三部分：分层练习（含解析与标准答案）
# ================================================================

PRACTICE_BOOK = """
============================================================
分层练习导航（建议顺序）
============================================================
Level 1（基础巩固）:
  demo_13_1_practice_axis_and_stats
  目标：彻底吃透 axis=0 / axis=1、argmax 与索引映射

Level 2（条件与广播）:
  demo_13_2_practice_where_and_mask
  目标：掌握 where、布尔索引、广播批量加分

Level 3（绘图表达）:
  demo_13_3_practice_plot_design
  目标：同一份数据用折线/散点/柱状图表达不同问题

Level 4（多图整合）:
  demo_13_4_practice_subplot_dashboard
  目标：把分析结论组织成 2x2 仪表盘

Level 5（完整实战）:
  demo_14_1_project_sales_pipeline
  目标：NumPy 数据处理 -> Matplotlib 可视化 -> 输出结论
============================================================
"""


def demo_13_1_practice_axis_and_stats():
    """练习1：axis/聚合/最值索引映射（附详细解析）"""
    print("\n=== 练习1：axis + 聚合 + 最值索引 ===")

    # 场景：5名学生，4门课
    # 行 = 学生，列 = 科目
    subjects = np.array(["语文", "数学", "英语", "物理"])
    students = np.array(["张三", "李四", "王五", "赵六", "小明"])
    scores = np.array([
        [86, 91, 77, 88],
        [72, 84, 90, 79],
        [95, 88, 92, 94],
        [81, 76, 85, 83],
        [89, 93, 80, 87]
    ])

    print("成绩矩阵:\n", scores)
    print("shape =", scores.shape, "（行=学生数, 列=科目数）")

    # 解析1：每个学生平均分 -> 按行统计，因此 axis=1
    stu_mean = scores.mean(axis=1)
    print("\n[解析1] 每个学生平均分（axis=1）:")
    for i, name in enumerate(students):
        print(f"{name}: {stu_mean[i]:.2f}")

    # 解析2：每门课平均分 -> 按列统计，因此 axis=0
    sub_mean = scores.mean(axis=0)
    print("\n[解析2] 每门课平均分（axis=0）:")
    for i, sub in enumerate(subjects):
        print(f"{sub}: {sub_mean[i]:.2f}")

    # 解析3：每门课最高分对应的学生索引
    best_idx = np.argmax(scores, axis=0)
    print("\n[解析3] 每门课最高分同学（argmax 返回索引，不是分数）:")
    for j, idx in enumerate(best_idx):
        print(f"{subjects[j]}: {students[idx]}（{scores[idx, j]}分）")

    print("\n练习总结：先判断“按行还是按列”，再决定 axis。")


def demo_13_2_practice_where_and_mask():
    """练习2：where/布尔索引/广播（附易错点）"""
    print("\n=== 练习2：where + 布尔索引 + 广播 ===")

    scores = np.array([
        [58, 76, 89, 91],
        [62, 55, 84, 73],
        [95, 87, 92, 88],
        [41, 67, 72, 60]
    ])
    print("原始分数:\n", scores)

    # 任务1：及格标记（>=60 为 1，否则 0）
    passed = np.where(scores >= 60, 1, 0)
    print("\n[任务1答案] 及格标记矩阵:\n", passed)

    # 任务2：抓取所有低于60的分数
    low = scores[scores < 60]
    print("\n[任务2答案] 所有不及格分数:", low)

    # 任务3：曲线加分（不同科目加不同分）
    # 广播解释：shape (4,4) + (4,) -> 每一行都加同一个 1D 向量
    bonus = np.array([2, 3, 1, 0])
    new_scores = np.clip(scores + bonus, 0, 100)
    print("\n[任务3答案] 科目加分后:\n", new_scores)

    # 易错点说明：where 不是短路求值
    print("\n易错提示：np.where(cond, A, B) 会先计算 A 和 B，再按 cond 选值。")
    print("复杂递归分支请优先使用 if/else，不要直接塞进 where。")


def demo_13_3_practice_plot_design():
    """练习3：同一数据的三种图形表达与解析"""
    print("\n=== 练习3：一份数据画三种图 ===")

    np.random.seed(7)
    days = np.arange(1, 15)
    visits = np.random.randint(120, 280, size=14)
    conversion = np.round(np.random.uniform(0.03, 0.11, size=14), 3)
    sales = (visits * conversion * 38).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("同一组业务数据的三种表达方式", fontsize=14, fontweight="bold")

    # 图1：趋势问题用折线
    axes[0].plot(days, visits, "o-", color="#1f77b4", linewidth=2, markersize=5)
    axes[0].set_title("访问量趋势（折线图）")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Visits")
    axes[0].grid(True, alpha=0.3)

    # 图2：相关性问题用散点
    axes[1].scatter(visits, sales, c=conversion, cmap="viridis", s=90, edgecolors="white")
    axes[1].set_title("访问量 vs 销售额（散点图）")
    axes[1].set_xlabel("Visits")
    axes[1].set_ylabel("Sales")
    axes[1].grid(True, alpha=0.3)

    # 图3：类别对比用柱状
    week1 = sales[:7].sum()
    week2 = sales[7:].sum()
    axes[2].bar(["Week1", "Week2"], [week1, week2], color=["#ff7f0e", "#2ca02c"])
    axes[2].set_title("周销售额对比（柱状图）")
    axes[2].set_ylabel("Sales")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n解析：")
    print("1) 看趋势 -> 折线；2) 看相关 -> 散点；3) 看阶段对比 -> 柱状。")
    print("先明确你要回答的问题，再决定图形类型。")


def demo_13_4_practice_subplot_dashboard():
    """练习4：2x2 子图整合（含布局与标注细节）"""
    print("\n=== 练习4：2x2 仪表盘整合 ===")

    np.random.seed(123)
    x = np.arange(1, 21)
    y = 50 + np.cumsum(np.random.randn(20) * 2 + 0.8)
    z = np.random.randint(20, 100, size=20)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("多图整合训练", fontsize=15, fontweight="bold")

    axes[0, 0].plot(x, y, color="royalblue", linewidth=2)
    axes[0, 0].set_title("趋势图")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].hist(z, bins=8, color="teal", alpha=0.8, edgecolor="white")
    axes[0, 1].set_title("分布图")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].scatter(x, z, s=80, c=z, cmap="coolwarm", edgecolors="white")
    axes[1, 0].set_title("关系图")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].bar(["A", "B", "C", "D"], [z[:5].mean(), z[5:10].mean(), z[10:15].mean(), z[15:].mean()],
                   color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    axes[1, 1].set_title("分组对比")
    axes[1, 1].grid(axis="y", alpha=0.25)

    # 布局解析：tight_layout 负责子图间距；rect 为 suptitle 预留空间
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n布局解析：")
    print("- 先画内容，再统一标题/网格风格。")
    print("- suptitle + tight_layout(rect=...) 一起用，避免顶端被挤压。")


# ================================================================
# 第十四部分：完整小项目（NumPy -> 分析 -> 绘图 -> 结论）
# ================================================================

def demo_14_1_project_sales_pipeline():
    """完整实战：模拟业务数据并产出分析结论"""
    print("\n=== 完整小项目：销售数据分析管线 ===")

    np.random.seed(2026)
    days = np.arange(1, 31)

    # 1) 数据生成：访问量、转化率、客单价
    visits = np.random.randint(300, 900, size=30)
    conversion = np.round(np.random.uniform(0.02, 0.09, size=30), 4)
    unit_price = np.random.randint(35, 80, size=30)

    # 2) NumPy 向量化计算
    orders = np.round(visits * conversion).astype(int)
    revenue = orders * unit_price
    ad_cost = np.random.randint(3000, 9000, size=30)
    roi = np.round((revenue - ad_cost) / ad_cost, 3)

    # 3) 指标提炼
    total_revenue = revenue.sum()
    avg_roi = roi.mean()
    best_day = np.argmax(revenue)
    worst_day = np.argmin(revenue)

    print(f"总营收: {total_revenue}")
    print(f"平均 ROI: {avg_roi:.3f}")
    print(f"营收最高日: Day{days[best_day]}（{revenue[best_day]}）")
    print(f"营收最低日: Day{days[worst_day]}（{revenue[worst_day]}）")

    # 4) 绘图表达
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("30天销售分析 Dashboard", fontsize=16, fontweight="bold")

    axes[0, 0].plot(days, revenue, "o-", color="#2b8cbe", linewidth=2, markersize=4)
    axes[0, 0].axhline(revenue.mean(), color="gray", linestyle="--", linewidth=1,
                       label=f"mean={revenue.mean():.0f}")
    axes[0, 0].set_title("日营收趋势")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Revenue")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.25)

    sc = axes[0, 1].scatter(visits, revenue, c=conversion, cmap="plasma", s=85, edgecolors="white")
    axes[0, 1].set_title("访问量 vs 营收")
    axes[0, 1].set_xlabel("Visits")
    axes[0, 1].set_ylabel("Revenue")
    axes[0, 1].grid(True, alpha=0.25)
    plt.colorbar(sc, ax=axes[0, 1], label="Conversion")

    axes[1, 0].bar(days, roi, color=np.where(roi >= 0, "#31a354", "#de2d26"))
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("ROI 每日表现（绿=盈利，红=亏损）")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("ROI")
    axes[1, 0].grid(axis="y", alpha=0.25)

    weekly = np.array([revenue[:7].sum(), revenue[7:14].sum(), revenue[14:21].sum(), revenue[21:28].sum(), revenue[28:].sum()])
    axes[1, 1].bar(["W1", "W2", "W3", "W4", "W5"], weekly, color="#756bb1")
    axes[1, 1].set_title("分周营收")
    axes[1, 1].set_ylabel("Revenue")
    axes[1, 1].grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 5) 结论模板（练习写分析报告）
    print("\n结论模板（你可以按这个格式写报告）:")
    print("1. 整体趋势：营收是否稳步上升，波动区间是多少。")
    print("2. 驱动关系：访问量与营收相关性是否明显。")
    print("3. 风险识别：ROI 为负的日期有哪些，是否集中出现。")
    print("4. 行动建议：提高转化率/降低获客成本哪个优先。")


# ================================================================
# 第十五部分：高频报错速查（补充版）
# ================================================================

TROUBLESHOOTING_MAP = """
============================================================
高频报错速查（NumPy + Matplotlib）
============================================================
1) ValueError: operands could not be broadcast together
   原因：形状不兼容
   排查：先 print(a.shape, b.shape)，从尾维开始对齐检查

2) IndexError: index X is out of bounds
   原因：索引越界（常见于 for 循环边界）
   排查：先看 len/shape，再确认是否把 <= 写成 <

3) TypeError: ufunc ... did not contain a loop with signature ...
   原因：数组 dtype 变成字符串/object
   排查：print(arr.dtype)，必要时 arr = arr.astype(float)

4) AttributeError: module 'matplotlib' has no attribute 'plot'
   原因：写成 import matplotlib as plt
   修复：import matplotlib.pyplot as plt

5) 图没显示或空白
   原因：未调用 plt.show() 或 savefig 顺序错误
   修复：先 savefig 再 show；脚本末尾必须 show

6) 中文乱码或负号显示异常
   修复：
   plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
   plt.rcParams['axes.unicode_minus'] = False

7) tight_layout 对 3D 子图效果差
   修复：改用 fig.subplots_adjust(...) 手动调边距
============================================================
"""


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  NumPy + Matplotlib 绘图完全教程")
    print("=" * 60)
    print()
    print("可运行的示例函数：")
    demos = [name for name in sorted(dir()) if name.startswith('demo_')]
    for d in demos:
        print(f"  {d}()")
    print()
    print("使用方式：取消注释要运行的 demo 行，然后执行：")
    print('  python "NumPy与Matplotlib绘图完全教程.py"')
    print()
    print("推荐起始路径：")
    print("  1. demo_1_1_list_vs_numpy  → 理解为什么要用 NumPy")
    print("  2. demo_2_1_create         → 数组创建方法")
    print("  3. demo_3_1_first_plot     → 第一张图")
    print("  4. demo_5_1_parametric_curves → 参数方程曲线")
    print("  5. demo_7_1_3d_surfaces    → 3D 曲面")
    print("  6. demo_11_dashboard       → 综合练习")
    print("  7. demo_13_1_practice_axis_and_stats → 练习1（axis）")
    print("  8. demo_13_2_practice_where_and_mask  → 练习2（where）")
    print("  9. demo_13_3_practice_plot_design     → 练习3（图形表达）")
    print(" 10. demo_13_4_practice_subplot_dashboard → 练习4（多图整合）")
    print(" 11. demo_14_1_project_sales_pipeline   → 完整小项目")
    print()
    print(PRACTICE_BOOK)
    print(BEST_PRACTICES)
    print(CHEAT_SHEET)
    print(TROUBLESHOOTING_MAP)
