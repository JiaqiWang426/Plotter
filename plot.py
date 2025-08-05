import numpy as np
from scipy.stats import norm
import re
'''
项目1：python画图脚本
输入：手动输入csv中的一列或多列，每一格的数据
输出：将概率进行累加，并将每一列的数据绘制成不同颜色的累计分布图。由于概率之间差异较小，比如良品率是99%与总概率100%接近，考虑用指数函数进行放大以更直观

日志：汪嘉祺
7.16: 读取粘贴板数据，整合成阵列，绘折现图，自定义图像标签
7.17：将所有数据四舍五入到两位小数，通过基数排序（radix sort）来排序数值大小，计算数的频率以及累计概率
7:18：通过分段函数放大处于累计概率头部和底部的数据，压缩累计概率处于中间的数据。优化读取剪切板这一步的逻辑加快运行速度
      自定义四舍五入到几位小数，过滤了空白格等数据失真。增加了自定义是否在同一张图上继续绘制曲线的功能。
7.20：增加了第二个y轴：z值。增加了数据点z值的计算。
7.21：完善z值逻辑，删掉了第二个y轴。完善了gui。增加了可互动的悬浮式小窗，直接显示数据点的x，y值。
7.22：修复了累计概率和z值可能会被plot到同一张图上的bug。通过最小值分离开来增加了小尾巴。
      修复了窗口边框挡住悬浮小窗的bug。将文字改为中文。当数据点数过多时，会选择31个点来绘制曲线，避免拥挤。
      将labels移动到图像外边，显示最大最小值总数中位数。加粗了label中data{idx + 1}的字体
7.23：累计概率浮动小窗增加了z值，z值浮动小窗增加了累计概率。修复了因top_bottom_amplify
      放缩y值导致的失真。
7.24：增加了网格。调整x轴范围至0.9最小值到1.1最大值。修复了radix sort无法解决负数
    正数混合的问题。禁止了输入为1个值或常值导致无法计算Z值的特殊情况。
7.25：禁止了输入为0个值导致无法计算Z值的特殊状况。修正了中位数的计算。修正了输入数据
    量过少导致网格与刻度挤在一起的bug。通过懒加载加快了程序gui的开启时间。
        '''

bounds = [ #图像y轴的刻度点
        1e-6,      # 0.0001%
        1e-5,      # 0.001%
        1e-4,      # 0.01%
        1e-3,      # 0.1%
        1e-2,      # 1%
        1e-1,      # 10%
        0.3,       # 30%
        0.5,       # 50%
        0.7,       # 70%
        0.9,       # 90%
        0.95,      # 95%
        0.99,      # 99%
        0.999,     # 99.9%
        0.9999,    # 99.99%
        0.99999,   # 99.999%
        0.999999,   # 99.9999%
        1.05
    ]

# —— 懒加载matplotlib, 定义y轴放缩函数 —— #
def _mpl():
    """按需导入 matplotlib 及自定义 Scale，只调用一次"""
    global plt
    try:
        return plt            # 已经导过，直接返回
    except NameError:
        pass
    import matplotlib.pyplot as _plt
    from matplotlib.scale import ScaleBase, register_scale
    from matplotlib.transforms import Transform
    import matplotlib.ticker as ticker

    # —— 对于CDF图的y轴刻度的分段等距放大 —— #
    class PiecewiseLinearTransform(Transform):
        input_dims = output_dims = 1 #只处理一维数据
        is_separable = True
        def __init__(self, bounds):
            super().__init__()
            self.b   = np.array(bounds) #刻度
            self.n   = len(self.b) - 1 #段数
            self.seg = 1.0 / self.n #每段长度

        def transform_non_affine(self, y):#连续的分段放射变换
            y   = np.array(y)
            out = np.zeros_like(y, dtype=float)
            out[y >= self.b[-1]] = 1.0
            mask = (y > self.b[0]) & (y < self.b[-1])
            ys   = y[mask]
            idx  = np.searchsorted(self.b, ys) - 1
            lo, hi = self.b[idx], self.b[idx+1]
            out[mask] = idx*self.seg + (ys - lo)/(hi - lo)*self.seg
            return out

        def inverted(self):
            return InvertedPiecewiseTransform(self.b)

    class InvertedPiecewiseTransform(Transform):
        input_dims = output_dims = 1
        is_separable = True
        def __init__(self, bounds):
            super().__init__()
            self.b   = np.array(bounds)
            self.n   = len(self.b) - 1
            self.seg = 1.0 / self.n

        def transform_non_affine(self, y):
            y   = np.array(y)
            out = np.empty_like(y, dtype=float)
            out[y <=   0 ] = self.b[0]
            out[y >=   1 ] = self.b[-1]
            mask = (y > 0) & (y < 1)
            ym   = y[mask]
            idx  = np.minimum((ym/self.seg).astype(int), self.n-1)
            lo, hi = self.b[idx], self.b[idx+1]
            out[mask] = lo + (ym - idx*self.seg)*(hi - lo)/self.seg
            return out

    class PiecewiseScale(ScaleBase):
        name = 'piecewise'
        def __init__(self, axis, **kwargs):
            super().__init__(axis)      # 把 axis 传给父类
            self.bounds = kwargs['bounds']

        def get_transform(self):
            return PiecewiseLinearTransform(self.bounds)

        def set_default_locators_and_formatters(self, axis):
            
            axis.set_major_locator(ticker.FixedLocator(self.bounds))
            axis.set_major_formatter(ticker.FixedFormatter([
                "0.0001%","0.001%","0.01%","0.1%","1%","10%",
                "30%", "50%", "70%","90%","95%","99%","99.9%",
                "99.99%","99.999%","99.9999%"
            ]))

    register_scale(PiecewiseScale)

    globals()["plt"] = _plt          #写回全局
    return _plt

def clipboard_to_array(root, round_input = None):
    #将粘贴板中的数据转换为数字组成的阵列
    #自动跳过数据中的字符串
    clipboard = root.clipboard_get().strip('\n').split("\n")#去掉\n
    rows = []
    for idx, raw_row in enumerate(clipboard):
        cleaned_row = []
        if idx == 0: #检测第一行是否含有名字（无法转换为float的字符串）
            for x in raw_row.split('\t'):
                try:
                    if round_input is None:
                        cleaned_row.append(float(x))
                    else:
                        cleaned_row.append(np.round(float(x), round_input))    
                except ValueError:
                    if isinstance(x, str):
                        cleaned_row.append(x)
        else:      
            for x in raw_row.split('\t'):
                try:
                    if round_input is None:
                        cleaned_row.append(float(x))
                    else:
                        cleaned_row.append(np.round(float(x), round_input))
                except ValueError:
                    cleaned_row.append(-1e30) #用-1e30作为占位符填充数据失真
                    #这里默认数据中不会出现-1e30
        if not cleaned_row:
            continue
        rows.append(cleaned_row)
    try: 
        rows[1]
    except IndexError:
        raise ValueError #ValueError会触发showerror，见plot_gui.py
    cols = [list(col) for col in list(zip(*rows))]
    data_cols = [DataColumn(col, idx, round_input) #datacolumn见后面
             for idx, col in enumerate(cols)]

    return data_cols


def count_sort(Arr, n): #比较第n位数
    count = [[] for _ in range(10)]
    for num in Arr:
        nth_digit = (num // 10 ** (n)) % 10
        count[nth_digit].append(num)
    output = []
    for digit in range(10):
        output.extend(count[digit])
    return output

def radix_sort(Arr, round_input: int):#round_input是四舍五入到几位数
    scaled = [int((10 ** round_input) * num) for num in Arr]
    n = 0
    if len(scaled) != 0:
        maximum = max(scaled)
    else:
        return []
    while maximum // (10 ** n) > 0:
        scaled = count_sort(scaled, n) 
        n += 1
    output = [num / (10 ** round_input) for num in scaled]
    return output

def radix_sort_with_negative(Arr, round_input: int):
    Arr_positive, Arr_negative = [], []
    for num in Arr:
        if num >= 0:
            Arr_positive.append(num)
        else:
            Arr_negative.append(num)
    return [-1 * u for u in list(reversed(radix_sort([-1 * v for v in Arr_negative], round_input)))] + radix_sort(Arr_positive, round_input)

def cum_prob(Arr):#计算累计概率, 此时Arr经过radix sort，是有序的
    N = len(Arr)
    std = np.std(Arr, ddof = 1)
    if np.isnan(std) or std <= 0: #这一步是检查输入是否合法（能不能算出正的标准差）
        raise ValueError #valueerror会触发showerror
    freq = {}
    for item in Arr:
        freq[item] = freq.get(item, 0) + 1#计数
    dic = {}
    cumulative = 0
    n = len(list(freq.keys()))
    # —— 计算累计概率（见readme） —— #
    for index in range(n // 2):
        key = list(freq.keys())[index]
        cum_probability = (freq[key] + cumulative) / N
        z = norm.ppf(cum_probability) #根据累计概率从正太分布对应z值
        dic[key] = [cum_probability, z]
        cumulative += freq[key]
    for index in range(n // 2, n):
        key = list(freq.keys())[index]
        cum_probability = (cumulative) / N
        z = norm.ppf(cum_probability)
        dic[key] = [cum_probability, z]
        cumulative += freq[key]
    return dic

# —— 将剪切板中的数据（列）转换成这个class —— #
class DataColumn:
    PLACEHOLDER = -1e30        #剪切板中不合法的数据点（比如字符串）将被替换为这个
    def __init__(self, raw_col: list[float | str], idx: int, round_input = None):
        first = raw_col[0]
        self.name = (f"data{idx+1}"
                     if isinstance(first, (int, float))
                     else str(first).strip()) #检测是不是名字
        nums = (raw_col[1:] if self.name != f"data{idx+1}" else raw_col)#数据部分array
        self.values = [float(x) for x in nums if x != self.PLACEHOLDER]#过滤掉占位符的真实数据
        self.N   = len(self.values)
        self.med = (np.median(self.values) #中位数
                    if self.N else float("nan"))
        if round_input is None:
            sorted_vals = sorted(self.values)
        else:
            sorted_vals = radix_sort_with_negative(self.values, round_input)
        self.stats  = cum_prob(sorted_vals) #包含累计概率和z值

def _tex_escape(s: str) -> str:
    # 转义 mathtext 特殊字符
    return re.sub(r'([_$%#&^{}])', r'\\\1', s)

# —— 图例（legend）格式 —— #
def _label(dc: DataColumn, min_, max_):
    safe_name = _tex_escape(dc.name)
    return (rf"$\mathbf{{{safe_name}}}$" + "\n"
            f"N={dc.N}, med={dc.med},\n"
            f"max={max_}, min={min_}")

# —— 绘制CDF —— #
def plot_distribution(data_cols: list[DataColumn],
                       new_figure = True, 
    grid_var = True): #针对矩阵：内含多个字典
#将读取到的数据进行绘图
    plt = _mpl()
    global bounds
    overall_max_x = max(max(dc.stats.keys()) for dc in data_cols)
    overall_min_x = min(min(dc.stats.keys()) for dc in data_cols)
    if new_figure:
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.set_yscale('piecewise', bounds=bounds)
    else:
        ax1 = plt.gca()
        fig = ax1.figure
        if ax1.get_yscale() != 'piecewise':
            ax1.set_yscale('piecewise', bounds=bounds)
        overall_max_x = max(overall_max_x, ax1.get_xlim()[1])
        overall_min_x = min(overall_min_x, ax1.get_xlim()[0])
        import matplotlib.ticker as mticker
        ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.rcParams['font.sans-serif'] = ['Calibrius','SimSun']
# 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    ax1.margins(y=0.08)  
    ax1.set_title("Title", fontweight = 'bold')
    ax1.set_xlabel("Value", fontweight = 'bold')
    ax1.set_ylabel("Cum Prob", fontweight = 'bold')
    ax1.set_ylim(bounds[0], bounds[-1]) 

    lines = []
    
    for dc in data_cols:
        x        = list(dc.stats.keys())
        cum, z   = zip(*dc.stats.values())      
        label    = _label(dc, min(x), max(x))

        line, = ax1.plot(x, cum,
                         marker='o',
                         linewidth=1,
                         label=label)
        line.zs = z
        lines.append(line)
    
    scalar = (overall_max_x - overall_min_x) * 0.1 

    ax1.set_xlim(overall_min_x - scalar, overall_max_x + scalar)
    ax1.set_xticks(np.linspace(overall_min_x - scalar,
                                overall_max_x + scalar, 11
                                  ))

    ax1.tick_params(axis='x', rotation=45)
    
    if new_figure:
        # first draw: show just these lines
        leg = ax1.legend(
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        )
    else:
        # overlay: gather every Line2D that's on ax1, pull its label,
        # and re‑draw the legend so we keep any edits
        all_lines  = ax1.get_lines()
        all_labels = [ln.get_label() for ln in all_lines]
        leg = ax1.legend(
            all_lines, all_labels,
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        ) 
    for txt in leg.get_texts():
        txt.set_picker(True)
    annot = ax1.annotate(
        "", xy=(0,0), xytext=(15, -15), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        clip_on=False
    )
    annot.set_visible(False)

    def update_annot(line, ind):
        xdata, ydata, zdata = line.get_xdata(), line.get_ydata(), line.zs
        i = ind["ind"][0]
        xi, yi, zi = xdata[i], ydata[i], zdata[i]
        annot.xy = (xi, yi)
        annot.set_text(f"Val={xi}\n"
                       f"Cum_prob={100 * yi:.3f}%\n"
                       f"Z={zi:.3f}\u03C3")
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax1:
            for line in lines:
                cont, ind = line.contains(event)
                if cont:
                    update_annot(line, ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.tight_layout()
    plt.grid(grid_var, which='major', axis='both', )
    fig.subplots_adjust(right=0.75)
    plt.show(block = False)
    return

# —— 绘制Z值曲线，逻辑与CDF基本一样 —— #
def plot_Z(data_cols: list[DataColumn], new_figure = True, grid_var = True):
    plt = _mpl()
    overall_max_x = max(max(dc.stats.keys()) for dc in data_cols)
    overall_min_x = min(min(dc.stats.keys()) for dc in data_cols)
    overall_min_y = min(
    val[1] for dc in data_cols for val in dc.stats.values()
)
    overall_max_y = max(
    val[1] for dc in data_cols for val in dc.stats.values()
)
    if new_figure:
        fig, ax2 = plt.subplots(figsize = (8,5))
    else:
        ax2 = plt.gca()
        fig = ax2.figure
        overall_max_x = max(overall_max_x, ax2.get_xlim()[1])
        overall_min_x = min(overall_min_x, ax2.get_xlim()[0])
        overall_max_y = max(overall_max_y, ax2.get_ylim()[1])
        overall_min_y = min(overall_min_y, ax2.get_ylim()[0])
        import matplotlib.ticker as mticker
        ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.rcParams['font.sans-serif'] = ['Calibrius','SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    ax2.margins(y=0.08) 
    ax2.set_title("Title", fontweight = "bold")
    ax2.set_xlabel("Value", fontweight = 'bold')
    ax2.set_ylabel("Z value(\u03C3)", fontweight = 'bold')
    scalar = (overall_max_x - overall_min_x) * 0.1 
    ax2.set_xlim(overall_min_x - scalar, overall_max_x + scalar)
    ax2.set_xticks(np.linspace(overall_min_x - scalar,
                                overall_max_x + scalar, 11
                                  ))
    rounded_max = int(overall_max_y + 1)
    yticks = list(range(-rounded_max, rounded_max + 1)) 
    yticks_labels = [str(ytick) + "\u03C3" for ytick in yticks]
    ax2.set_yticks(ticks = yticks, labels = yticks_labels)
    ax2.tick_params(axis='x', rotation=45)

    lines = []
    for dc in data_cols:
        x        = list(dc.stats.keys())
        cum, z   = zip(*dc.stats.values())
        label    = _label(dc, min(x), max(x))
        line, = ax2.plot(x, z,
                        marker='s',
                        linewidth=1,
                        label=label)
        line.zs = cum
        lines.append(line)
        
    
    if new_figure:
        # first draw: show just these lines
        leg = ax2.legend(
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        )
    else:
        # overlay: gather every Line2D that's on ax1, pull its label,
        # and re‑draw the legend so we keep any edits
        all_lines  = ax2.get_lines()
        all_labels = [ln.get_label() for ln in all_lines]
        leg = ax2.legend(
            all_lines, all_labels,
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        ) 
    for txt in leg.get_texts():
        txt.set_picker(True)
    annot = ax2.annotate(
        "", xy=(0,0), xytext=(15, -15), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        clip_on=False
    )
    annot.set_visible(False)

    def update_annot(line, ind):
        xdata, ydata, zdata = line.get_xdata(), line.get_ydata(), line.zs
        i = ind["ind"][0]
        xi, yi, zi = xdata[i], ydata[i], zdata[i]
        annot.xy = (xi, yi)
        annot.set_text(f"Val={xi}\n"
                       f"Z={yi:.3f}\u03C3\n"
                       f"Cum Prob={100 * zi:.3f}%")
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax2:
            for line in lines:
                cont, ind = line.contains(event)
                if cont:
                    update_annot(line, ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.tight_layout()
    plt.grid(grid_var, which='major', axis='both', )
    fig.subplots_adjust(right=0.75)
    plt.show(block = False)
    return
    
#打包：
#cd C:\Users\1000405157\Desktop\Work\plot program
#python -m PyInstaller --noconsole --onefile plot_gui.py
#python -m PyInstaller --noconsole --onedir plot_gui.py
#"C:\Users\1000405157\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\pyinstaller.exe" --onedir --noconsole "C:\Users\1000405157\Desktop\Work\plot program\plot_gui.py"


#ax1.xaxis.set_major_locator(MaxNLocator(nbins=11, prune='both'))  # 最多11个主刻度

