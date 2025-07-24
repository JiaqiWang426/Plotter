import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
import matplotlib.ticker as ticker
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
        '''

bounds = [ #图像y轴的刻度点
        1e-6,      # 0.0001%
        1e-5,      # 0.001%
        1e-4,      # 0.01%
        1e-3,      # 0.1%
        1e-2,      # 1%
        1e-1,      # 10%
        0.3,       # 30%
        0.7,       # 70%
        0.9,       # 90%
        0.95,      # 95%
        0.99,      # 99%
        0.999,     # 99.9%
        0.9999,    # 99.99%
        0.99999,   # 99.999%
        0.999999   # 99.9999%
    ]

class PiecewiseLinearTransform(Transform):
    input_dims = output_dims = 1
    is_separable = True
    def __init__(self, bounds):
        super().__init__()
        self.b   = np.array(bounds)
        self.n   = len(self.b) - 1
        self.seg = 1.0 / self.n

    def transform_non_affine(self, y):
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
        super().__init__(axis)      # ← 把 axis 传给父类
        self.bounds = kwargs['bounds']

    def get_transform(self):
        return PiecewiseLinearTransform(self.bounds)

    def set_default_locators_and_formatters(self, axis):
        
        axis.set_major_locator(ticker.FixedLocator(self.bounds))
        axis.set_major_formatter(ticker.FixedFormatter([
            "0.0001%","0.001%","0.01%","0.1%","1%","10%",
            "30%","70%","90%","95%","99%","99.9%",
            "99.99%","99.999%","99.9999%"
        ]))

register_scale(PiecewiseScale)

def clipboard_to_array(root, round_input):
    #将粘贴板中的数据转换为数字组成的阵列
    #自动跳过数据中的字符串
    clipboard = root.clipboard_get().strip('\n').split("\n")#去掉\n
    rows = []
    for raw_row in clipboard:
        cleaned_row = []
        for x in raw_row.split('\t'):
            try:
                cleaned_row.append(float(x))
            except ValueError:
                cleaned_row.append(-1e30) #用-1e30作为占位符填充数据失真
                #这里默认数据中不会出现-1e30
        if not cleaned_row:
            continue
        rows.append(np.round(cleaned_row, round_input))
    cols = [list(col) for col in list(zip(*rows))]
    return cols

def count_sort(Arr, n): #比较第n位数
    count = [[] for _ in range(10)]
    for num in Arr:
        nth_digit = (num // 10 ** (n)) % 10
        count[nth_digit].append(num)
    output = []
    for digit in range(10):
        output.extend(count[digit])
    return output

def radix_sort(Arr, round_input):
    scaled = [int((10 ** round_input) * num) for num in Arr]
    n = 0
    maximum = max(scaled)
    while maximum // (10 ** n) > 0:
        scaled = count_sort(scaled, n) 
        n += 1
    output = [num / (10 ** round_input) for num in scaled]
    return output

def filter(Arr, placeholder = -1e30):
    return [x for x in Arr if x != placeholder]

def cum_prob(Arr):#计算累计概率, 此时Arr经过radix sort，是有序的
    #在这里再算一下z值
    N = len(Arr)
    mean = np.mean(Arr)
    std = np.std(Arr)
    freq = {}
    if N >= 1:
        if Arr[0] == Arr[1]:
            Arr[0] -= 1e-10 #小尾巴
    for item in Arr:
        freq[item] = freq.get(item, 0) + 1#计数
    dic = {}
    cumulative = 0
    for num in freq.keys():
        cum_probability = (freq[num] + cumulative) / N
        z = z_value(num, mean, std)
        dic[num] = [cum_probability, z]
        cumulative += freq[num]
    maximum = list(freq.keys())[-1]
    dic[maximum].append(N)#dic的最后一个数据的params会多储藏一个值N
    return dic

def z_value(x, mean, sigma):
    if sigma != 0:
        return (x - mean) / sigma
    else:
        return "Incorrect standard variation"

#[list(np.cumsum(row)) for row in data]
def top_bottom_amplifier(y: float) -> float:
    """
    将 y (0~1) 映射到 [0,1] 上的分段等长区间，用于放大两端、压缩中间。
    y: 原始概率值，比如 1e-6 ~ 0.999999
    返回值: 映射后的位置（0~1）
    """
    global bounds
    n_segs = len(bounds) - 1
    out_seg = 1.0 / n_segs     # 每段输出长度 = 1/14

    #边缘处理：y ≤ 最小、y ≥ 最大
    if y <= bounds[0]:
        return 0.0
    if y >= bounds[-1]:
        return 1.0

    #找出 y 属于哪个区间 [bounds[i], bounds[i+1])
    for i in range(n_segs):
        lo, hi = bounds[i], bounds[i+1]
        if lo <= y < hi:
            #该段仿射映射：f(y) = offset + slope*(y - lo)
            slope  = out_seg / (hi - lo)
            offset = i * out_seg
            return offset + (y - lo) * slope

    # （理论上不可能跑到这里）
    return float('nan')

def plot_distribution(data, new_figure = True): #针对矩阵：内含多个字典
#将读取到的数据进行绘图
    global bounds
    overall_max_x = max(max(dic.keys()) for dic in data)
    if new_figure:
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.set_yscale('piecewise', bounds=bounds)
    else:
        ax1 = plt.gca()
        fig = ax1.figure
        if ax1.get_yscale() != 'piecewise':
            ax1.set_yscale('piecewise', bounds=bounds)
        overall_max_x = max(overall_max_x, ax1.get_xlim()[1])
    plt.rcParams['font.sans-serif'] = ['SimSun']
# 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    ax1.margins(y=0.08)  
    ax1.set_title("Title标题", fontweight = 'bold')
    ax1.set_xlabel("x值", fontweight = 'bold')
    ax1.set_ylabel("Cum Prob 累计概率", fontweight = 'bold')
    ax1.set_ylim(bounds[0], bounds[-1]) 

    lines = []
    
    for idx, dic in enumerate(data):
        x= list(dic.keys())
        params   = list(dic.values())#分别是(cum_prob, z)
        total = len(x)
        minimum, maximum = round(x[0], 3), x[-1]
        med = total // 2
        median = x[med]
        y = list(p[0] #放大后的y，用于映射到图例上
                   for p in params) #累计概率曲线
        z = [p[1] for p in params]
        N = params[-1][-1]#dic的最后一个数据的params会多一个值N
        if total > 31: #画大概三十多个点
            points_to_show_x = [x[int(total // 31 * i)]
                                 for i in range(1, 31)]
            if x[1] not in points_to_show_x:
                points_to_show_x = [x[1]] + points_to_show_x
            points_to_show_x = [x[0]] + points_to_show_x
            if x[-1] not in points_to_show_x:
                points_to_show_x.append(x[-1])
            points_to_show_y = [y[int(total // 31 * i)]
                                 for i in range(1, 31)]
            if y[1] not in points_to_show_y:
                points_to_show_y = [y[1]] + points_to_show_y
            points_to_show_y = [y[0]] + points_to_show_y
            if y[-1] not in points_to_show_y:
                points_to_show_y.append(y[-1])
            points_to_show_z = [z[int(total // 31 * i)]
                                 for i in range(1, 31)]
            if z[1] not in points_to_show_z:
                points_to_show_z = [z[1]] + points_to_show_z
            points_to_show_z = [z[0]] + points_to_show_z
            if z[-1] not in points_to_show_z:
                points_to_show_z.append(z[-1])
        else:
            points_to_show_x, points_to_show_y, points_to_show_z = x, y, z
        label1 = (rf"$\mathbf{{data{idx+1}}}$" + "\n"
    f"N={N}, med={median}, \nmax={maximum}, min={minimum}")
        line, =ax1.plot(points_to_show_x, points_to_show_y,
                         marker = 'o', linewidth = 1, label = label1) 
        line.zs = points_to_show_z
        lines.append(line)
    ax1.set_xlim(0, overall_max_x)
    ax1.set_xticks(np.linspace(0, overall_max_x, 11))        
    if new_figure:
        # first draw: show just these lines
        leg = ax1.legend(
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 10}
        )
    else:
        # overlay: gather every Line2D that's on ax1, pull its label,
        # and re‑draw the legend so we keep any edits
        all_lines  = ax1.get_lines()
        all_labels = [ln.get_label() for ln in all_lines]
        leg = ax1.legend(
            all_lines, all_labels,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 10}
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
        annot.set_text(f"Val={xi:.3f}\n"
                       f"Cum_prob={100 * yi:.1f}%\n"
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
    fig.subplots_adjust(right=0.75)
    plt.show(block = False)
    return

def plot_Z(data, new_figure = True):
    overall_max_x = max(max(dic.keys()) for dic in data)
    if new_figure:
        fig, ax2 = plt.subplots(figsize = (8,5))
    else:
        ax2 = plt.gca()
        fig = ax2.figure
        overall_max_x = max(overall_max_x, ax2.get_xlim()[1])
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    ax2.margins(y=0.08) 
    ax2.set_title("Title标题", fontweight = "bold")
    ax2.set_xlabel("x值", fontweight = 'bold')
    ax2.set_ylabel("Z value Z值(\u03C3)", fontweight = 'bold')
    ax2.set_ylim(-4, 4)

    lines = []
    for idx, dic in enumerate(data):
        x= list(dic.keys())
        params = list(dic.values())
        total, minimum, maximum= len(x), round(x[0], 3), x[-1]
        med = total // 2
        median = x[med]
        N = params[-1][-1]
        y = [p[1] for p in params] #z值曲线
        z = [p[0] for p in params]
        if total > 31: #希望只画出31个点
            points_to_show_x = [x[int(total // 31 * i)]
                                 for i in range(1, 31)]
            if x[1] not in points_to_show_x:
                points_to_show_x = [x[1]] + points_to_show_x
            points_to_show_x = [x[0]] + points_to_show_x
            if x[-1] not in points_to_show_x:
                points_to_show_x.append(x[-1])
            points_to_show_y = [y[int(total // 31 * i)]
                                 for i in range(1, 31)]
            if y[1] not in points_to_show_y:
                points_to_show_y = [y[1]] + points_to_show_y
            points_to_show_y = [y[0]] + points_to_show_y
            if y[-1] not in points_to_show_y:
                points_to_show_y.append(y[-1])
            points_to_show_z = [z[int(total // 31 * i)]
                                 for i in range(1, 31)]
            if z[1] not in points_to_show_z:
                points_to_show_z = [z[1]] + points_to_show_z
            points_to_show_z = [z[0]] + points_to_show_z
            if z[-1] not in points_to_show_z:
                points_to_show_z.append(z[-1])
        else:
            points_to_show_x, points_to_show_y, points_to_show_z = x, y, z
        label2 = (rf"$\mathbf{{data{idx+1}}}$" + "\n"
    f"N={N}, med={median}, \nmax={maximum}, min={minimum}")
        line, = ax2.plot(points_to_show_x, points_to_show_y,
                          marker = 's',
                          linewidth = 1, label = label2)
        line.zs = points_to_show_z
        lines.append(line)
    ax2.set_xlim(0, overall_max_x)
    ax2.set_xticks(np.linspace(0, overall_max_x, 11))  
    if new_figure:
        # first draw: show just these lines
        leg = ax2.legend(
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 10}
        )
    else:
        # overlay: gather every Line2D that's on ax1, pull its label,
        # and re‑draw the legend so we keep any edits
        all_lines  = ax2.get_lines()
        all_labels = [ln.get_label() for ln in all_lines]
        leg = ax2.legend(
            all_lines, all_labels,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 10}
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
        annot.set_text(f"Val={xi:.3f}\n"
                       f"Z={yi:.3f}\u03C3\n"
                       f"Cum Prob={100 * zi:.1f}%")
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
    fig.subplots_adjust(right=0.75)
    plt.show(block = False)
    return

def main(root):
    first_plot = True
    round_input = None
    # 先让用户输入保留小数位
    while True:
        s = input("输入数据要保留几位小数：")
        try:
            round_input = int(s)
            break
        except ValueError:
            print("请输入整数！")

    while True:
        # 读取并处理数据
        data_cols = clipboard_to_array(root, round_input)
        data_dicts = [cum_prob(filter(radix_sort(col, round_input))) for col in data_cols]
        if first_plot: # 决定这次 new_figure
            new_fig = True
        else:
            ans = input("是否在新图上绘制？(y=新图 / n=叠加旧图)：").strip().lower()
            new_fig = (ans == 'y')
        plot_distribution(data_dicts, new_figure=new_fig)
        first_plot = False
        cont = input("还要再画一组数据吗？(y/n)：").strip().lower() # 是否继续绘制
        if cont != 'y':
            break # 最后一次阻塞式显示，防止脚本直接退出
    plt.show()

def test():
    print("\033[1m这是加粗的文字\033[0m")
    return
    
if __name__ == "__main__":
    test_input = input("输入1测试：")
    if test_input:
        test()
    else:
        main()
    
    

path = r"C:\Users\1000405157\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\pyinstaller.exe --onefile plot.py"


'''
ticks1 = [top_bottom_amplifier(_) for _ in bounds]
    labels1 = [
        "0.0001%",
        "0.001%",
        "0.01%",
        "0.1%",
        "1%",
        "10%",
        "30%",
        "70%",
        "90%",
        "95%",
        "99%",
        "99.9%",
        "99.99%",
        "99.999%",
        "99.9999%"
    ]
    ax1.set_yticks(ticks1)
    ax1.set_yticklabels(labels1)    
'''