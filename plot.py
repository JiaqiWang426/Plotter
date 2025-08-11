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
    global plt
    try:
        return plt#已经导过，直接返回
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
            for x in raw_row.split('\t'): #excel每行的数据是用\t隔开的
                try:
                    if round_input is None:
                        cleaned_row.append(float(x))
                    else:
                        cleaned_row.append(np.round(float(x), round_input))    
                        #四舍五入
                except ValueError:
                    if isinstance(x, str):#检测第一行是不是标题
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
    cols = [list(col) for col in list(zip(*rows))]#将rows转换成columns
    data_cols = [DataColumn(col, idx, round_input) #datacolumn见后面
             for idx, col in enumerate(cols)]

    return data_cols


def count_sort(Arr, n): #比较第n位数
    count = [[] for _ in range(10)]
    for num in Arr:
        nth_digit = (num // 10 ** (n)) % 10 #第n位数
        count[nth_digit].append(num)
    output = []
    for digit in range(10):
        output.extend(count[digit])
    return output

def radix_sort(Arr, round_input: int):#round_input是四舍五入到几位数
    scaled = [int((10 ** round_input) * num) for num in Arr] #转换为整数
    n = 0
    if len(scaled) != 0:
        maximum = max(scaled)
    else:
        return []
    while maximum // (10 ** n) > 0: #第n位数不为0
        scaled = count_sort(scaled, n) 
        n += 1
    output = [num / (10 ** round_input) for num in scaled]#还原回去
    return output

def radix_sort_with_negative(Arr, round_input: int):#处理负数的情况
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
        self.values = [float(x) for x in nums if x != self.PLACEHOLDER]#过滤掉占位符后的真实数据
        self.N   = len(self.values)
        self.med = (np.median(self.values) #中位数
                    if self.N else float("nan"))
        if round_input is None:
            sorted_vals = sorted(self.values)
        else:
            sorted_vals = radix_sort_with_negative(self.values, round_input)
        self.stats  = cum_prob(sorted_vals) #包含累计概率和z值

# —— 参考线部分：与曲线的交点以及图例 —— #  
def _short_label(lbl: str) -> str:
    """从多行/TeX标签里提取短名（图例第一行），用于交点清单。"""
    first = lbl.splitlines()[0]
    m = re.fullmatch(r'\$\\mathbf\{(.*?)\}\$', first)
    return m.group(1) if m else first

def find_y_given_x0(xdata, ydata, x0):
    #求给定x0，曲线中y的值。这里的xdata应该是严格递增的
    i = np.searchsorted(xdata, x0) - 1 #找到x0可以插入的索引（再减1）
    #或者说刚好比x0小的数的索引
    if i < 0 or i >= len(xdata) - 1:
        return None
    x1, x2 = xdata[i], xdata[i+1]
    y1, y2 = ydata[i], ydata[i+1]
    #x0会在(x1, y1), (x2, y2)组成的直线中
    if x2 == x1:#应该不需要考虑，因为xdata是严格递增的
        return y1
    k = (y2 - y1) / (x2 - x1) #斜率
    return y1 + k * (x0 - x1) #x0在该线段中的y值

def find_x_given_y0(xdata, ydata, y0):
    #求给定y0，曲线中x的值。这里的ydata应该是严格递增的
    #但由于累计概率计算方式比较奇葩，在中间的位置可能会出现y[med + 1] == y[med]
    xs = []
    for i in range(len(ydata) - 1):
        y1, y2 = ydata[i], ydata[i+1]
        if y1 == y0:
            xs.append(xdata[i])
        elif (y1 - y0) * (y2 - y0) < 0:
            t = (y0 - y1) / (y2 - y1)
            xs.append(xdata[i] + t * (xdata[i+1] - xdata[i]))
    return xs

def _get_data_lines(ax):
    """当前坐标轴上所有‘数据曲线’（排除参考线 _is_ref=True）。"""
    return [ln for ln in ax.get_lines() if not getattr(ln, "_is_ref", False)]

def _recompute_ref_labels(ax):
    """根据‘当前所有数据曲线’重新计算每条参考线的交点并更新它们的 label。"""
    if not hasattr(ax, "_ref_lines"):
        return
    data_lines = _get_data_lines(ax)
    for ref in list(ax._ref_lines):
        if ref.axes is not ax:
            continue
        # 需要知道参考线类型与取值
        is_x = getattr(ref, "_is_x", None)
        val  = getattr(ref, "_val", None)
        if is_x is None or val is None:
            continue

        pairs = []
        for ln in data_lines:
            xd, yd = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
            lab = _short_label(ln.get_label())
            if is_x:
                y = find_y_given_x0(xd, yd, val)
                if y is not None:
                    pairs.append(f"{lab}:({val:.4g},{y:.4g})")
            else:
                xs = find_x_given_y0(xd, yd, val)
                for xv in xs:
                    pairs.append(f"{lab}:({xv:.4g},{val:.4g})")

        base = (f"x = {val:g}" if is_x else f"y = {100 * val:g}%")
        ref.set_label(base + (" | " + "; ".join(pairs) if pairs else " | no intersections in range"))

def _refresh_ref_legend(ax, fig):
    """在右下角刷新参考线图例，并把主图例加回去防止覆盖。"""
    if not hasattr(ax, "_ref_lines"):
        ax._ref_lines = []
    if getattr(ax, "_ref_legend", None):#去掉旧的参考线图例#
        try:
            ax._ref_legend.remove()
        except Exception:
            pass
        ax._ref_legend = None
    if getattr(ax, "_main_legend", None):#把主图例加回去#
        ax.add_artist(ax._main_legend)

    handles = [ln for ln in ax._ref_lines if ln.axes is ax]
    if handles:
        ax._ref_legend = ax.legend(
            handles=handles, loc='lower right', frameon=True,
            prop={'size': 9}, title="Ref lines"
        )
    fig.canvas.draw_idle()
        
def _add_ref_line(ax, fig, mode, _data_lines_ignored=None):
    import tkinter as tk
    from tkinter import simpledialog, messagebox

    class XYLineDialog(simpledialog.Dialog):
        def body(self, master):
            tk.Label(master, text="x =").grid(row=0, column=0, sticky="e")
            tk.Label(master, text="y =").grid(row=1, column=0, sticky="e")
            if mode == "plot_CDF":
                tk.Label(master, text = '%').grid(row = 1, column = 2, sticky = 'w')
            elif mode == "plot_Z":
                tk.Label(master, text = "\u03C3").grid(row = 1, column = 2, sticky = 'w')
            self.x_entry = tk.Entry(master); self.x_entry.grid(row=0, column=1, padx=6)
            self.y_entry = tk.Entry(master); self.y_entry.grid(row=1, column=1, padx=6)
            return self.x_entry         # 初始焦点

        def validate(self):
            x_raw, y_raw = self.x_entry.get().strip(), self.y_entry.get().strip()
            if not x_raw and not y_raw:
                messagebox.showerror("Wrong input", "Please at least input x or y")
                return False
            try:
                self.x_val = float(x_raw) if x_raw else None
                if mode == 'plot_CDF':
                    self.y_val = float(y_raw) / 100 if y_raw else None
                elif mode == 'plot_Z':
                    self.y_val = float(y_raw) if y_raw else None
                return True
            except Exception:
                messagebox.showerror("Wrong input", "Please input a figure.")
                return False
        
    dlg = XYLineDialog(parent=None, title="Add Ref Line")
    if not hasattr(dlg, "x_val") and not hasattr(dlg, "y_val"):
        return        # 用户取消

    # 可能一次要加两条线
    for is_x, val in [(True, getattr(dlg, "x_val", None)),
                      (False, getattr(dlg, "y_val", None))]:
        if val is None:
            continue

        # 依据当前坐标轴已有的所有数据曲线计算交点
        data_lines = _get_data_lines(ax)

        # 画线 + 元数据
        if is_x:
            ref = ax.axvline(val, linestyle='--', linewidth=1.5, color='tab:red')
            ref._is_x  = True
            base_name  = f"x = {val:g}"
        else:
            ref = ax.axhline(val, linestyle='--', linewidth=1.5, color='tab:green')
            ref._is_x  = False
            base_name  = f"y = {val:g}"
        ref._is_ref = True
        ref._val    = val
        ref.set_picker(True)

        # 首次计算交点文本
        pairs = []
        for ln in data_lines:
            xd, yd = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
            lab = _short_label(ln.get_label())
            if ref._is_x:
                y = find_y_given_x0(xd, yd, val)
                if y is not None:
                    pairs.append(f"{lab}:({val:.4g},{y:.4g})")
            else:
                xs = find_x_given_y0(xd, yd, val)
                for xv in xs:
                    pairs.append(f"{lab}:({xv:.4g},{val:.4g})")
        ref.set_label(base_name + (" | " + "; ".join(pairs) if pairs else " | no intersections in range"))

        # 维护参考线列表
        if not hasattr(ax, "_ref_lines"):
            ax._ref_lines = []
        ax._ref_lines.append(ref)

    # 刷新图例（右下角），保留主图例
    _refresh_ref_legend(ax, fig)

# —— 图例（legend）格式 —— #
def _tex_escape(s: str) -> str:
    # 转义mathtext特殊字符
    return re.sub(r'([_$%#&^{}])', r'\\\1', s)

def _label(dc: DataColumn, min_, max_):
    safe_name = _tex_escape(dc.name)
    return (rf"$\mathbf{{{safe_name}}}$" + "\n" #通过LaTeX中的\mathbf进行加粗
            f"N={dc.N}, med={dc.med},\n"
            f"max={max_}, min={min_}")

# —— 绘制CDF —— #
def plot_CDF(data_cols: list[DataColumn],
                       new_figure = True, 
    grid_var = True): #针对矩阵：内含多个字典
#将读取到的数据进行绘图
    plt = _mpl() #懒加载
    global bounds
    overall_max_x = max(max(dc.stats.keys()) for dc in data_cols)
    overall_min_x = min(min(dc.stats.keys()) for dc in data_cols)
    if new_figure:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_yscale('piecewise', bounds=bounds)
    else: #如果选择在原图上画
        ax = plt.gca()#获取当前图像中的当前所有内容，并赋值给变量ax
        fig = ax.figure
        if ax.get_yscale() != 'piecewise': #确保用的是piecewise这个自定义的放大函数
            ax.set_yscale('piecewise', bounds=bounds)
        overall_max_x = max(overall_max_x, ax.get_xlim()[1])#更新最大最小值
        overall_min_x = min(overall_min_x, ax.get_xlim()[0])
    plt.rcParams['font.sans-serif'] = ['Calibrius','SimSun'] #英文用calibrius, 中文用宋体
    plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
    ax.margins(y=0.08)  
    ax.set_title("Title", fontweight = 'bold')
    ax.set_xlabel("Value", fontweight = 'bold')
    ax.set_ylabel("Cum Prob", fontweight = 'bold')
    ax.set_ylim(bounds[0], bounds[-1]) 

    lines = []
    
    for dc in data_cols:
        x = list(dc.stats.keys())
        cum, z = zip(*dc.stats.values())      
        label = _label(dc, min(x), max(x))

        line, = ax.plot(x, cum,
                         marker='o',
                         linewidth=1,
                         label=label)
        line.zs = z
        lines.append(line)
    
    scalar = (overall_max_x - overall_min_x) * 0.1 

    ax.set_xlim(overall_min_x - scalar, overall_max_x + scalar)
    ax.set_xticks(np.linspace(overall_min_x - scalar,
                                overall_max_x + scalar, 11
                                  ))

    ax.tick_params(axis='x', rotation=45)
    
    if new_figure:
        # first draw: show just these lines
        leg = ax.legend(
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        )
        ax._main_legend = leg
    else:
        old_leg = getattr(ax, "_main_legend", None)
        if old_leg is not None:
            for t in old_leg.get_texts():
                t.set_picker(False)
            old_leg.remove()

        # 只保留“数据曲线”，过滤掉参考线（_is_ref=True）
        data_lines_all  = [ln for ln in ax.get_lines() if not getattr(ln, "_is_ref", False)]
        data_labels_all = [ln.get_label() for ln in data_lines_all]
        leg = ax.legend(
            data_lines_all, data_labels_all,
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        )
        ax._main_legend = leg

        # 如已有参考线图例，则重新加回画布（防止被新 legend 覆盖）
        if hasattr(ax, "_ref_legend") and ax._ref_legend:
            ax.add_artist(ax._ref_legend)

        _recompute_ref_labels(ax)   
        _refresh_ref_legend(ax, fig)  

        for txt in leg.get_texts():
            txt.set_picker(True)
    ax._main_legend = leg

# —— 定义鼠标徘徊（hover）事件，触发annotation（悬浮小窗） —— #
    annot = ax.annotate(
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
        if event.inaxes == ax:
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

# —— 在画布（figure）上设置参考线的交互按钮 —— #
    
    if not hasattr(ax, "_add_btn"):
        ax._add_btn = ax.text(
            0.99, 1.02, "[Add Line]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="0.9", ec="0.7", alpha=0.8)
        )
        ax._add_btn.set_picker(True)
        ax._add_btn._no_edit = True

    if not hasattr(ax, "_clear_btn"):
        ax._clear_btn = ax.text(
            0.80, 1.02, "[Clear Lines]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="0.9", ec="0.7", alpha=0.8)
        )
        ax._clear_btn.set_picker(True)
        ax._clear_btn._no_edit = True

    mode = 'plot_CDF'
    def on_pick(event):
        artist = event.artist
        # 用 ax 属性来判断，而不是闭包里的局部变量
        if artist is getattr(ax, "_add_btn", None):
            # 防抖/防重入：一个对话框未关，不再弹新框
            if getattr(fig, "_ref_dialog_open", False):
                return
            fig._ref_dialog_open = True
            try:
                _add_ref_line(ax, fig, mode)
            finally:
                fig._ref_dialog_open = False
            return

        if artist is getattr(ax, "_clear_btn", None):
            if hasattr(ax, "_ref_lines"):
                for ln in list(ax._ref_lines):
                    try:
                        ln.remove()
                    except Exception:
                        pass
                ax._ref_lines.clear()
            if hasattr(ax, "_ref_legend") and ax._ref_legend:
                try:
                    ax._ref_legend.remove()
                except Exception:
                    pass
                ax._ref_legend = None
            fig.canvas.draw_idle()
            return

    # 只绑定一次这个 pick 回调（按 figure）
    if getattr(fig, "_ref_btn_pick_cid", None) is None:
        fig._ref_btn_pick_cid = fig.canvas.mpl_connect("pick_event", on_pick)

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
        fig, ax = plt.subplots(figsize = (8,5))
    else:
        ax = plt.gca()
        fig = ax.figure
        overall_max_x = max(overall_max_x, ax.get_xlim()[1])
        overall_min_x = min(overall_min_x, ax.get_xlim()[0])
        overall_max_y = max(overall_max_y, ax.get_ylim()[1])
        overall_min_y = min(overall_min_y, ax.get_ylim()[0])
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.rcParams['font.sans-serif'] = ['Calibrius','SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    ax.margins(y=0.08) 
    ax.set_title("Title", fontweight = "bold")
    ax.set_xlabel("Value", fontweight = 'bold')
    ax.set_ylabel("Z value(\u03C3)", fontweight = 'bold')
    scalar = (overall_max_x - overall_min_x) * 0.1 
    ax.set_xlim(overall_min_x - scalar, overall_max_x + scalar)
    ax.set_xticks(np.linspace(overall_min_x - scalar,
                                overall_max_x + scalar, 11
                                  )) #默认分成十段，有10%余量
    rounded_max = int(overall_max_y + 1)
    yticks = list(range(-rounded_max, rounded_max + 1)) #y轴sigma四舍五入为整数
    yticks_labels = [str(ytick) + "\u03C3" for ytick in yticks]
    ax.set_yticks(ticks = yticks, labels = yticks_labels)
    ax.tick_params(axis='x', rotation=45)#x轴刻度转45度

    lines = []
    for dc in data_cols:
        x = list(dc.stats.keys())
        cum, z = zip(*dc.stats.values())
        label = _label(dc, min(x), max(x))
        line, = ax.plot(x, z,
                        marker='s',
                        linewidth=1,
                        label=label)
        line.zs = cum
        lines.append(line)
        
    
    if new_figure:
        # first draw: show just these lines
        leg = ax.legend(
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        )
        ax._main_legend = leg
    else:
        old_leg = getattr(ax, "_main_legend", None)
        if old_leg is not None:
            for t in old_leg.get_texts():
                t.set_picker(False)
            old_leg.remove()

        data_lines_all  = [ln for ln in ax.get_lines() if not getattr(ln, "_is_ref", False)]
        data_labels_all = [ln.get_label() for ln in data_lines_all]
        leg = ax.legend(
            data_lines_all, data_labels_all,
            loc='upper left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, prop={'size': 9}
        )
        ax._main_legend = leg

        # 如已有参考线图例，则重新加回画布（防止被新 legend 覆盖）
        if hasattr(ax, "_ref_legend") and ax._ref_legend:
            ax.add_artist(ax._ref_legend)

        _recompute_ref_labels(ax)     # 或 ax
        _refresh_ref_legend(ax, fig)  # 或 ax, fig

    for txt in leg.get_texts():
        txt.set_picker(True)

# —— 定义鼠标徘徊（hover）事件，触发annotation（悬浮小窗） —— #
    annot = ax.annotate(
        "", xy=(0,0), xytext=(15, -15), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        clip_on=False
    )
    annot.set_visible(False)#默认不可见，检测到鼠标徘徊才可见

    def update_annot(line, ind):#编辑悬浮小窗
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
        if event.inaxes == ax:
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

    fig.canvas.mpl_connect("motion_notify_event", hover)#鼠标徘徊事件

# —— 在画布上设置参考线交互按钮 —— #
    if not hasattr(ax, "_add_btn"):
        ax._add_btn = ax.text(
            0.99, 1.02, "[Add Line]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="0.9", ec="0.7", alpha=0.8)
        )
        ax._add_btn.set_picker(True)
        ax._add_btn._no_edit = True

    if not hasattr(ax, "_clear_btn"):
        ax._clear_btn = ax.text(
            0.80, 1.02, "[Clear Lines]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="0.9", ec="0.7", alpha=0.8)
        )
        ax._clear_btn.set_picker(True)
        ax._clear_btn._no_edit = True

    mode = 'plot_Z'
    def on_pick(event):
        artist = event.artist
        # 用 ax 属性来判断，而不是闭包里的局部变量
        if artist is getattr(ax, "_add_btn", None):
            # 防抖/防重入：一个对话框未关，不再弹新框
            if getattr(fig, "_ref_dialog_open", False):
                return
            fig._ref_dialog_open = True
            try:
                _add_ref_line(ax, fig, mode)
            finally:
                fig._ref_dialog_open = False
            return

        if artist is getattr(ax, "_clear_btn", None):
            if hasattr(ax, "_ref_lines"):
                for ln in list(ax._ref_lines):
                    try:
                        ln.remove()
                    except Exception:
                        pass
                ax._ref_lines.clear()
            if hasattr(ax, "_ref_legend") and ax._ref_legend:
                try:
                    ax._ref_legend.remove()
                except Exception:
                    pass
                ax._ref_legend = None
            fig.canvas.draw_idle()
            return

    # 只绑定一次这个 pick 回调（按 figure）
    if getattr(fig, "_ref_btn_pick_cid", None) is None:
        fig._ref_btn_pick_cid = fig.canvas.mpl_connect("pick_event", on_pick)

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


#ax.xaxis.set_major_locator(MaxNLocator(nbins=11, prune='both'))  # 最多11个主刻度

