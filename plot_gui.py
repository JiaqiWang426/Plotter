import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.simpledialog import Dialog
import re
from matplotlib.text import Text
import numpy as np

plt, plot = None, None
cum_figs, z_figs = [], []

# —— 懒加载matplotlib和plot.py —— #
def ensure_plot_libs():
    global plt, plot
    if plt is None:
        import matplotlib.pyplot as _plt
        import plot as _plot
        plt, plot = _plt, _plot

def fig_is_alive(fig):
    """fig 存在且窗口没被关掉"""
    return fig is not None and plt.fignum_exists(fig.number)

def _register_close(fig, lst):
    """窗口真正关闭时，把 fig 从 lst 中移除"""
    def _on_close(evt):
        try:
            lst.remove(fig)
        except ValueError:
            pass
    fig.canvas.mpl_connect('close_event', _on_close)

def _latest_alive(lst):
    """从后往前找第一个活着的 Figure，找不到返回 None"""
    while lst and not plt.fignum_exists(lst[-1].number):
        lst.pop()
    return lst[-1] if lst else None

# —— 自定义x轴，y轴标题以及主标题，图例曲线名称 —— #
def enable_text_editing(fig, ax, master):
    # make title & axis labels pickable
    if getattr(fig, "_text_edit_enabled", False):
        return
    ax.title.set_picker(True)
    ax.xaxis.label.set_picker(True)
    ax.yaxis.label.set_picker(True)
    # make legend texts pickable
    legend = ax.get_legend()
    if legend:
        for txt in legend.get_texts():
            txt.set_picker(True)

    def on_pick(event):
        artist = event.artist
        if artist in ax.get_xticklabels() or artist in ax.get_yticklabels():
            return

        old = artist.get_text()
        parts    = old.split('\n', 1)
        math_part = parts[0]
        rest      = '\n' + parts[1] if len(parts) > 1 else ''

        # 只拦截和匹配 "$\mathbf{…}$"
        m = re.fullmatch(r'\$\\mathbf\{(.*?)\}\$', math_part)
        if m:
            inner = m.group(1)
            prompt = f"Current title: {inner}\nPlease input new title: "
            new_inner = simpledialog.askstring("Edit", prompt, parent=master)
            if new_inner and new_inner != inner:
                # 如果有中文（或其他非 ASCII），就 plain text + bold
                if any(ord(ch) > 127 for ch in new_inner):
                    artist.set_text(new_inner + rest)
                    artist.set_fontweight('bold')
                else:
                    # 继续用 TeX math 加粗
                    new_math = rf"$\mathbf{{{new_inner}}}$"
                    artist.set_text(new_math + rest)

                fig.canvas.draw_idle()
        else:
            # Title / axis label / 其它都走全量替换
            prompt = f"Current title: {old}\nPlease input new title: "
            new = simpledialog.askstring("Edit", prompt, parent=master)
            if new is not None and new != old:
                artist.set_text(new)
                fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('pick_event', on_pick)
    fig._text_edit_enabled = True
    fig._text_edit_cid    = cid 

# —— 自定义x轴刻度 —— #
class AxisDialog(Dialog):
    def body(self, master):
        tk.Label(master, text="Minimum: ").grid(row=0, column=0, sticky="e")
        tk.Label(master, text="Maximum: ").grid(row=1, column=0, sticky="e")
        tk.Label(master, text="Number of segments:").grid(row=2, column=0, sticky="e")

        self.min_entry = tk.Entry(master)
        self.max_entry = tk.Entry(master)
        self.num_entry = tk.Entry(master)

        self.min_entry.grid(row=0, column=1, padx=4)
        self.max_entry.grid(row=1, column=1, padx=4)
        self.num_entry.grid(row=2, column=1, padx=4)
        return self.min_entry  # 初始焦点

    def validate(self):
        try:
            self.min_val = float(self.min_entry.get())
            self.max_val = float(self.max_entry.get())
            self.num_val = int(self.num_entry.get())
            assert self.min_val < self.max_val and self.num_val > 0
            return True
        except Exception:
            messagebox.showerror("Wrong input", "Please ensure: \n1. min < max\n2. number of segments > 0")
            return False
    
    def apply(self):
        self.result = True
        
def enable_xaxis_editing(fig, ax, master):
    if getattr(fig, "_xaxis_edit_enabled", False):
        return
    for label in ax.get_xticklabels():
        label.set_picker(True)

    def on_pick(event):
        artist = event.artist
        if not (isinstance(artist, Text) and artist in ax.get_xticklabels()):
            return
        dialog = AxisDialog(master, title = "Custom x-axis")
        if not hasattr(dialog, "min_val"):
            return
        min_val = dialog.min_val
        max_val = dialog.max_val
        num = dialog.num_val
        new_ticks = np.linspace(min_val, max_val, num + 1)
        ax.set_xticks(new_ticks,
                    labels = [f"{t:.4f}" for t in new_ticks],
                    rotation = 45)
        ax.set_xlim(min_val, max_val)
        fig.canvas.draw_idle()
        for label in ax.get_xticklabels():
            label.set_picker(True)
        cid = fig.canvas.draw_idle()
        fig._xaxis_edit_enabled = True      # 标记
        fig._xaxis_edit_cid     = cid 
    fig.canvas.mpl_connect("pick_event", on_pick)

# —— 绘制CDF —— #
def plot_cum_distribution():
    ensure_plot_libs()
    global cum_figs
    try:
        round_input = abs(int(entry_round.get()))
    except ValueError:
        messagebox.showerror("不正确的输入", "请输入四舍五入到几位小数。")
        return
    try:
        data_cols = plot.clipboard_to_array(root, round_input)
    except ValueError:
        messagebox.showerror("Wrong input",
"Unable to calculate z-score。Please ensure:\n\nEach column has at least two distinct data")
        

    base_fig = _latest_alive(cum_figs) if overlay_var.get() else None
    if base_fig is not None:            # 叠加画
        plt.figure(base_fig.number)
        ax = plt.gca()
        new_fig = False
    else:                               # 新建画
        new_fig = True

    plot.plot_distribution(data_cols,
                            new_figure=new_fig,
                            grid_var = grid_var.get())

    if new_fig:
        fig = plt.gcf()
        cum_figs.append(fig)            # 记录到列表
        _register_close(fig, cum_figs)  # 断开时自动移除
        ax = fig.axes[0]
    else:
        fig = base_fig                  # 复用的那张

    enable_text_editing(fig, ax, root)
    enable_xaxis_editing(fig, ax, root)

def plot_z_value():
    ensure_plot_libs()
    global z_figs
    try:
        round_input = abs(int(entry_round.get()))
    except ValueError:
        messagebox.showerror("不正确的输入", "请输入四舍五入到几位小数。")
        return
    try:
        data_cols = plot.clipboard_to_array(root, round_input)
    except ValueError:
        messagebox.showerror("Wrong input",
"Unable to calculate z-score。Please ensure:\n\nEach column has at least two distinct data")
    base_fig = _latest_alive(z_figs) if overlay_var.get() else None
    if base_fig is not None:            # 叠加画
        plt.figure(base_fig.number)
        ax = plt.gca()
        new_fig = False
    else:                               # 新建画
        new_fig = True

    plot.plot_Z(data_cols, new_figure=new_fig)

    if new_fig:
        fig = plt.gcf()
        z_figs.append(fig)            # 记录到列表
        _register_close(fig, z_figs)  # 断开时自动移除
        ax = fig.axes[0]
    else:
        fig = base_fig                  # 复用的那张

    enable_text_editing(fig, ax, root)
    enable_xaxis_editing(fig, ax, root)
    
# —— GUI 布局 —— #
root = tk.Tk()
root.title("Interface")
root.geometry("220x250")
container = tk.Frame(root)
container.pack(fill="both", expand=True, padx=10, pady=10)
# 输入区
btn_cum = tk.Button(container, 
                    text="Plot CDF curve", 
                    command=plot_cum_distribution).pack(
                        fill = "x", padx=5)
btn_z = tk.Button(container, 
                  text = "Plot z-score (sigma) curve", 
                  command = plot_z_value).pack(
                      fill = "x", padx = 5)
frame_round = tk.Frame(container)
frame_round.pack(fill="x", pady=5)
tk.Label(frame_round, text="Number of decimal places").pack(side="left")
entry_round = tk.Entry(frame_round, width=5)
entry_round.pack(side="left")

overlay_var = tk.BooleanVar(value=False)
chk_overlay = tk.Checkbutton(container, text="Plot on the previous figure", 
               variable=overlay_var).pack(fill="x", pady=5)
grid_var = tk.BooleanVar(value = True)
chk_grid = tk.Checkbutton(container, text = "Add grids",
                          variable = grid_var).pack(fill = "x",
                                                pady = 5)
root.mainloop()



    