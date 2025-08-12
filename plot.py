import sys, os, logging, traceback, threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.simpledialog import Dialog
import re
from matplotlib.text import Text
import numpy as np

# —— 捕获异常，一旦代码报错则跳出错误弹窗 —— #
LOG_FILE = os.path.join(os.path.expanduser("~"), "plot_gui_error.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def _format_exc(exc_type, exc_value, exc_tb) -> str:
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

def _show_error_dialog(title: str, brief: str):
    # 在 GUI 环境下弹窗；若 GUI 未就绪或已销毁，尽量吞掉而不再抛错
    try:
        messagebox.showerror(title, brief)
    except Exception:
        pass

def _handle_unhandled(exc_type, exc_value, exc_tb):
    #记录完整堆栈到日志
    text = _format_exc(exc_type, exc_value, exc_tb)
    logging.error("UNHANDLED EXCEPTION\n%s", text)

    #弹窗简要提示（标题 + 简短错误 + 日志位置）
    brief = f"{exc_type.__name__}: {exc_value}\n\n详细堆栈已保存到：\n{LOG_FILE}"
    _show_error_dialog("程序发生未处理的异常", brief)

#普通未捕获异常（主线程）
sys.excepthook = _handle_unhandled

#线程中的未捕获异常（Python 3.8+）
def _thread_excepthook(args: threading.ExceptHookArgs):
    _handle_unhandled(args.exc_type, args.exc_value, args.exc_traceback)
threading.excepthook = _thread_excepthook

#Tkinter 回调中的异常（按钮/事件等）
def _tk_report_callback_exception(self, exc, val, tb):
    _handle_unhandled(exc, val, tb)
# 用类级别覆盖，确保在 root 创建前生效
tk.Tk.report_callback_exception = _tk_report_callback_exception

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
    #每个figure都有编号，检测figure是否还存在
    return fig is not None and plt.fignum_exists(fig.number)

def _register_close(fig, lst):
    #当用户关闭窗口时，从lst中移除图片
    def _on_close(event):
        try:
            lst.remove(fig)
        except ValueError:
            pass
    fig.canvas.mpl_connect('close_event', _on_close)

def _latest_alive(lst):
    #从lst从后往前找第一个还存在的画布fig
    while lst and not plt.fignum_exists(lst[-1].number):
        lst.pop()
    return lst[-1] if lst else None

# —— 自定义x轴，y轴标题以及主标题，图例曲线名称 —— #
def enable_text_editing(fig, ax, master):
    def _refresh_legend_pickables(ax, fig):
        legend = getattr(ax, "_main_legend", None)
        fig._legend_texts = []
        if legend:
            for txt in legend.get_texts():
                if hasattr(txt, "_no_edit") and txt._no_edit:
                    continue
                txt.set_picker(True)
                fig._legend_texts.append(txt)

    if not getattr(fig, "_text_edit_initialized", False):
        ax.title.set_picker(True)
        ax.xaxis.label.set_picker(True)
        ax.yaxis.label.set_picker(True)
        fig._editing_lock = False
        fig._last_edit_t  = 0.0
        fig._text_edit_initialized = True

    _refresh_legend_pickables(ax, fig)

    def on_pick(event):
        import time
        artist = event.artist

        # 只处理允许编辑的文字
        allowed = [ax.title, ax.xaxis.label, ax.yaxis.label] + getattr(fig, "_legend_texts", [])
        if not (isinstance(artist, Text) and artist in allowed):
            return
        if getattr(artist, "_no_edit", False):
            return

        # —— 关键保险丝：同一 Text 若已有一个编辑框，立即返回 —— #
        if getattr(artist, "_edit_open", False):
            return
        artist._edit_open = True  # 立刻置位，防止并行回调再进来

        # 轻量防抖（防止某些后端重复触发）
        if time.monotonic() - getattr(fig, "_last_edit_t", 0) < 0.25:
            artist._edit_open = False
            return

        if getattr(fig, "_editing_lock", False):
            artist._edit_open = False
            return

        fig._editing_lock = True
        fig._last_edit_t  = time.monotonic()
        try:
            old = artist.get_text()
            parts = old.split('\n', 1)
            math_part = parts[0]
            rest = '\n' + parts[1] if len(parts) > 1 else ''

            m = re.fullmatch(r'\$\\mathbf\{(.*?)\}\$', math_part)
            if m:
                inner = m.group(1)
                prompt = f"Current title: {inner}\nPlease input new title: "
                new_inner = simpledialog.askstring("Edit", prompt, parent=master)
                if new_inner and new_inner != inner:
                    new_text = rf"$\mathbf{{{plot._tex_escape(new_inner)}}}${rest}"
                    artist.set_text(new_text)
                    handles, _ = ax.get_legend_handles_labels()
                    try:
                        idx = getattr(fig, "_legend_texts", []).index(artist)
                        if idx < len(handles):
                            handles[idx].set_label(new_text)
                    except ValueError:
                        pass
            else:
                prompt = f"Current title: {old}\nPlease input new title: "
                new = simpledialog.askstring("Edit", prompt, parent=master)
                if new is not None and new != old:
                    artist.set_text(new)

            fig.canvas.draw_idle()
        finally:
            fig._editing_lock = False
            artist._edit_open = False

    # 断开旧回调，确保只有一个
    old_cid = getattr(fig, "_text_edit_cid", None)
    if old_cid is not None:
        try:
            fig.canvas.mpl_disconnect(old_cid)
        except Exception:
            pass
    fig._text_edit_cid = fig.canvas.mpl_connect('pick_event', on_pick)




# —— 自定义x轴刻度 —— #
#用户输入最大值，最小值，分段数。根据这三个输入来自定义x轴
class AxisDialog(Dialog):
    def body(self, master): #输入界面
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
        for label in ax.get_xticklabels():
            label.set_picker(True)
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
    fig._xaxis_edit_cid = fig.canvas.mpl_connect("pick_event", on_pick)
    fig._xaxis_edit_enabled = True

# —— 绘制CDF —— #
def plot_cum_distribution():
    ensure_plot_libs()
    global cum_figs
    if entry_round.get() != "":
        try:
            round_input = abs(int(entry_round.get()))
        except ValueError:
            messagebox.showerror("Wrong input", "Please input integers.")
            return
    else:
        round_input = None
    try:
        data_cols = plot.clipboard_to_array(root, round_input)
    except ValueError:
        messagebox.showerror("Wrong input",
"Unable to calculate z-score。Please ensure:\n\nEach column has at least two distinct data")
        

    base_fig = _latest_alive(cum_figs) if overlay_var.get() else None
    if base_fig is not None:            #叠加画
        plt.figure(base_fig.number)
        ax = plt.gca()
        new_fig = False
    else:                               #新建画
        new_fig = True

    plot.plot_CDF(data_cols,
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
    if entry_round.get() != "":
        try:
            round_input = abs(int(entry_round.get()))
        except ValueError:
            messagebox.showerror("Wrong input", "Please input integers.")
            return
    else:
        round_input = None
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
    plot.plot_Z(data_cols, new_figure=new_fig, grid_var = grid_var.get())

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
root.title("Interface") #标题
root.geometry("220x250") #大小
container = tk.Frame(root) #交互界面主体
container.pack(fill="both", expand=True, padx=10, pady=10) 
# 输入区
btn_cum = tk.Button(container, 
                    text="Plot CDF curve", 
                    command=plot_cum_distribution).pack(
                        fill = "x", padx=5) #CDF曲线按钮
btn_z = tk.Button(container, 
                  text = "Plot z-score (sigma) curve", 
                  command = plot_z_value).pack(
                      fill = "x", padx = 5) #Z值曲线按钮
frame_round = tk.Frame(container)
frame_round.pack(fill="x", pady=5) #填充整个x轴，高度为5
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

