import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import simpledialog
import plot  
import re
from matplotlib.text import Text

def fig_is_alive(fig):
    """fig 存在且窗口没被关掉"""
    return fig is not None and plt.fignum_exists(fig.number)

def _bind_close_flag(fig, flag_name):
    def _on_close(event):
        globals()[flag_name] = None
    fig.canvas.mpl_connect('close_event', _on_close)

def enable_text_editing(fig, ax, master):
    # make title & axis labels pickable
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
        if not isinstance(artist, Text):
            return

        old = artist.get_text()
        parts    = old.split('\n', 1)
        math_part = parts[0]
        rest      = '\n' + parts[1] if len(parts) > 1 else ''

        # 只拦截和匹配 "$\mathbf{…}$"
        m = re.fullmatch(r'\$\\mathbf\{(.+?)\}\$', math_part)
        if m:
            inner = m.group(1)
            prompt = f"当前系列名：{inner}\n请输入新的系列名："
            new_inner = simpledialog.askstring("修改系列名", prompt, parent=master)
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
            prompt = f"当前文本：{old}\n请输入新的文本："
            new = simpledialog.askstring("修改文本", prompt, parent=master)
            if new is not None and new != old:
                artist.set_text(new)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

cum_fig, z_fig = None, None
def plot_cum_distribution():
    global cum_fig
    try:
        round_input = abs(int(entry_round.get()))
    except ValueError:
        messagebox.showerror("不正确的输入", "请输入正整数。")
        return
    data_cols = plot.clipboard_to_array(root, round_input)
    data_dicts = []
    for col in data_cols:
        dic = plot.cum_prob(
        plot.filter(plot.radix_sort(col, round_input)))
        data_dicts.append(dic)
    if overlay_var.get() and fig_is_alive(cum_fig):
        plt.figure(cum_fig.number)
        ax = plt.gca()
        enable_text_editing(plt.gcf(), ax, root)
        new_fig = False
    else:
        new_fig = True
    plot.plot_Z(data_dicts, new_figure=new_fig)
    if new_fig:
        cum_fig = plt.gcf()
        ax = cum_fig.axes[0]
        enable_text_editing(cum_fig, ax, root)
        _bind_close_flag(cum_fig, 'cum_fig')

def plot_z_value():
    global z_fig
    try:
        round_input = abs(int(entry_round.get()))
    except ValueError:
        messagebox.showerror("不正确的输入", "请输入正整数。")
        return
    data_cols = plot.clipboard_to_array(root, round_input)
    data_dicts = [plot.cum_prob(
        plot.filter(plot.radix_sort(col, round_input)))
              for col in data_cols]
    if overlay_var.get() and fig_is_alive(z_fig):
        plt.figure(z_fig.number)
        ax = plt.gca()
        enable_text_editing(plt.gcf(), ax, root)
        new_fig = False
    else:
        new_fig = True
    plot.plot_Z(data_dicts, new_figure=new_fig)
    if new_fig:
        z_fig = plt.gcf()
        ax = z_fig.axes[0]
        enable_text_editing(z_fig, ax, root)
        _bind_close_flag(z_fig, 'z_fig')
# —— GUI 布局 —— #
root = tk.Tk()
root.title("Plotter")
root.geometry("200x250")
container = tk.Frame(root)
container.pack(fill="both", expand=True, padx=10, pady=10)
# 输入区
btn_cum = tk.Button(container, 
                    text="绘制累计概率曲线", 
                    command=plot_cum_distribution).pack(
                        fill = "x", padx=5)
btn_z = tk.Button(container, 
                  text = "绘制Z值（sigma）曲线", 
                  command = plot_z_value).pack(
                      fill = "x", padx = 5)
frame_round = tk.Frame(container)
frame_round.pack(fill="x", pady=5)
tk.Label(frame_round, text="保留几位小数").pack(side="left")
entry_round = tk.Entry(frame_round, width=5)
entry_round.pack(side="left")

overlay_var = tk.BooleanVar(value=False)
chk_overlay = tk.Checkbutton(container, text="在原图上绘制", 
               variable=overlay_var).pack(fill="x", pady=5)
root.mainloop()



    