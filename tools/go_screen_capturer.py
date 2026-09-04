"""
围棋棋盘专用连续截图工具 (Go Board Screen Capturer)
------------------------------------------------------
功能特点：
1. 区域锁定：交互式拉框选区，框选一次野狐/弈客棋盘后永久固定范围，绝不跑偏。
2. 全局热键：后台静默监听（默认 F8），打谱时无需切换窗口，按一下即截一张。
3. 智能命名：支持对局前缀 + 手数连续递增（如 fox_game01_move_015.png）。
4. 容错支持：自带“撤销上一张”按钮（按错时一键删除上张图并回退手数）。
5. 音频反馈：截图成功伴随轻柔提示音，打谱截图无缝衔接。
6. 自动记忆：退出自动保存选区坐标与路径，下次打开免重新框选。
"""

import os
import sys
import json
import time
import ctypes
from ctypes import wintypes
import threading
import winsound
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# 开启 Windows 高 DPI 适配，确保坐标与物理像素 1:1 绝对对齐
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "capturer_config.json")

# Windows GDI 截图实现（极速、原生无依赖、避免 PIL ImageGrab 的底层兼容问题）
def capture_screen_region(x1, y1, x2, y2) -> Image.Image:
    """截取屏幕上指定绝对坐标的矩形区域，返回 PIL RGB 图像。"""
    w = max(1, int(x2 - x1))
    h = max(1, int(y2 - y1))
    
    hwnd = 0  # 整个桌面窗口
    hwndDC = ctypes.windll.user32.GetWindowDC(hwnd)
    mfcDC = ctypes.windll.gdi32.CreateCompatibleDC(hwndDC)
    saveBitMap = ctypes.windll.gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    ctypes.windll.gdi32.SelectObject(mfcDC, saveBitMap)
    
    # SRCCOPY = 0x00CC0020
    ctypes.windll.gdi32.BitBlt(mfcDC, 0, 0, w, h, hwndDC, int(x1), int(y1), 0x00CC0020)
    
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ('biSize', wintypes.DWORD),
            ('biWidth', wintypes.LONG),
            ('biHeight', wintypes.LONG),
            ('biPlanes', wintypes.WORD),
            ('biBitCount', wintypes.WORD),
            ('biCompression', wintypes.DWORD),
            ('biSizeImage', wintypes.DWORD),
            ('biXPelsPerMeter', wintypes.LONG),
            ('biYPelsPerMeter', wintypes.LONG),
            ('biClrUsed', wintypes.DWORD),
            ('biClrImportant', wintypes.DWORD),
        ]
    
    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down DIB
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    
    buf = ctypes.create_string_buffer(w * h * 4)
    ctypes.windll.gdi32.GetDIBits(mfcDC, saveBitMap, 0, h, buf, ctypes.byref(bmi), 0)
    
    ctypes.windll.gdi32.DeleteObject(saveBitMap)
    ctypes.windll.gdi32.DeleteDC(mfcDC)
    ctypes.windll.user32.ReleaseDC(hwnd, hwndDC)
    
    img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1)
    return img.convert('RGB')


class RegionSelectorOverlay:
    """全屏半透明拉框选区覆盖层"""
    def __init__(self, parent, callback):
        self.parent = parent
        self.callback = callback
        
        self.top = tk.Toplevel(parent)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.35)
        self.top.attributes("-topmost", True)
        self.top.config(cursor="cross")
        
        self.canvas = tk.Canvas(self.top, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 提示文字
        self.canvas.create_text(
            self.top.winfo_screenwidth() // 2, 80,
            text="按住鼠标左键拖动以框选棋盘区域，松开鼠标完成选区；按 ESC 键取消",
            fill="#FFFF00", font=("Microsoft YaHei", 18, "bold")
        )
        
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.info_id = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.top.bind("<Escape>", lambda e: self.close())

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.info_id:
            self.canvas.delete(self.info_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00FF00", width=2
        )

    def on_mouse_drag(self, event):
        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
        w = abs(cur_x - self.start_x)
        h = abs(cur_y - self.start_y)
        
        if self.info_id:
            self.canvas.delete(self.info_id)
        self.info_id = self.canvas.create_text(
            (self.start_x + cur_x) // 2,
            min(self.start_y, cur_y) - 20,
            text=f"大小: {w} × {h} 像素",
            fill="#00FF00", font=("Consolas", 14, "bold")
        )

    def on_button_release(self, event):
        end_x, end_y = event.x, event.y
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        self.close()
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            self.callback(x1, y1, x2, y2)
        else:
            messagebox.showwarning("提示", "选区面积过小，已忽略。")

    def close(self):
        self.top.destroy()


class GoScreenCapturerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("围棋棋盘连续截图工具 v1.0")
        self.root.geometry("540x660")
        self.root.resizable(False, False)
        
        # 核心变量
        self.save_dir = tk.StringVar(value=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "screenshots")))
        self.game_prefix = tk.StringVar(value="fox_game01")
        self.current_move = tk.IntVar(value=1)
        self.step_size = tk.IntVar(value=1)
        self.zero_pad = tk.IntVar(value=3)
        self.hotkey_name = tk.StringVar(value="F8")
        self.sound_beep = tk.BooleanVar(value=True)
        
        # 选区范围: (x1, y1, x2, y2)
        self.region_bbox = None
        self.total_captured = 0
        self.last_saved_filepath = None
        
        # 加载历史记忆配置
        self.load_config()
        
        # 构建界面
        self.build_ui()
        
        # 启动热键监听后台线程
        self.running = True
        self.hotkey_thread = threading.Thread(target=self.hotkey_listener_loop, daemon=True)
        self.hotkey_thread.start()
        
        # 退出钩子
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "save_dir" in cfg and os.path.exists(cfg["save_dir"]):
                        self.save_dir.set(cfg["save_dir"])
                    if "game_prefix" in cfg:
                        self.game_prefix.set(cfg["game_prefix"])
                    if "current_move" in cfg:
                        self.current_move.set(cfg["current_move"])
                    if "step_size" in cfg:
                        self.step_size.set(cfg["step_size"])
                    if "hotkey" in cfg:
                        self.hotkey_name.set(cfg["hotkey"])
                    if "bbox" in cfg and cfg["bbox"] and len(cfg["bbox"]) == 4:
                        self.region_bbox = tuple(cfg["bbox"])
            except Exception as e:
                print("加载配置失败:", e)

    def save_config(self):
        cfg = {
            "save_dir": self.save_dir.get(),
            "game_prefix": self.game_prefix.get(),
            "current_move": self.current_move.get(),
            "step_size": self.step_size.get(),
            "hotkey": self.hotkey_name.get(),
            "bbox": self.region_bbox
        }
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("保存配置失败:", e)

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 区域设置
        region_group = ttk.LabelFrame(main_frame, text=" 1. 截图范围锁定 (棋盘区域) ", padding="10")
        region_group.pack(fill=tk.X, pady=(0, 10))

        self.region_label = ttk.Label(
            region_group,
            text=self.get_region_display_text(),
            font=("Consolas", 10, "bold"),
            foreground="#0066cc"
        )
        self.region_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_box = ttk.Frame(region_group)
        btn_box.pack(side=tk.RIGHT)
        
        ttk.Button(btn_box, text="🎯 框选区域", command=self.start_region_select).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_box, text="👁️ 预览", command=self.preview_current_region).pack(side=tk.LEFT, padx=3)

        # 2. 对局命名与计数
        naming_group = ttk.LabelFrame(main_frame, text=" 2. 命名与手数控制 ", padding="10")
        naming_group.pack(fill=tk.X, pady=(0, 10))

        # 对局前缀
        row1 = ttk.Frame(naming_group)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="对局前缀:", width=10).pack(side=tk.LEFT)
        prefix_entry = ttk.Entry(row1, textvariable=self.game_prefix)
        prefix_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        prefix_entry.bind("<KeyRelease>", lambda e: self.update_name_preview())

        # 当前第几手
        row2 = ttk.Frame(naming_group)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="当前手数:", width=10).pack(side=tk.LEFT)
        move_spin = ttk.Spinbox(row2, from_=1, to=500, textvariable=self.current_move, width=8)
        move_spin.pack(side=tk.LEFT, padx=5)
        move_spin.bind("<KeyRelease>", lambda e: self.update_name_preview())
        move_spin.bind("<<Increment>>", lambda e: self.root.after(50, self.update_name_preview))
        move_spin.bind("<<Decrement>>", lambda e: self.root.after(50, self.update_name_preview))

        ttk.Label(row2, text="递增步长:").pack(side=tk.LEFT, padx=(15, 2))
        step_spin = ttk.Spinbox(row2, from_=1, to=10, textvariable=self.step_size, width=5)
        step_spin.pack(side=tk.LEFT, padx=5)

        ttk.Button(row2, text="重置为第1手", command=self.reset_move_to_1).pack(side=tk.RIGHT, padx=5)

        # 实时命名预览
        row3 = ttk.Frame(naming_group)
        row3.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row3, text="下张命名:").pack(side=tk.LEFT)
        self.preview_name_label = ttk.Label(row3, text="", font=("Consolas", 10, "bold"), foreground="#cc3300")
        self.preview_name_label.pack(side=tk.LEFT, padx=5)
        self.update_name_preview()

        # 3. 保存路径与快捷键
        save_group = ttk.LabelFrame(main_frame, text=" 3. 存储与快捷键 ", padding="10")
        save_group.pack(fill=tk.X, pady=(0, 10))

        # 保存目录
        path_row = ttk.Frame(save_group)
        path_row.pack(fill=tk.X, pady=3)
        ttk.Label(path_row, text="保存目录:", width=10).pack(side=tk.LEFT)
        ttk.Entry(path_row, textvariable=self.save_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(path_row, text="浏览...", command=self.choose_save_dir).pack(side=tk.RIGHT)

        # 快捷键与声音
        opt_row = ttk.Frame(save_group)
        opt_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(opt_row, text="全局热键:", width=10).pack(side=tk.LEFT)
        hotkey_cb = ttk.Combobox(opt_row, textvariable=self.hotkey_name, values=["F8", "F9", "F10", "F12"], width=7, state="readonly")
        hotkey_cb.pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(opt_row, text="截图成功伴随提示音 (Beep)", variable=self.sound_beep).pack(side=tk.LEFT, padx=15)
        ttk.Button(opt_row, text="📂 打开目录", command=self.open_save_folder).pack(side=tk.RIGHT)

        # 4. 操作与状态面板
        action_group = ttk.LabelFrame(main_frame, text=" 4. 操作与监控 ", padding="12")
        action_group.pack(fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(action_group)
        btn_row.pack(fill=tk.X, pady=5)

        self.btn_capture = tk.Button(
            btn_row, text=f"📸 立即截图 (或按 {self.hotkey_name.get()})",
            bg="#28a745", fg="white", font=("Microsoft YaHei", 12, "bold"),
            relief=tk.RAISED, height=2, command=self.execute_capture
        )
        self.btn_capture.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_undo = tk.Button(
            btn_row, text="↩️ 撤销上张",
            bg="#dc3545", fg="white", font=("Microsoft YaHei", 10, "bold"),
            width=10, relief=tk.RAISED, command=self.undo_last_capture
        )
        self.btn_undo.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态日志
        self.status_text = tk.StringVar(value="准备就绪。先点击【🎯 框选区域】圈定棋盘，然后在野狐中按快捷键开始！")
        self.status_label = ttk.Label(
            action_group, textvariable=self.status_text,
            wraplength=480, font=("Microsoft YaHei", 9), foreground="#333333"
        )
        self.status_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 底部计数统计
        self.stats_label = ttk.Label(
            action_group,
            text=f"本局已截图: 0 张",
            font=("Microsoft YaHei", 9, "bold"),
            foreground="#008080"
        )
        self.stats_label.pack(anchor="e", pady=(5, 0))

    def get_region_display_text(self):
        if not self.region_bbox:
            return "未选定范围（请先框选）"
        x1, y1, x2, y2 = self.region_bbox
        return f"已锁定: ({x1}, {y1}) → ({x2}, {y2}) [{x2-x1}×{y2-y1}]"

    def get_current_filename(self) -> str:
        prefix = self.game_prefix.get().strip() or "go_game"
        move = self.current_move.get()
        pad = self.zero_pad.get()
        return f"{prefix}_move_{str(move).zfill(pad)}.png"

    def update_name_preview(self):
        try:
            self.preview_name_label.config(text=self.get_current_filename())
        except Exception:
            pass

    def reset_move_to_1(self):
        self.current_move.set(1)
        self.total_captured = 0
        self.stats_label.config(text=f"本局已截图: {self.total_captured} 张")
        self.update_name_preview()
        self.status_text.set("已重置手数为第 1 手，计数器归零。")

    def choose_save_dir(self):
        d = filedialog.askdirectory(initialdir=self.save_dir.get(), title="选择截图保存目录")
        if d:
            self.save_dir.set(d)

    def open_save_folder(self):
        d = self.save_dir.get()
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def start_region_select(self):
        # 最小化或隐藏主窗口，方便用户看到完整的目标棋盘
        self.root.iconify()
        self.root.after(300, lambda: RegionSelectorOverlay(self.root, self.on_region_selected))

    def on_region_selected(self, x1, y1, x2, y2):
        self.region_bbox = (x1, y1, x2, y2)
        self.region_label.config(text=self.get_region_display_text())
        self.root.deiconify()
        self.status_text.set(f"✅ 选区锁定成功！坐标: ({x1}, {y1}) 到 ({x2}, {y2})，尺寸: {x2-x1}×{y2-y1}。现在可以在野狐中按快捷键截图了。")
        self.save_config()

    def preview_current_region(self):
        if not self.region_bbox:
            messagebox.showwarning("提示", "请先点击【🎯 框选区域】选定棋盘！")
            return
        
        x1, y1, x2, y2 = self.region_bbox
        img = capture_screen_region(x1, y1, x2, y2)
        
        # 弹出预览窗口
        top = tk.Toplevel(self.root)
        top.title("当前选区截图预览")
        
        # 缩小展示
        max_view = 500
        w, h = img.size
        scale = min(max_view / w, max_view / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        preview_img = img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)
        
        lbl = ttk.Label(top, image=photo)
        lbl.image = photo
        lbl.pack(padx=10, pady=10)
        
        info = ttk.Label(top, text=f"真实分辨率: {w} × {h} 像素 (已确认对准棋盘)", font=("Microsoft YaHei", 10))
        info.pack(pady=(0, 10))

    def execute_capture(self):
        """核心截图保存方法"""
        if not self.region_bbox:
            messagebox.showwarning("提示", "请先点击【🎯 框选区域】选定棋盘！")
            return
        
        save_folder = self.save_dir.get().strip()
        os.makedirs(save_folder, exist_ok=True)
        
        filename = self.get_current_filename()
        filepath = os.path.join(save_folder, filename)
        
        # 抓取图像
        x1, y1, x2, y2 = self.region_bbox
        try:
            img = capture_screen_region(x1, y1, x2, y2)
            img.save(filepath)
        except Exception as e:
            self.status_text.set(f"❌ 截图保存失败: {e}")
            return
        
        self.last_saved_filepath = filepath
        curr_move = self.current_move.get()
        step = self.step_size.get()
        
        self.total_captured += 1
        self.current_move.set(curr_move + step)
        self.update_name_preview()
        
        self.status_text.set(f"📸 已成功保存: {filename}\n下一手将保存为: {self.get_current_filename()}")
        self.stats_label.config(text=f"本局已截图: {self.total_captured} 张")
        
        if self.sound_beep.get():
            try:
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass

    def undo_last_capture(self):
        """撤销上一张截图（删除文件并恢复手数）"""
        if not self.last_saved_filepath or not os.path.exists(self.last_saved_filepath):
            messagebox.showinfo("提示", "没有可撤销的截图记录。")
            return
        
        try:
            del_file = os.path.basename(self.last_saved_filepath)
            os.remove(self.last_saved_filepath)
            
            # 回退手数
            step = self.step_size.get()
            self.current_move.set(max(1, self.current_move.get() - step))
            self.total_captured = max(0, self.total_captured - 1)
            
            self.update_name_preview()
            self.stats_label.config(text=f"本局已截图: {self.total_captured} 张")
            self.status_text.set(f"↩️ 已撤销并删除: {del_file}\n手数已回退到: 第 {self.current_move.get()} 手")
            self.last_saved_filepath = None
            
            if self.sound_beep.get():
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception as e:
            messagebox.showerror("错误", f"删除文件失败: {e}")

    def hotkey_listener_loop(self):
        """后台低开销轮询全局快捷键（检测按键边缘）"""
        key_map = {
            "F8": 0x77,
            "F9": 0x78,
            "F10": 0x79,
            "F12": 0x7B
        }
        
        last_pressed = False
        while self.running:
            target_hk = self.hotkey_name.get()
            vk_code = key_map.get(target_hk, 0x77)
            
            # 检查高位是否被按下 (0x8000)
            is_down = bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
            
            # 上升沿触发（只在刚按下的那一瞬间触发一次，避免长按重复连截）
            if is_down and not last_pressed:
                self.root.after(0, self.execute_capture)
            
            last_pressed = is_down
            time.sleep(0.04)  # 40ms 轮询周期，CPU 占用接近 0%

    def on_close(self):
        self.running = False
        self.save_config()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GoScreenCapturerApp(root)
    root.mainloop()
