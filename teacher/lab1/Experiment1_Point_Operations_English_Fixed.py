"""Experiment 1: Point Operations (Linear Stretch / Gamma / Histogram Equalization)."""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2  # used only for image I/O and resizing
import numpy as np
import matplotlib


def _running_in_notebook():
    """Return True when the code is executed inside a Jupyter/IPython notebook."""
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell_name == "ZMQInteractiveShell"
    except Exception:
        return False


# Use TkAgg only when a desktop Tk environment is likely available. This avoids
# backend errors when the notebook is opened in a headless or online environment.
if not _running_in_notebook():
    has_display = bool(os.environ.get("DISPLAY")) or sys.platform.startswith(("win", "darwin"))
    if has_display:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ============================================================
#                    Core algorithm implementation
# ============================================================


def compute_histogram(img):
    """
    Compute the 256-bin histogram of an 8-bit grayscale image.

    Idea: create a counter array of length 256, flatten the image, and increase
    the counter corresponding to each pixel value.
    """
    hist = np.zeros(256, dtype=np.int64)
    flat = img.ravel()
    for v in flat:
        hist[int(v)] += 1
    return hist



def compute_percentile(img, p):
    """
    Compute the p-th percentile gray level without calling np.percentile.

    The cumulative histogram CDF[k] stores the number of pixels with gray level
    less than or equal to k. The first k whose cumulative count reaches p% of
    the image size is the percentile gray level.
    """
    p = float(p)
    if p <= 0:
        return int(img.min())
    if p >= 100:
        return int(img.max())

    hist = compute_histogram(img)
    total = img.size
    target = total * p / 100.0
    cum = 0
    for k in range(256):
        cum += hist[k]
        if cum >= target:
            return k
    return 255



def linear_stretch(img, lo_pct, hi_pct, out_lo=0, out_hi=255):
    """
    Piecewise linear contrast stretch.

    Formula:
        T(r) = (r - a) / (b - a) * (out_hi - out_lo) + out_lo

    where a and b are the low and high percentile gray levels. Pixels outside
    [a, b] are clipped to the selected output range.
    """
    out_lo = int(out_lo)
    out_hi = int(out_hi)
    if out_hi < out_lo:
        out_lo, out_hi = out_hi, out_lo

    a = compute_percentile(img, lo_pct)
    b = compute_percentile(img, hi_pct)
    if b - a < 1:
        b = a + 1

    img_f = img.astype(np.float32)
    out = (img_f - a) / (b - a) * (out_hi - out_lo) + out_lo
    out = np.clip(out, out_lo, out_hi)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8), a, b



def gamma_correction(img, gamma, c=1.0):
    """
    Gamma correction.

    Formula:
        T(r) = c * 255 * (r / 255)^gamma

    gamma < 1 brightens dark regions; gamma > 1 darkens the image; c controls
    the global gain.
    """
    gamma = max(float(gamma), 1e-6)
    img_f = img.astype(np.float32) / 255.0
    out = c * np.power(img_f, gamma) * 255.0
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)



def histogram_equalization(img):
    """
    Manual histogram equalization.

    Steps:
      1. Compute the histogram h(k).
      2. Compute the cumulative distribution CDF(k).
      3. Build a lookup table using the non-empty CDF minimum correction.
      4. Map every pixel through the lookup table.

    The CDF-min correction keeps constant images stable and avoids mapping a
    single-gray-level image to pure white.
    """
    hist = compute_histogram(img)
    cdf_counts = np.zeros(256, dtype=np.float64)
    cdf_counts[0] = hist[0]
    for k in range(1, 256):
        cdf_counts[k] = cdf_counts[k - 1] + hist[k]

    nonzero_cdf = cdf_counts[cdf_counts > 0]
    if nonzero_cdf.size == 0:
        transfer_curve = np.arange(256, dtype=np.float64)
        return img.copy(), transfer_curve

    cdf_min = nonzero_cdf[0]
    denom = img.size - cdf_min
    if denom <= 0:
        transfer_curve = np.arange(256, dtype=np.float64)
        return img.copy(), transfer_curve

    transfer_curve = (cdf_counts - cdf_min) / denom * 255.0
    transfer_curve = np.clip(transfer_curve, 0, 255)
    lut = np.round(transfer_curve).astype(np.uint8)
    out = lut[img]
    return out, transfer_curve



def AHE_histogram(img, window=8):
    for i in range(0, img.shape[0], window):
        compute_histogram(img[i:i+window])
        

# ============================================================
#                  Theme and reusable GUI widgets
# ============================================================

BG_DARK = "#1e1e2e"
BG_PANEL = "#252537"
BG_LIGHT = "#f7f7fa"
FG_TEXT = "#e4e4ef"
FG_MUTED = "#9090a8"
ACCENT = "#7c9cff"
ACCENT_2 = "#ff8a8a"

OP_LINEAR = "Linear Stretch"
OP_GAMMA = "Gamma Correction"
OP_HEQ = "Histogram Equalization"



def make_demo_image():
    """Create a low-contrast synthetic grayscale image for immediate testing."""
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    img = (np.sin(x / 20) * np.cos(y / 25) * 30 + 90 + 20 * (x / 255)).astype(np.float32)
    img[40:90, 40:90] = 60
    img[40:90, 166:216] = 110
    img[166:216, 40:90] = 150
    img[166:216, 166:216] = 180
    img += np.random.randn(256, 256) * 4
    return np.clip(img, 0, 255).astype(np.uint8)


class LabeledSlider(tk.Frame):
    """A labeled Tkinter slider that shows the current numeric value."""

    def __init__(self, master, text, frm, to, init, cb, fmt="{:.2f}", res=0.01):
        super().__init__(master, bg=BG_PANEL)
        self.cb = cb
        self.fmt = fmt
        self.res = res

        top = tk.Frame(self, bg=BG_PANEL)
        top.pack(fill="x", pady=(8, 0))
        tk.Label(top, text=text, bg=BG_PANEL, fg=FG_TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        self.val_lbl = tk.Label(
            top,
            text=fmt.format(init),
            bg=BG_PANEL,
            fg=ACCENT,
            font=("Consolas", 10, "bold"),
        )
        self.val_lbl.pack(side="right")

        self.var = tk.DoubleVar(value=init)
        ttk.Scale(
            self,
            from_=frm,
            to=to,
            variable=self.var,
            orient="horizontal",
            command=self._on_change,
        ).pack(fill="x", pady=(2, 6))

    def _on_change(self, _):
        value = self.var.get()
        if self.res >= 1:
            value = round(value)
        self.val_lbl.config(text=self.fmt.format(value))
        self.cb()

    def get(self):
        value = self.var.get()
        return round(value) if self.res >= 1 else value


# ============================================================
#                           GUI application
# ============================================================

class App:
    """Interactive GUI for comparing point operations and histograms."""

    def __init__(self, root):
        self.root = root
        self.root.title("Experiment 1 · Point Operations — Algorithm Version")
        self.root.geometry("1320x820")
        self.root.configure(bg=BG_DARK)
        self.img = make_demo_image()
        self.result = self.img.copy()
        self._style()
        self._ui()
        self._update()

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TScale",
            background=BG_PANEL,
            troughcolor="#3a3a50",
            bordercolor=BG_PANEL,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        style.configure(
            "TCombobox",
            fieldbackground="#3a3a50",
            background="#3a3a50",
            foreground=FG_TEXT,
            arrowcolor=FG_TEXT,
        )

    def _ui(self):
        side = tk.Frame(self.root, bg=BG_PANEL, width=290)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(
            side,
            text="⚙  Parameter Control",
            bg=BG_PANEL,
            fg=FG_TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=18, pady=(20, 6))
        tk.Frame(side, bg=ACCENT, height=2).pack(fill="x", padx=18)

        method_box = tk.Frame(side, bg=BG_PANEL)
        method_box.pack(fill="x", padx=18, pady=(16, 4))
        tk.Label(
            method_box,
            text="Processing Method",
            bg=BG_PANEL,
            fg=FG_TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self.op_var = tk.StringVar(value=OP_LINEAR)
        ttk.Combobox(
            method_box,
            textvariable=self.op_var,
            state="readonly",
            values=[OP_LINEAR, OP_GAMMA, OP_HEQ],
        ).pack(fill="x", pady=(4, 6))
        self.op_var.trace_add("write", lambda *args: self._on_op_change())

        slider_wrap = tk.Frame(side, bg=BG_PANEL)
        slider_wrap.pack(fill="x", padx=18)

        self.lo = LabeledSlider(slider_wrap, "Low Percentile (%)", 0, 49, 2, self._update, "{:.1f}", 0.5)
        self.hi = LabeledSlider(slider_wrap, "High Percentile (%)", 51, 100, 98, self._update, "{:.1f}", 0.5)
        self.olo = LabeledSlider(slider_wrap, "Output Lower Bound", 0, 128, 0, self._update, "{:.0f}", 1)
        self.ohi = LabeledSlider(slider_wrap, "Output Upper Bound", 128, 255, 255, self._update, "{:.0f}", 1)
        self.gm = LabeledSlider(slider_wrap, "γ (Gamma)", 0.1, 3.0, 1.0, self._update, "{:.2f}", 0.01)
        self.gc = LabeledSlider(slider_wrap, "Gain c", 0.2, 2.0, 1.0, self._update, "{:.2f}", 0.01)

        tk.Frame(side, bg=BG_PANEL, height=14).pack()
        self._button(side, "📁  Open Image", self._open)
        self._button(side, "💾  Save Result", self._save)
        self._button(side, "🔄  Reset Parameters", self._reset, ACCENT_2)

        info_panel = tk.Frame(side, bg="#2c2c40")
        info_panel.pack(side="bottom", fill="x", padx=12, pady=12)
        self.info = tk.Label(
            info_panel,
            text="",
            bg="#2c2c40",
            fg=FG_MUTED,
            font=("Consolas", 9),
            justify="left",
            anchor="w",
        )
        self.info.pack(fill="x", padx=10, pady=10)

        main = tk.Frame(self.root, bg=BG_LIGHT)
        main.pack(side="right", fill="both", expand=True)
        self.fig = plt.Figure(figsize=(11, 7), facecolor=BG_LIGHT)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        toolbar = NavigationToolbar2Tk(self.canvas, main)
        toolbar.update()
        toolbar.configure(bg=BG_LIGHT)

        self._on_op_change()

    def _button(self, parent, text, command, color=ACCENT):
        tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground="#5a7ae0",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            pady=8,
        ).pack(fill="x", padx=18, pady=4)

    def _on_op_change(self):
        for widget in (self.lo, self.hi, self.olo, self.ohi, self.gm, self.gc):
            widget.pack_forget()

        op = self.op_var.get()
        if op == OP_LINEAR:
            for widget in (self.lo, self.hi, self.olo, self.ohi):
                widget.pack(fill="x")
        elif op == OP_GAMMA:
            for widget in (self.gm, self.gc):
                widget.pack(fill="x")
        self._update()

    def _open(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Open Image", "Failed to read the selected image.")
            return

        self.img = cv2.resize(image, (256, 256))
        self._update()

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if path:
            cv2.imwrite(path, self.result)

    def _reset(self):
        defaults = [
            (self.lo, 2),
            (self.hi, 98),
            (self.olo, 0),
            (self.ohi, 255),
            (self.gm, 1.0),
            (self.gc, 1.0),
        ]
        for widget, value in defaults:
            widget.var.set(value)
            widget.val_lbl.config(text=widget.fmt.format(value))
        self._update()

    def _update(self):
        op = self.op_var.get()
        a_pt = None
        b_pt = None
        transfer_curve = None

        if op == OP_LINEAR:
            self.result, a_pt, b_pt = linear_stretch(
                self.img,
                self.lo.get(),
                self.hi.get(),
                int(self.olo.get()),
                int(self.ohi.get()),
            )
        elif op == OP_GAMMA:
            self.result = gamma_correction(self.img, self.gm.get(), self.gc.get())
        else:
            self.result, transfer_curve = histogram_equalization(self.img)

        hist_in = compute_histogram(self.img)
        hist_out = compute_histogram(self.result)

        self.fig.clear()
        grid = self.fig.add_gridspec(
            2,
            3,
            width_ratios=[1, 1.2, 1],
            hspace=0.4,
            wspace=0.32,
            left=0.05,
            right=0.97,
            top=0.93,
            bottom=0.08,
        )
        ax_img_in = self.fig.add_subplot(grid[0, 0])
        ax_transfer = self.fig.add_subplot(grid[:, 1])
        ax_img_out = self.fig.add_subplot(grid[0, 2])
        ax_hist_in = self.fig.add_subplot(grid[1, 0])
        ax_hist_out = self.fig.add_subplot(grid[1, 2])

        ax_img_in.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        ax_img_in.set_title("Original Image", fontsize=11, fontweight="bold")
        ax_img_in.axis("off")

        ax_img_out.imshow(self.result, cmap="gray", vmin=0, vmax=255)
        ax_img_out.set_title(f"Result: {op}", fontsize=11, fontweight="bold")
        ax_img_out.axis("off")

        ax_hist_in.bar(np.arange(256), hist_in, width=1.0, color="#4a90d9")
        ax_hist_in.set_xlim(0, 255)
        ax_hist_in.set_title("Original Histogram", fontsize=10)
        ax_hist_in.grid(alpha=0.25)

        ax_hist_out.bar(np.arange(256), hist_out, width=1.0, color="#ff8a65")
        ax_hist_out.set_xlim(0, 255)
        ax_hist_out.set_title("Processed Histogram", fontsize=10)
        ax_hist_out.grid(alpha=0.25)

        r = np.arange(256)
        ax_transfer.plot(r, r, "k--", alpha=0.3, lw=1, label="y = x (identity)")

        if op == OP_LINEAR:
            out_lo = int(self.olo.get())
            out_hi = int(self.ohi.get())
            transfer = (r - a_pt) / (b_pt - a_pt) * (out_hi - out_lo) + out_lo
            transfer = np.clip(transfer, out_lo, out_hi)
            transfer = np.clip(transfer, 0, 255)
            ax_transfer.plot(r, transfer, "#e74c3c", lw=2.5, label=f"a={a_pt}, b={b_pt}")
            ax_transfer.axvline(a_pt, color="#888", ls=":")
            ax_transfer.axvline(b_pt, color="#888", ls=":")
            info = (
                f"Method: Linear Stretch\n"
                f"a = {a_pt}, b = {b_pt}\n"
                f"Output range = [{out_lo}, {out_hi}]"
            )
        elif op == OP_GAMMA:
            gamma = self.gm.get()
            gain = self.gc.get()
            transfer = np.clip(gain * 255 * (r / 255) ** gamma, 0, 255)
            ax_transfer.plot(r, transfer, "#e74c3c", lw=2.5, label=f"γ={gamma:.2f}, c={gain:.2f}")
            info = f"Method: Gamma Correction\nγ = {gamma:.3f}\nc = {gain:.3f}"
        else:
            ax_transfer.plot(r, transfer_curve, "#e74c3c", lw=2.5, label="Equalization LUT")
            info = "Method: Histogram Equalization\nT(r) uses CDF-min correction"

        ax_transfer.set_xlim(0, 255)
        ax_transfer.set_ylim(0, 255)
        ax_transfer.set_title("Transfer Function T(r)", fontsize=12, fontweight="bold")
        ax_transfer.set_xlabel("Input Gray Level r")
        ax_transfer.set_ylabel("Output Gray Level T(r)")
        ax_transfer.legend(fontsize=9, framealpha=0.9)
        ax_transfer.grid(alpha=0.3)
        ax_transfer.set_aspect("equal", "box")

        self.canvas.draw()
        self.info.config(text=info)


# ============================================================
#                      Non-GUI notebook preview
# ============================================================


def build_preview_figure():
    """Create a quick non-GUI preview figure for notebook environments."""
    img = make_demo_image()
    stretched, _, _ = linear_stretch(img, 2, 98, 0, 255)
    gamma_img = gamma_correction(img, 0.6, 1.0)
    equalized, _ = histogram_equalization(img)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Original", "Linear Stretch", "Gamma γ=0.6", "Histogram Equalization"]
    images = [img, stretched, gamma_img, equalized]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig



def run_self_check():
    """Run lightweight checks for the core point-operation functions."""
    constant = np.full((16, 16), 128, dtype=np.uint8)
    equalized, _ = histogram_equalization(constant)
    assert np.array_equal(equalized, constant), "Constant-image equalization should remain unchanged."

    sample = np.arange(256, dtype=np.uint8).reshape(16, 16)
    stretched, a, b = linear_stretch(sample, 0, 100, 50, 200)
    assert a == 0 and b == 255, "Percentile endpoints are not correct."
    assert stretched.min() >= 50 and stretched.max() <= 200, "Linear stretch output range is wrong."

    hist = compute_histogram(sample)
    assert hist.sum() == sample.size, "Histogram count does not match image size."
    return "Self-check passed."



def main():
    """Launch the interactive Tkinter GUI."""
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__" and not _running_in_notebook():
    main()
