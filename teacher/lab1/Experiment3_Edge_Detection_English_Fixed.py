"""Experiment 3: Edge Detection.

This English version keeps the original core algorithmic structure:
- manual reflect padding
- manual 2-D filtering/convolution-style accumulation
- Sobel, Prewitt, Roberts, and Laplacian operators
- a from-scratch Canny pipeline with Gaussian smoothing, gradient computation,
  non-maximum suppression, double thresholding, and hysteresis connection

OpenCV is used only for image I/O, resizing, and drawing the built-in demo
image. The interactive Tkinter GUI is not launched automatically inside
Jupyter notebooks; call main() explicitly in a local desktop environment.
"""

import os
import sys
from typing import Dict, Optional, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
#                         Environment helpers
# ============================================================


def _running_in_notebook() -> bool:
    """Return True when the code is executed inside Jupyter/IPython."""
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell_name in {"ZMQInteractiveShell", "Shell"}
    except Exception:
        return False


def _has_graphical_display() -> bool:
    """Return True when a Tkinter window can probably be opened."""
    return sys.platform.startswith("win") or sys.platform == "darwin" or bool(os.environ.get("DISPLAY"))


# ============================================================
#                         Core algorithms
# ============================================================


def pad_image(img: np.ndarray, pH: int, pW: int, mode: str = "reflect") -> np.ndarray:
    """Pad a grayscale image.

    The experiment mainly uses reflect padding, matching the original code.
    The reflect rule mirrors pixels around the border without repeating the
    edge pixel, which is equivalent to OpenCV's BORDER_REFLECT_101 for normal
    kernel sizes.
    """
    if img.ndim != 2:
        raise ValueError("pad_image expects a 2-D grayscale image.")
    if pH < 0 or pW < 0:
        raise ValueError("Padding sizes must be non-negative.")
    if mode not in {"zero", "reflect"}:
        raise ValueError("mode must be either 'zero' or 'reflect'.")

    H, W = img.shape
    img_f = img.astype(np.float32)

    if pH == 0 and pW == 0:
        return img_f.copy()

    out = np.zeros((H + 2 * pH, W + 2 * pW), dtype=np.float32)
    out[pH:pH + H, pW:pW + W] = img_f

    if mode == "zero":
        return out

    # Fallback for extremely small test images. For the normal experiment
    # images, the manual slicing below follows the original implementation.
    if H <= pH or W <= pW:
        return np.pad(img_f, ((pH, pH), (pW, pW)), mode="reflect").astype(np.float32)

    if pH > 0:
        out[:pH, pW:pW + W] = img_f[pH:0:-1, :]
        out[pH + H:, pW:pW + W] = img_f[H - 2:H - 2 - pH:-1, :]

    if pW > 0:
        out[:, :pW] = out[:, 2 * pW:pW:-1]
        out[:, pW + W:] = out[:, pW + W - 2:W - 2:-1]

    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2-D filtering by shifted-image accumulation.

    This follows the original experiment code and applies the kernel in
    correlation form. For symmetric kernels such as Gaussian and Laplacian,
    this is identical to mathematical convolution. For first-order gradient
    kernels, the sign convention may differ, but the gradient magnitude used
    for edge display remains consistent.
    """
    if img.ndim != 2:
        raise ValueError("convolve2d expects a 2-D grayscale image.")
    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2-D array.")

    H, W = img.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    padded = pad_image(img.astype(np.float32), pH, pW, "reflect")
    out = np.zeros((H, W), dtype=np.float32)

    for m in range(kH):
        for n in range(kW):
            out += kernel[m, n] * padded[m:m + H, n:n + W]
    return out


def make_gaussian_kernel(k: int, sigma: float) -> np.ndarray:
    """Build a normalized 2-D Gaussian kernel from scratch."""
    k = int(k) | 1
    if k < 1:
        raise ValueError("Kernel size must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    r = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            x = i - r
            y = j - r
            kernel[i, j] = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

    total = float(kernel.sum())
    if total <= 0:
        raise ValueError("Gaussian kernel sum is zero.")
    return kernel / total


def gradient(img: np.ndarray, op: str = "Sobel") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute first-order gradients with Sobel, Prewitt, or Roberts kernels."""
    if op == "Sobel":
        kx = np.array(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=np.float32,
        )
        ky = kx.T
    elif op == "Prewitt":
        kx = np.array(
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]],
            dtype=np.float32,
        )
        ky = kx.T
    elif op == "Roberts":
        kx = np.array(
            [[1, 0],
             [0, -1]],
            dtype=np.float32,
        )
        ky = np.array(
            [[0, 1],
             [-1, 0]],
            dtype=np.float32,
        )
    else:
        raise ValueError("op must be one of: 'Sobel', 'Prewitt', or 'Roberts'.")

    gx = convolve2d(img.astype(np.float32), kx)
    gy = convolve2d(img.astype(np.float32), ky)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def laplacian(img: np.ndarray) -> np.ndarray:
    """Compute the 4-neighbor Laplacian second derivative response."""
    kernel = np.array(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=np.float32,
    )
    return convolve2d(img.astype(np.float32), kernel)


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly normalize any numeric array to uint8 in [0, 255]."""
    arr_f = arr.astype(np.float32)
    shifted = arr_f - float(arr_f.min())
    max_value = float(shifted.max())
    if max_value > 1e-6:
        shifted = shifted / max_value * 255.0
    return np.clip(shifted, 0, 255).astype(np.uint8)


def non_max_suppression(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Thin edges by keeping only local maxima along the gradient direction."""
    H, W = mag.shape
    angle = np.arctan2(gy, gx) * 180.0 / np.pi
    angle[angle < 0] += 180.0

    out = np.zeros_like(mag, dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a < 180):
                p, q = mag[i, j - 1], mag[i, j + 1]
            elif 22.5 <= a < 67.5:
                p, q = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                p, q = mag[i - 1, j], mag[i + 1, j]
            else:
                p, q = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if mag[i, j] >= p and mag[i, j] >= q:
                out[i, j] = mag[i, j]

    return out


def hysteresis(nms: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Apply double-threshold hysteresis connection for Canny edges."""
    if t_low < 0 or t_high < 0:
        raise ValueError("Canny thresholds must be non-negative.")
    if t_high <= t_low:
        t_high = t_low + 1

    H, W = nms.shape
    strong = nms > t_high

    # Keep the original double-threshold idea but avoid the common corner case
    # where T_low=0 would make zero-gradient pixels behave like weak edges.
    weak = (nms >= t_low) & (nms <= t_high) & (nms > 0)

    edges = np.zeros((H, W), dtype=np.uint8)
    edges[strong] = 255

    ys, xs = np.where(strong)
    stack = list(zip(ys.tolist(), xs.tolist()))

    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if weak[ni, nj] and edges[ni, nj] == 0:
                        edges[ni, nj] = 255
                        stack.append((ni, nj))

    return edges


def canny_edge(
    img: np.ndarray,
    ks: int,
    sigma: float,
    t_low: float,
    t_high: float,
    op: str = "Sobel",
    use_nms: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the complete manual Canny edge-detection pipeline.

    Steps:
    1. Gaussian smoothing
    2. Gradient computation
    3. Non-maximum suppression
    4. Double thresholding
    5. Hysteresis connection
    """
    ks = int(ks) | 1
    if t_high <= t_low:
        t_high = t_low + 1

    gaussian_kernel = make_gaussian_kernel(ks, sigma)
    blurred = convolve2d(img.astype(np.float32), gaussian_kernel)
    gx, gy, mag = gradient(blurred, op)
    nms = non_max_suppression(mag, gx, gy) if use_nms else mag.copy()
    edges = hysteresis(nms, t_low, t_high)
    return edges, mag, nms


def edge_statistics(edge_img: np.ndarray) -> Dict[str, float]:
    """Return simple edge-count metrics for a binary or grayscale edge image."""
    count = int(np.count_nonzero(edge_img))
    ratio = count / float(edge_img.size) if edge_img.size else 0.0
    return {
        "edge_pixels": count,
        "edge_ratio": ratio,
    }


def make_demo_image() -> np.ndarray:
    """Create the built-in grayscale demo image used by the GUI."""
    img = np.full((180, 180), 60, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 200, -1)
    cv2.circle(img, (130, 55), 28, 150, -1)
    cv2.line(img, (15, 140), (165, 140), 230, 2)
    cv2.putText(img, "EDGE", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 220, 2)
    return img


def apply_edge_detection(
    img: Optional[np.ndarray] = None,
    kernel_size: int = 5,
    sigma: float = 1.0,
    t_low: float = 30,
    t_high: float = 80,
    op: str = "Sobel",
    use_nms: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute all edge maps shown by the interface without launching the GUI."""
    if img is None:
        img = make_demo_image()

    if img.ndim != 2:
        raise ValueError("apply_edge_detection expects a grayscale image.")

    kernel_size = int(kernel_size) | 1
    t_high = max(float(t_high), float(t_low) + 1)

    gaussian_kernel = make_gaussian_kernel(kernel_size, sigma)
    blurred = convolve2d(img.astype(np.float32), gaussian_kernel)

    _, _, sobel_mag = gradient(blurred, "Sobel")
    _, _, prewitt_mag = gradient(blurred, "Prewitt")
    lap_response = np.abs(laplacian(blurred))

    canny, selected_mag, nms = canny_edge(
        img,
        kernel_size,
        sigma,
        float(t_low),
        float(t_high),
        op=op,
        use_nms=use_nms,
    )

    stats = edge_statistics(canny)

    return {
        "original": img,
        "blurred": blurred,
        "sobel": normalize_uint8(sobel_mag),
        "prewitt": normalize_uint8(prewitt_mag),
        "laplacian": normalize_uint8(lap_response),
        "canny": canny,
        "selected_gradient_magnitude": selected_mag,
        "nms": nms,
        "edge_pixels": np.array(stats["edge_pixels"]),
        "edge_ratio": np.array(stats["edge_ratio"]),
    }


def threshold_sweep(
    img: Optional[np.ndarray] = None,
    lows: Tuple[int, ...] = (20, 30, 40),
    highs: Tuple[int, ...] = (60, 80, 100),
    kernel_size: int = 5,
    sigma: float = 1.0,
    op: str = "Sobel",
) -> list:
    """Generate simple Canny threshold statistics for report tables."""
    if img is None:
        img = make_demo_image()

    rows = []
    for low in lows:
        for high in highs:
            if high <= low:
                continue
            edges, _, _ = canny_edge(img, kernel_size, sigma, low, high, op=op, use_nms=True)
            stats = edge_statistics(edges)
            rows.append({
                "T_low": int(low),
                "T_high": int(high),
                "edge_pixels": int(stats["edge_pixels"]),
                "edge_ratio_percent": 100.0 * float(stats["edge_ratio"]),
            })
    return rows


# ============================================================
#                         GUI
# ============================================================

BG_DARK = "#1e1e2e"
BG_PANEL = "#252537"
BG_LIGHT = "#f7f7fa"
FG_TEXT = "#e4e4ef"
FG_MUTED = "#9090a8"
ACCENT = "#7c9cff"
ACCENT_2 = "#ff8a8a"


class LabeledSlider(tk.Frame):
    """A compact slider with a value label."""

    def __init__(
        self,
        master,
        text: str,
        frm: float,
        to: float,
        init: float,
        callback,
        fmt: str = "{:.2f}",
        res: float = 0.01,
    ):
        super().__init__(master, bg=BG_PANEL)
        self.callback = callback
        self.fmt = fmt
        self.res = res

        top = tk.Frame(self, bg=BG_PANEL)
        top.pack(fill="x", pady=(8, 0))

        tk.Label(
            top,
            text=text,
            bg=BG_PANEL,
            fg=FG_TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

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
        self.callback()

    def set(self, value: float) -> None:
        self.var.set(value)
        self.val_lbl.config(text=self.fmt.format(round(value) if self.res >= 1 else value))

    def get(self):
        value = self.var.get()
        return round(value) if self.res >= 1 else value


class App:
    """Tkinter interface for interactive edge-detection experiments."""

    def __init__(self, root):
        self.root = root
        root.title("Experiment 3: Edge Detection")
        root.geometry("1340x830")
        root.configure(bg=BG_DARK)

        self.img = make_demo_image()
        self.canny = np.zeros_like(self.img)
        self._style()
        self._build_ui()
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

    def _build_ui(self):
        side = tk.Frame(self.root, bg=BG_PANEL, width=300)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(
            side,
            text="Parameter Controls",
            bg=BG_PANEL,
            fg=FG_TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=18, pady=(20, 6))
        tk.Frame(side, bg=ACCENT, height=2).pack(fill="x", padx=18)

        tk.Label(
            side,
            text="Preprocessing: Gaussian Smoothing",
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18, pady=(14, 0))

        wrap = tk.Frame(side, bg=BG_PANEL)
        wrap.pack(fill="x", padx=18)
        self.ks = LabeledSlider(wrap, "Gaussian Kernel Size", 3, 9, 5, self._update, "{:.0f}", 1)
        self.ks.pack(fill="x")
        self.sig = LabeledSlider(wrap, "Sigma", 0.3, 3.0, 1.0, self._update, "{:.2f}", 0.01)
        self.sig.pack(fill="x")

        op_box = tk.Frame(side, bg=BG_PANEL)
        op_box.pack(fill="x", padx=18, pady=(10, 4))
        tk.Label(
            op_box,
            text="Gradient Operator",
            bg=BG_PANEL,
            fg=FG_TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self.op = tk.StringVar(value="Sobel")
        ttk.Combobox(
            op_box,
            textvariable=self.op,
            state="readonly",
            values=["Sobel", "Prewitt", "Roberts"],
        ).pack(fill="x", pady=(4, 4))
        self.op.trace_add("write", lambda *args: self._update())

        tk.Label(
            side,
            text="Canny Double Threshold",
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18, pady=(8, 0))

        threshold_box = tk.Frame(side, bg=BG_PANEL)
        threshold_box.pack(fill="x", padx=18)
        self.tlo = LabeledSlider(threshold_box, "Low Threshold", 0, 200, 30, self._update, "{:.0f}", 1)
        self.tlo.pack(fill="x")
        self.thi = LabeledSlider(threshold_box, "High Threshold", 0, 400, 80, self._update, "{:.0f}", 1)
        self.thi.pack(fill="x")

        self.nms_on = tk.BooleanVar(value=True)
        tk.Checkbutton(
            side,
            text="Enable Non-Maximum Suppression (NMS)",
            variable=self.nms_on,
            bg=BG_PANEL,
            fg=FG_TEXT,
            activebackground=BG_PANEL,
            activeforeground=FG_TEXT,
            selectcolor=BG_PANEL,
            font=("Segoe UI", 10),
            command=self._update,
        ).pack(anchor="w", padx=18, pady=(8, 0))

        tk.Frame(side, bg=BG_PANEL, height=10).pack()
        self._button(side, "Open Image", self._open)
        self._button(side, "Save Canny Result", self._save)
        self._button(side, "Reset Parameters", self._reset, ACCENT_2)

        info_frame = tk.Frame(side, bg="#2c2c40")
        info_frame.pack(side="bottom", fill="x", padx=12, pady=12)
        self.info = tk.Label(
            info_frame,
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

        self.fig = plt.Figure(figsize=(12, 8), facecolor=BG_LIGHT)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        toolbar = NavigationToolbar2Tk(self.canvas, main)
        toolbar.update()
        toolbar.configure(bg=BG_LIGHT)

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

    def _open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Open Image", "The selected file could not be loaded as a grayscale image.")
            return

        self.img = cv2.resize(image, (180, 180))
        self._update()

    def _save(self):
        path = filedialog.asksaveasfilename(
            title="Save Canny Result",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        ok = cv2.imwrite(path, self.canny)
        if not ok:
            messagebox.showerror("Save Image", "The Canny result could not be saved.")

    def _reset(self):
        self.ks.set(5)
        self.sig.set(1.0)
        self.tlo.set(30)
        self.thi.set(80)
        self.op.set("Sobel")
        self.nms_on.set(True)
        self._update()

    def _update(self):
        kernel_size = int(self.ks.get()) | 1
        sigma = float(self.sig.get())
        t_low = int(self.tlo.get())
        t_high = max(int(self.thi.get()), t_low + 1)
        op = self.op.get()

        state = apply_edge_detection(
            self.img,
            kernel_size=kernel_size,
            sigma=sigma,
            t_low=t_low,
            t_high=t_high,
            op=op,
            use_nms=self.nms_on.get(),
        )

        self.canny = state["canny"]
        mag_now = state["selected_gradient_magnitude"]

        self.fig.clear()
        gs = self.fig.add_gridspec(
            2,
            3,
            hspace=0.42,
            wspace=0.3,
            left=0.06,
            right=0.97,
            top=0.94,
            bottom=0.08,
        )
        axes = [self.fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

        titles = [
            "Original",
            "Sobel",
            "Prewitt",
            "Gradient Magnitude + Canny Thresholds",
            "Laplacian",
            f"Canny ({op}{' + NMS' if self.nms_on.get() else ''})",
        ]
        images = [
            state["original"],
            state["sobel"],
            state["prewitt"],
            None,
            state["laplacian"],
            state["canny"],
        ]

        for ax, title, image in zip(axes, titles, images):
            if image is None:
                continue
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

        hist_ax = axes[3]
        max_mag = max(float(mag_now.max()), 1.0)
        hist_ax.hist(mag_now.ravel(), bins=80, color="#4a90d9", alpha=0.85, range=(0, max_mag))
        hist_ax.axvline(t_low, color="#27ae60", lw=2.4, label=f"Low = {t_low}")
        hist_ax.axvline(t_high, color="#e74c3c", lw=2.4, label=f"High = {t_high}")
        hist_ax.axvspan(t_low, t_high, color="gold", alpha=0.18, label="Hysteresis range")
        hist_ax.set_title(titles[3], fontsize=11, fontweight="bold")
        hist_ax.set_xlabel(r"$|\nabla I|$")
        hist_ax.set_ylabel("Pixel count (log scale)")
        hist_ax.set_yscale("log")
        hist_ax.legend(fontsize=9)
        hist_ax.grid(alpha=0.3)

        self.canvas.draw()

        edge_pixels = int(state["edge_pixels"])
        edge_ratio = float(state["edge_ratio"]) * 100.0
        high_note = "" if int(self.thi.get()) >= t_high else " (auto-adjusted)"
        self.info.config(
            text=(
                f"Gaussian: k={kernel_size}, sigma={sigma:.2f}\n"
                f"Operator: {op}\n"
                f"NMS: {'ON' if self.nms_on.get() else 'OFF'}\n"
                f"Thresholds: ({t_low}, {t_high}){high_note}\n"
                f"Edge pixels: {edge_pixels}\n"
                f"Edge ratio: {edge_ratio:.2f}%"
            )
        )


# ============================================================
#                         Non-GUI helpers
# ============================================================


def build_preview_figure(
    img: Optional[np.ndarray] = None,
    kernel_size: int = 5,
    sigma: float = 1.0,
    t_low: float = 30,
    t_high: float = 80,
    op: str = "Sobel",
    use_nms: bool = True,
):
    """Build a static preview figure without launching the Tkinter GUI."""
    state = apply_edge_detection(
        img,
        kernel_size=kernel_size,
        sigma=sigma,
        t_low=t_low,
        t_high=t_high,
        op=op,
        use_nms=use_nms,
    )

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    display_items = [
        ("Original", state["original"]),
        ("Sobel", state["sobel"]),
        ("Prewitt", state["prewitt"]),
        ("Laplacian", state["laplacian"]),
        (f"Canny ({op})", state["canny"]),
    ]

    flat_axes = axes.ravel()
    for ax, (title, image) in zip(flat_axes, display_items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    mag = state["selected_gradient_magnitude"]
    hist_ax = flat_axes[-1]
    hist_ax.hist(mag.ravel(), bins=80, range=(0, max(float(mag.max()), 1.0)))
    hist_ax.axvline(t_low, lw=2, label=f"Low = {t_low}")
    hist_ax.axvline(max(t_high, t_low + 1), lw=2, label=f"High = {max(t_high, t_low + 1)}")
    hist_ax.set_title("Gradient Magnitude Histogram")
    hist_ax.set_xlabel(r"$|\nabla I|$")
    hist_ax.set_ylabel("Pixel count")
    hist_ax.legend()

    fig.tight_layout()
    return fig


def run_self_check() -> str:
    """Run lightweight checks for the core edge-detection functions."""
    sample = make_demo_image()

    gaussian = make_gaussian_kernel(5, 1.0)
    assert gaussian.shape == (5, 5), "Gaussian kernel has the wrong shape."
    assert abs(float(gaussian.sum()) - 1.0) < 1e-5, "Gaussian kernel is not normalized."

    for op in ["Sobel", "Prewitt", "Roberts"]:
        gx, gy, mag = gradient(sample.astype(np.float32), op)
        assert gx.shape == sample.shape, f"{op} gx has the wrong shape."
        assert gy.shape == sample.shape, f"{op} gy has the wrong shape."
        assert mag.shape == sample.shape, f"{op} magnitude has the wrong shape."
        assert np.isfinite(mag).all(), f"{op} magnitude contains invalid values."

    lap = laplacian(sample.astype(np.float32))
    assert lap.shape == sample.shape, "Laplacian result has the wrong shape."
    assert np.isfinite(lap).all(), "Laplacian result contains invalid values."

    edges, mag, nms = canny_edge(sample, 5, 1.0, 30, 80, op="Sobel", use_nms=True)
    assert edges.shape == sample.shape, "Canny result has the wrong shape."
    assert edges.dtype == np.uint8, "Canny result must be uint8."
    assert set(np.unique(edges)).issubset({0, 255}), "Canny result should be binary."
    assert mag.shape == sample.shape, "Canny magnitude has the wrong shape."
    assert nms.shape == sample.shape, "NMS result has the wrong shape."

    state = apply_edge_detection(sample, kernel_size=5, sigma=1.0, t_low=30, t_high=80)
    for key in ["sobel", "prewitt", "laplacian", "canny"]:
        assert state[key].shape == sample.shape, f"{key} output has the wrong shape."

    rows = threshold_sweep(sample, lows=(20, 30), highs=(60, 80))
    assert len(rows) == 4, "Threshold sweep returned the wrong number of rows."
    assert all(row["edge_pixels"] >= 0 for row in rows), "Threshold sweep has invalid edge counts."

    return "Self-check passed."


def main():
    """Launch the interactive Tkinter GUI."""
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if _running_in_notebook():
        # Notebook users can call run_self_check(), build_preview_figure(), or main() explicitly.
        pass
    elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print(run_self_check())
        print("No graphical display was detected. Run main() in a local desktop environment to open the GUI.")
    else:
        try:
            main()
        except tk.TclError as exc:
            print(run_self_check())
            print(f"Could not open the Tkinter GUI: {exc}")
            print("Run main() in a local desktop environment to open the GUI.")
