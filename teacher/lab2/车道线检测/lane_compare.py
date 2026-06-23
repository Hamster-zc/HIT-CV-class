# lane_compare_three_outputs.py
import sys
import cv2
import numpy as np
from collections import deque

# -----------------------
#  工具函数
# -----------------------
def ensure_odd(x):
    x = int(x)
    return x if x % 2 == 1 else x + 1

def convolve2d(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0).astype(np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(ih):
        for j in range(iw):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out

# -----------------------
#  手工实现：高斯、Sobel、非极大抑制、滞后阈值、标准 Hough（line）
# -----------------------
def gaussian_kernel(size, sigma=1.0):
    size = ensure_odd(size)
    r = size // 2
    x, y = np.mgrid[-r:r+1, -r:r+1]
    normal = 1.0 / (2.0 * np.pi * sigma * sigma)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma * sigma))) * normal
    return g

def manual_gaussian_blur(gray, kernel_size=5, sigma=1.0):
    k = gaussian_kernel(kernel_size, sigma)
    blurred = convolve2d(gray.astype(np.float32), k)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return blurred

def manual_sobel(gray):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    Ix = convolve2d(gray.astype(np.float32), Kx)
    Iy = convolve2d(gray.astype(np.float32), Ky)
    G = np.hypot(Ix, Iy)
    Theta = np.arctan2(Iy, Ix)
    return G, Theta

def non_max_suppression(gradient, direction):
    M, N = gradient.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180
    for i in range(1, M-1):
        for j in range(1, N-1):
            a = angle[i, j]
            q = 255
            r = 255
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = gradient[i, j+1]
                r = gradient[i, j-1]
            elif 22.5 <= a < 67.5:
                q = gradient[i+1, j-1]
                r = gradient[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q = gradient[i+1, j]
                r = gradient[i-1, j]
            elif 112.5 <= a < 157.5:
                q = gradient[i-1, j-1]
                r = gradient[i+1, j+1]
            if gradient[i, j] >= q and gradient[i, j] >= r:
                Z[i, j] = gradient[i, j]
            else:
                Z[i, j] = 0
    return Z

def hysteresis_threshold(img, low, high):
    strong = 255
    weak = 50
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    stack = deque(zip(strong_i.tolist(), strong_j.tolist()))
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= M or nj < 0 or nj >= N:
                    continue
                if res[ni, nj] == weak:
                    res[ni, nj] = strong
                    stack.append((ni, nj))
    res[res != strong] = 0
    return res

def manual_hough_lines(edge_img, rho_res=1, theta_res=np.pi/180, threshold=100):
    h, w = edge_img.shape
    diag = int(np.hypot(h, w))
    rhos = np.arange(-diag, diag + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(edge_img)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for (x, y) in zip(xs, ys):
        for t_idx in range(len(thetas)):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag
            accumulator[rho, t_idx] += 1
    lines = []
    idxs = np.where(accumulator > threshold)
    for r_idx, t_idx in zip(idxs[0], idxs[1]):
        rho = rhos[r_idx]
        theta = thetas[t_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        t = 1000
        x1 = int(x0 + t * (-b))
        y1 = int(y0 + t * (a))
        x2 = int(x0 - t * (-b))
        y2 = int(y0 - t * (a))
        lines.append([x1, y1, x2, y2])
    if len(lines) == 0:
        return np.empty((0, 1, 4), dtype=int)
    return np.array(lines).reshape(-1, 1, 4)

# -----------------------
#  OpenCV 内置管线封装
# -----------------------
def builtin_pipeline_edges(gray, gaussian_kernel_size=5, sigma=1.0, canny_low=50, canny_high=150):
    k = ensure_odd(gaussian_kernel_size)
    blurred = cv2.GaussianBlur(gray, (k, k), sigma)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges

def builtin_hough_lines(edge_img, rho=1, theta=np.pi/180, threshold=50, min_line_len=40, max_line_gap=20):
    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return np.empty((0,1,4), dtype=int)
    return lines

# -----------------------
#  ROI & 后处理/绘图
# -----------------------
def region_of_interest(image, vertices_ratio=(0.1, 0.9, 0.6)):
    h, w = image.shape[:2]
    left_x = int(w * vertices_ratio[0])
    right_x = int(w * vertices_ratio[1])
    top_y = int(h * (1 - vertices_ratio[2]))
    polygons = np.array([[
        (left_x, h),
        (right_x, h),
        (int(w*0.55), top_y),
        (int(w*0.45), top_y)
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    if len(image.shape) == 3:
        masked = cv2.bitwise_and(image, image, mask=mask)
    else:
        masked = cv2.bitwise_and(image, mask)
    return masked

def average_slope_intercept_from_lines(image, lines):
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []
    if lines is None or len(lines) == 0:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.hypot(y2 - y1, x2 - x1)
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left_lines.append((slope, intercept))
            left_weights.append(length)
        else:
            right_lines.append((slope, intercept))
            right_weights.append(length)
    left_lane = None
    right_lane = None
    if len(left_weights) > 0:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
    if len(right_weights) > 0:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 1e-6:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def draw_lane_lines_on_image(orig_image, lines, color=(0,255,0), thickness=8):
    line_img = np.zeros_like(orig_image)
    left_lane, right_lane = average_slope_intercept_from_lines(orig_image, lines)
    y1 = orig_image.shape[0]
    y2 = int(y1 * 0.6)
    for lane in (left_lane, right_lane):
        pts = make_line_points(y1, y2, lane)
        if pts is not None:
            cv2.line(line_img, pts[0], pts[1], color, thickness)
    return line_img

def scale_lines(lines, scale_x, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    if lines is None or len(lines) == 0:
        return lines
    scaled = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        sx1 = int(round(x1 * scale_x))
        sy1 = int(round(y1 * scale_y))
        sx2 = int(round(x2 * scale_x))
        sy2 = int(round(y2 * scale_y))
        scaled.append([[sx1, sy1, sx2, sy2]])
    return np.array(scaled, dtype=int).reshape(-1,1,4)

# -----------------------
#  单帧处理：同时运行 builtin & manual，并返回两个 overlay（原图尺寸）
# -----------------------
def process_frame_compare(orig_frame, scale=0.5, params=None):
    if params is None:
        params = {}
    orig_h, orig_w = orig_frame.shape[:2]
    if scale <= 0 or scale > 1.0:
        scale = 1.0
    small = cv2.resize(orig_frame, (int(orig_w*scale), int(orig_h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else orig_frame.copy()
    small_h, small_w = small.shape[:2]
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # builtin pipeline
    builtin_edges = builtin_pipeline_edges(small_gray,
                                          gaussian_kernel_size=params.get('gauss_k',5),
                                          sigma=params.get('gauss_sigma',1.0),
                                          canny_low=params.get('canny_low',50),
                                          canny_high=params.get('canny_high',150))
    builtin_roi = region_of_interest(builtin_edges, vertices_ratio=params.get('roi_ratio',(0.12,0.88,0.6)))
    builtin_lines = builtin_hough_lines(builtin_roi,
                                       rho=params.get('hough_rho',1),
                                       theta=params.get('hough_theta',np.pi/180),
                                       threshold=params.get('hough_thresh',40),
                                       min_line_len=params.get('min_line_len',40),
                                       max_line_gap=params.get('max_line_gap',30))

    # manual pipeline
    manual_blur = manual_gaussian_blur(small_gray, kernel_size=params.get('gauss_k',5), sigma=params.get('gauss_sigma',1.0))
    G, Theta = manual_sobel(manual_blur)
    nms = non_max_suppression(G, Theta)
    manual_edges = hysteresis_threshold(nms, low=params.get('canny_low',50), high=params.get('canny_high',150))
    manual_roi = region_of_interest(manual_edges, vertices_ratio=params.get('roi_ratio',(0.12,0.88,0.6)))
    manual_lines = manual_hough_lines(manual_roi,
                                     rho_res=params.get('hough_rho',1),
                                     theta_res=params.get('hough_theta',np.pi/180),
                                     threshold=params.get('manual_hough_thresh',100))

    # scale lines back to original size
    scale_x = orig_w / float(small_w)
    scale_y = orig_h / float(small_h)
    builtin_lines_up = scale_lines(builtin_lines, scale_x, scale_y)
    manual_lines_up = scale_lines(manual_lines, scale_x, scale_y)

    builtin_overlay = draw_lane_lines_on_image(orig_frame, builtin_lines_up, color=(0,255,0), thickness=8)
    manual_overlay = draw_lane_lines_on_image(orig_frame, manual_lines_up, color=(0,0,255), thickness=8)

    builtin_combined = cv2.addWeighted(orig_frame, 0.8, builtin_overlay, 0.6, 0)
    manual_combined = cv2.addWeighted(orig_frame, 0.8, manual_overlay, 0.6, 0)

    return builtin_combined, manual_combined

# -----------------------
#  VideoWriter 创建辅助（尝试 mp4v，失败回退 .avi XVID）
# -----------------------
def open_writer_try(path, size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if writer.isOpened():
        return writer, path
    alt = path.rsplit('.', 1)[0] + '.avi'
    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
    writer2 = cv2.VideoWriter(alt, fourcc2, fps, size)
    if writer2.isOpened():
        return writer2, alt
    return None, None

# -----------------------
#  新的视频处理函数：输出三份单独视频（原, builtin, manual）
# -----------------------
def process_video_three_outputs(input_video_path, output_base_path, scale=0.5, params=None):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("ERROR: cannot open input video:", input_video_path)
        return
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # 强制宽高为偶数
    if orig_width % 2 != 0:
        orig_width -= 1
    if orig_height % 2 != 0:
        orig_height -= 1
    out_size = (orig_width, orig_height)

    # 输出文件名：基于 output_base_path（带扩展名）
    # 例如 output_base_path = ".../output_compare.mp4" -> 会生成:
    #    .../output_compare_original.mp4, ..._builtin.mp4, ..._manual.mp4 (或回退为 .avi)
    base_root = output_base_path.rsplit('.', 1)[0]
    orig_out_path = base_root + '_original.mp4'
    builtin_out_path = base_root + '_builtin.mp4'
    manual_out_path = base_root + '_manual.mp4'

    orig_writer, orig_used = open_writer_try(orig_out_path, out_size, fps)
    if orig_writer is None:
        print("ERROR: cannot open writer for original output. Try different codec or reduce resolution.")
        cap.release()
        return
    builtin_writer, builtin_used = open_writer_try(builtin_out_path, out_size, fps)
    if builtin_writer is None:
        print("ERROR: cannot open writer for builtin output.")
        orig_writer.release()
        cap.release()
        return
    manual_writer, manual_used = open_writer_try(manual_out_path, out_size, fps)
    if manual_writer is None:
        print("ERROR: cannot open writer for manual output.")
        orig_writer.release()
        builtin_writer.release()
        cap.release()
        return

    print("Writers opened ->", orig_used, builtin_used, manual_used)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    print("Start processing video -> output size:", out_size, " fps:", fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # process frame: get two overlays at original size
        builtin_combined, manual_combined = process_frame_compare(frame, scale=scale, params=params)

        # 准备写入：确保 dtype、通道、尺寸一致
        def sanitize_frame(f):
            if f is None:
                return None
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[2] == 4:
                f = f[:, :, :3]
            if (f.shape[1], f.shape[0]) != out_size:
                f = cv2.resize(f, out_size, interpolation=cv2.INTER_AREA)
            return f

        f_orig = sanitize_frame(frame)
        f_builtin = sanitize_frame(builtin_combined)
        f_manual = sanitize_frame(manual_combined)

        if f_orig is None or f_builtin is None or f_manual is None:
            print("Warning: one of frames is None; skipping frame", frame_idx)
            continue

        orig_writer.write(f_orig)
        builtin_writer.write(f_builtin)
        manual_writer.write(f_manual)

        frame_idx += 1
        if frame_count > 0 and frame_idx % 10 == 0:
            print(f"Progress: {frame_idx}/{frame_count} frames written", end='\r')

    # release
    cap.release()
    orig_writer.release()
    builtin_writer.release()
    manual_writer.release()
    cv2.destroyAllWindows()
    print("\nDone. Outputs saved to:")
    print(orig_used)
    print(builtin_used)
    print(manual_used)

# -----------------------
#  主入口
# -----------------------
if __name__ == '__main__':
    input_path = r'C:\ZC\HIT\Compulsory_Courses\CV\CV_lab\teacher\lab2\inputs\last.mp4'
    output_base = r'C:\ZC\HIT\Compulsory_Courses\CV\CV_lab\teacher\lab2\outputs\output_compare.mp4'
    params = {
        'gauss_k': 5,
        'gauss_sigma': 1.0,
        'canny_low': 50,
        'canny_high': 150,
        'hough_rho': 1,
        'hough_theta': np.pi/180,
        'hough_thresh': 40,
        'manual_hough_thresh': 120,
        'min_line_len': 40,
        'max_line_gap': 30,
        'roi_ratio': (0.12, 0.88, 0.6)
    }
    process_video_three_outputs(input_path, output_base, scale=0.5, params=params)
    sys.exit(0)