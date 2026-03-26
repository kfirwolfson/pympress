"""Quick diagnostic test for shape recognition with realistic hand-drawn-like strokes."""
import math
import numpy as np
from pympress import shape_recognition

def make_circle(cx, cy, r, n=60, noise=0.01):
    """Generate a noisy circle with non-uniform point spacing (like hand drawing)."""
    pts = []
    for i in range(n):
        # Non-uniform angle spacing to simulate variable drawing speed
        t = 2 * math.pi * i / n + np.random.uniform(-0.02, 0.02)
        x = cx + r * math.cos(t) + np.random.uniform(-noise, noise)
        y = cy + r * math.sin(t) + np.random.uniform(-noise, noise)
        pts.append([x, y])
    # Close the stroke (end near start)
    pts.append([pts[0][0] + np.random.uniform(-noise, noise),
                pts[0][1] + np.random.uniform(-noise, noise)])
    return pts

def make_ellipse(cx, cy, a, b, n=60, noise=0.01):
    """Generate a noisy ellipse."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n + np.random.uniform(-0.02, 0.02)
        x = cx + a * math.cos(t) + np.random.uniform(-noise, noise)
        y = cy + b * math.sin(t) + np.random.uniform(-noise, noise)
        pts.append([x, y])
    pts.append([pts[0][0] + np.random.uniform(-noise, noise),
                pts[0][1] + np.random.uniform(-noise, noise)])
    return pts

def make_rectangle(x0, y0, x1, y1, n_per_side=15, noise=0.005):
    """Generate a noisy rectangle stroke (going around the 4 sides)."""
    pts = []
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    for i in range(4):
        cx0, cy0 = corners[i]
        cx1, cy1 = corners[i + 1]
        for j in range(n_per_side):
            t = j / n_per_side
            x = cx0 + t * (cx1 - cx0) + np.random.uniform(-noise, noise)
            y = cy0 + t * (cy1 - cy0) + np.random.uniform(-noise, noise)
            pts.append([x, y])
    return pts

def make_line(x0, y0, x1, y1, n=30, noise=0.003):
    """Generate a noisy line."""
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * (x1 - x0) + np.random.uniform(-noise, noise)
        y = y0 + t * (y1 - y0) + np.random.uniform(-noise, noise)
        pts.append([x, y])
    return pts


np.random.seed(42)

print("=== Testing shape recognition ===\n")

# Test line
pts = make_line(0.1, 0.2, 0.8, 0.7)
result = shape_recognition.recognize_shape(pts)
print(f"Line:      {result[0] if result else None}")

# Test circle
pts = make_circle(0.5, 0.5, 0.15)
result = shape_recognition.recognize_shape(pts)
print(f"Circle:    {result[0] if result else None}")

# Test ellipse
pts = make_ellipse(0.5, 0.5, 0.2, 0.1)
result = shape_recognition.recognize_shape(pts)
print(f"Ellipse:   {result[0] if result else None}")

# Test rectangle
pts = make_rectangle(0.2, 0.2, 0.8, 0.7)
result = shape_recognition.recognize_shape(pts)
print(f"Rectangle: {result[0] if result else None}")

# Detailed diagnostics 
print("\n=== Detailed diagnostics ===\n")

# Circle diagnostics
pts_raw = make_circle(0.5, 0.5, 0.15)
pts_arr = np.array(pts_raw, dtype=float)
closed = shape_recognition._is_closed(pts_arr)
print(f"Circle: _is_closed = {closed}")
print(f"  n_points = {len(pts_arr)}")
d = np.linalg.norm(pts_arr[0] - pts_arr[-1])
bbox_diag = np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))
print(f"  start-end distance = {d:.6f}")
print(f"  bbox diagonal = {bbox_diag:.6f}")
print(f"  closure ratio = {d/bbox_diag:.6f} (threshold = {shape_recognition.CLOSURE_DISTANCE_RATIO})")

if closed:
    ell = shape_recognition.fit_ellipse(pts_arr)
    print(f"  fit_ellipse result: {ell}")
    rect = shape_recognition.fit_rectangle(pts_arr)
    print(f"  fit_rectangle result: {rect}")
else:
    print("  NOT CLOSED - ellipse/rect won't be tried")
    line = shape_recognition.fit_line(pts_arr)
    print(f"  fit_line result: {line is not None}")

# Rectangle diagnostics
pts_raw = make_rectangle(0.2, 0.2, 0.8, 0.7)
pts_arr = np.array(pts_raw, dtype=float)
closed = shape_recognition._is_closed(pts_arr)
print(f"\nRectangle: _is_closed = {closed}")
print(f"  n_points = {len(pts_arr)}")
d = np.linalg.norm(pts_arr[0] - pts_arr[-1])
bbox_diag = np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))
print(f"  start-end distance = {d:.6f}")
print(f"  bbox diagonal = {bbox_diag:.6f}")
if bbox_diag > 0:
    print(f"  closure ratio = {d/bbox_diag:.6f} (threshold = {shape_recognition.CLOSURE_DISTANCE_RATIO})")

if closed:
    rect = shape_recognition.fit_rectangle(pts_arr)
    print(f"  fit_rectangle result: {rect}")
    if rect is None:
        # Debug RDP
        epsilon = bbox_diag * shape_recognition.RDP_EPSILON_RATIO
        simplified = shape_recognition._rdp_simplify(pts_arr, epsilon)
        print(f"  RDP simplified to {len(simplified)} points (epsilon={epsilon:.6f})")
        print(f"  Simplified points:\n  {simplified}")
else:
    print("  NOT CLOSED - rectangle won't be tried")

# Ellipse diagnostics
pts_raw = make_ellipse(0.5, 0.5, 0.2, 0.1)
pts_arr = np.array(pts_raw, dtype=float)
closed = shape_recognition._is_closed(pts_arr)
print(f"\nEllipse: _is_closed = {closed}")
d = np.linalg.norm(pts_arr[0] - pts_arr[-1])
bbox_diag = np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))
print(f"  start-end distance = {d:.6f}")
print(f"  bbox diagonal = {bbox_diag:.6f}")
if bbox_diag > 0:
    print(f"  closure ratio = {d/bbox_diag:.6f} (threshold = {shape_recognition.CLOSURE_DISTANCE_RATIO})")

if closed:
    ell = shape_recognition.fit_ellipse(pts_arr)
    print(f"  fit_ellipse result: {ell}")
    if ell is None:
        # Debug individually
        centroid = pts_arr.mean(axis=0)
        centered = pts_arr - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        transformed = centered @ eigenvectors
        semi_a = float(np.percentile(np.abs(transformed[:, 0]), 95))
        semi_b = float(np.percentile(np.abs(transformed[:, 1]), 95))
        print(f"  semi_a={semi_a:.6f}, semi_b={semi_b:.6f}")
        radial = (transformed[:, 0] / semi_a) ** 2 + (transformed[:, 1] / semi_b) ** 2
        radial_std = float(np.std(radial - 1.0))
        print(f"  radial_std = {radial_std:.6f} (threshold = {shape_recognition.ELLIPSE_RADIAL_STD_RATIO})")
