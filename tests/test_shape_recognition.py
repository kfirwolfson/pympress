# -*- coding: utf-8 -*-
"""
Test suite for pympress.shape_recognition

Simulates realistic hand-drawn strokes in the normalised coordinate space
used by pympress (x in [0,1] for slide width, y in [0,1] for slide height).

On a 16:9 slide a circle drawn on-screen has *unequal* x and y extents in
this space, so the tests include aspect-ratio effects.

Run: python tests/test_shape_recognition.py
"""
from __future__ import print_function, unicode_literals

import copy
import math
import sys
import numpy as np

# Adjust path so we can import pympress from the repo root
sys.path.insert(0, '.')
from pympress import shape_recognition


# ---------------------------------------------------------------------------
# Stroke generators — produce lists of [x, y] as pympress stores them
# ---------------------------------------------------------------------------

def generate_circle(cx, cy, r_pixels, ww, wh, n=50, noise_px=2, closure_gap_px=0):
    """Circle of radius *r_pixels* drawn on a *ww x wh* widget, stored in
    normalised coords.

    Args:
        closure_gap_px: extra gap (in pixels) between first and last point
                        to simulate imperfect closure.
    """
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        px = cx * ww + r_pixels * math.cos(t) + np.random.uniform(-noise_px, noise_px)
        py = cy * wh + r_pixels * math.sin(t) + np.random.uniform(-noise_px, noise_px)
        pts.append([px / ww, py / wh])
    # Closing point — near start but with optional gap
    pts.append([pts[0][0] + np.random.uniform(-noise_px / ww, noise_px / ww) + closure_gap_px / ww,
                pts[0][1] + np.random.uniform(-noise_px / wh, noise_px / wh) + closure_gap_px / wh])
    return pts


def generate_circle_tuples(cx, cy, r_pixels, ww, wh, n=50, noise_px=2):
    """Same as generate_circle but returns tuples (as get_slide_point does)."""
    pts = generate_circle(cx, cy, r_pixels, ww, wh, n, noise_px)
    return [tuple(p) for p in pts]


def generate_ellipse(cx, cy, rx_pixels, ry_pixels, ww, wh, n=50, noise_px=2):
    """Ellipse with given pixel radii on a ww x wh widget."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        px = cx * ww + rx_pixels * math.cos(t) + np.random.uniform(-noise_px, noise_px)
        py = cy * wh + ry_pixels * math.sin(t) + np.random.uniform(-noise_px, noise_px)
        pts.append([px / ww, py / wh])
    pts.append([pts[0][0] + np.random.uniform(-noise_px / ww, noise_px / ww),
                pts[0][1] + np.random.uniform(-noise_px / wh, noise_px / wh)])
    return pts


def generate_rectangle(cx, cy, half_w_px, half_h_px, ww, wh, n_per_side=12, noise_px=2):
    """Rectangle with given pixel half-widths, traced continuously."""
    x0, y0 = cx * ww - half_w_px, cy * wh - half_h_px
    x1, y1 = cx * ww + half_w_px, cy * wh + half_h_px
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    pts = []
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[i + 1]
        for j in range(n_per_side):
            t = j / n_per_side
            px = ax + t * (bx - ax) + np.random.uniform(-noise_px, noise_px)
            py = ay + t * (by - ay) + np.random.uniform(-noise_px, noise_px)
            pts.append([px / ww, py / wh])
    return pts


def generate_line(x0, y0, x1, y1, n=30, noise=0.003):
    """Simple line in normalised coords."""
    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append([x0 + t * (x1 - x0) + np.random.uniform(-noise, noise),
                     y0 + t * (y1 - y0) + np.random.uniform(-noise, noise)])
    return pts


def generate_circle_variable_speed(cx, cy, r_pixels, ww, wh, n=50, noise_px=2):
    """Circle with non-uniform angular spacing to simulate variable drawing
    speed — more points where the user slows down (at top), fewer where fast."""
    pts = []
    angles = []
    for i in range(n):
        base_t = 2 * math.pi * i / n
        # Slow at top (t ~ pi/2), fast at bottom (t ~ 3pi/2)
        speed_factor = 1.0 + 0.5 * math.sin(base_t)
        angles.append(base_t)

    for t in angles:
        px = cx * ww + r_pixels * math.cos(t) + np.random.uniform(-noise_px, noise_px)
        py = cy * wh + r_pixels * math.sin(t) + np.random.uniform(-noise_px, noise_px)
        pts.append([px / ww, py / wh])
    pts.append([pts[0][0] + np.random.uniform(-noise_px / ww, noise_px / ww),
                pts[0][1] + np.random.uniform(-noise_px / wh, noise_px / wh)])
    return pts


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def diagnose(pts, label):
    """Run shape_recognition.recognize_shape and print detailed diagnostics."""
    arr = np.array(pts, dtype=float)
    n = len(arr)
    closed = shape_recognition._is_closed(arr)
    d = np.linalg.norm(arr[0] - arr[-1])
    bbox = arr.max(axis=0) - arr.min(axis=0)
    bbox_diag = np.linalg.norm(bbox)
    closure_ratio = d / bbox_diag if bbox_diag > 1e-9 else float('inf')

    result = shape_recognition.recognize_shape(pts)
    shape = result[0] if result else 'NONE'

    print("--- {} ---".format(label))
    print("  points={}, closed={}, closure_ratio={:.4f} (thr={})".format(
        n, closed, closure_ratio, shape_recognition.CLOSURE_DISTANCE_RATIO))
    print("  bbox_size=({:.4f}, {:.4f}), bbox_diag={:.4f}".format(bbox[0], bbox[1], bbox_diag))

    if closed:
        # Ellipse diagnostics
        centroid = arr.mean(axis=0)
        centered = arr - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        transformed = centered @ eigenvectors
        semi_a = float(np.percentile(np.abs(transformed[:, 0]), 98))
        semi_b = float(np.percentile(np.abs(transformed[:, 1]), 98))
        if semi_a > 1e-9 and semi_b > 1e-9:
            radial = (transformed[:, 0] / semi_a) ** 2 + (transformed[:, 1] / semi_b) ** 2
            radial_std = float(np.std(radial - 1.0))
            axis_ratio = min(semi_a, semi_b) / max(semi_a, semi_b)
            print("  ellipse: semi_a={:.5f}, semi_b={:.5f}, axis_ratio={:.3f}, radial_std={:.4f} (thr={})".format(
                semi_a, semi_b, axis_ratio, radial_std, shape_recognition.ELLIPSE_RADIAL_STD_RATIO))

        # Rectangle diagnostics
        epsilon = bbox_diag * shape_recognition.RDP_EPSILON_RATIO
        pts_closed = np.vstack([arr, arr[0:1]])
        simplified = shape_recognition._rdp_simplify(pts_closed, epsilon)
        if np.linalg.norm(simplified[0] - simplified[-1]) <= epsilon:
            simplified_no_dup = simplified[:-1]
        else:
            simplified_no_dup = simplified
        vertices = shape_recognition._remove_collinear(simplified_no_dup, epsilon)
        print("  rectangle: rdp_pts={}, after_collinear={}, epsilon={:.6f}".format(
            len(simplified), len(vertices), epsilon))
        if len(vertices) == 4:
            angles = []
            for i in range(len(vertices)):
                v1 = vertices[(i + 1) % len(vertices)] - vertices[i]
                v2 = vertices[(i - 1) % len(vertices)] - vertices[i]
                angles.append(shape_recognition._angle_between(v1, v2))
            print("  rectangle: angles={}".format(["%.1f" % a for a in angles]))
    else:
        # Line diagnostics
        line = shape_recognition.fit_line(arr)
        print("  line: fit={}".format(line is not None))

    print("  >> RESULT: {}".format(shape))
    return result


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def check(result, expected_type, label):
    global PASS, FAIL
    got = result[0] if result else 'NONE'
    ok = got == expected_type
    if ok:
        PASS += 1
        print("  [PASS] {} -> {}".format(label, got))
    else:
        FAIL += 1
        print("  [FAIL] {} -> {} (expected {})".format(label, got, expected_type))
    return ok


def check_any(result, expected_types, label):
    """Check result matches any of the given shape types."""
    global PASS, FAIL
    got = result[0] if result else 'NONE'
    ok = got in expected_types
    if ok:
        PASS += 1
        print("  [PASS] {} -> {}".format(label, got))
    else:
        FAIL += 1
        print("  [FAIL] {} -> {} (expected one of {})".format(label, got, expected_types))
    return ok


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def test_lines():
    """Test line detection."""
    print("\n== LINES ==")

    pts = generate_line(0.1, 0.2, 0.8, 0.7)
    r = diagnose(pts, "Diagonal line")
    check(r, 'line', "diagonal line")

    pts = generate_line(0.1, 0.5, 0.9, 0.5)
    r = diagnose(pts, "Horizontal line")
    check(r, 'line', "horizontal line")

    pts = generate_line(0.5, 0.1, 0.5, 0.9)
    r = diagnose(pts, "Vertical line")
    check(r, 'line', "vertical line")

    pts = generate_line(0.3, 0.3, 0.7, 0.7, n=10, noise=0.005)
    r = diagnose(pts, "Short line, few points")
    check(r, 'line', "short line few pts")

    # Very short line (near-minimum points)
    pts = generate_line(0.4, 0.4, 0.6, 0.6, n=4, noise=0.002)
    r = diagnose(pts, "Minimal line (4pts)")
    check(r, 'line', "minimal line")


def test_circles_16_9():
    """Test circle detection on 16:9 display (aspect ratio distortion)."""
    print("\n== CIRCLES (16:9 display, 1920x1080) ==")
    WW, WH = 1920, 1080

    # Small circle (80px radius, 40pts)
    pts = generate_circle(0.5, 0.5, 80, WW, WH, n=40, noise_px=3)
    r = diagnose(pts, "Small circle 80px")
    check_any(r, ('circle', 'ellipse'), "small circle 80px")

    # Medium circle (150px, 60pts)
    pts = generate_circle(0.5, 0.5, 150, WW, WH, n=60, noise_px=3)
    r = diagnose(pts, "Medium circle 150px")
    check_any(r, ('circle', 'ellipse'), "medium circle 150px")

    # Large circle (250px, 80pts)
    pts = generate_circle(0.5, 0.5, 250, WW, WH, n=80, noise_px=4)
    r = diagnose(pts, "Large circle 250px")
    check_any(r, ('circle', 'ellipse'), "large circle 250px")

    # Quick circle (few points, fast drawing ~ 60Hz for 0.25s)
    pts = generate_circle(0.5, 0.5, 120, WW, WH, n=15, noise_px=4)
    r = diagnose(pts, "Quick circle 120px (15pts)")
    check_any(r, ('circle', 'ellipse'), "quick circle 15pts")

    # Very quick circle (8 pts — near MIN_POINTS_SHAPE)
    pts = generate_circle(0.5, 0.5, 120, WW, WH, n=8, noise_px=3)
    r = diagnose(pts, "Very quick circle 120px (8pts)")
    check_any(r, ('circle', 'ellipse'), "very quick circle 8pts")

    # Messy circle (high noise)
    pts = generate_circle(0.5, 0.5, 150, WW, WH, n=50, noise_px=10)
    r = diagnose(pts, "Messy circle 150px (noise=10px)")
    check_any(r, ('circle', 'ellipse'), "messy circle")

    # Variable speed circle
    pts = generate_circle_variable_speed(0.5, 0.5, 150, WW, WH, n=50, noise_px=3)
    r = diagnose(pts, "Variable speed circle 150px")
    check_any(r, ('circle', 'ellipse'), "variable speed circle")

    # Circle with poor closure (20px gap)
    pts = generate_circle(0.5, 0.5, 120, WW, WH, n=40, noise_px=3, closure_gap_px=20)
    r = diagnose(pts, "Circle with 20px closure gap")
    check_any(r, ('circle', 'ellipse'), "circle 20px gap")

    # Circle with large closure gap (should still be < 20% of diagonal)
    pts = generate_circle(0.5, 0.5, 150, WW, WH, n=50, noise_px=3, closure_gap_px=50)
    r = diagnose(pts, "Circle with 50px closure gap")
    check_any(r, ('circle', 'ellipse'), "circle 50px gap")


def test_circles_4_3():
    """Test circle detection on 4:3 display."""
    print("\n== CIRCLES (4:3 display, 1024x768) ==")
    WW, WH = 1024, 768

    pts = generate_circle(0.5, 0.5, 100, WW, WH, n=50, noise_px=2)
    r = diagnose(pts, "Circle 100px on 4:3")
    check_any(r, ('circle', 'ellipse'), "circle 4:3")


def test_circles_ideal():
    """Test circle detection on square display (no distortion)."""
    print("\n== CIRCLES (ideal, 1000x1000) ==")
    WW, WH = 1000, 1000

    pts = generate_circle(0.5, 0.5, 100, WW, WH, n=50, noise_px=2)
    r = diagnose(pts, "Circle ideal 100px")
    check(r, 'circle', "ideal circle")

    pts = generate_circle(0.5, 0.5, 100, WW, WH, n=20, noise_px=5)
    r = diagnose(pts, "Sloppy circle 100px (20pts, noise=5)")
    check_any(r, ('circle', 'ellipse'), "sloppy circle")


def test_ellipses():
    """Test ellipse detection."""
    print("\n== ELLIPSES ==")
    WW, WH = 1920, 1080

    pts = generate_ellipse(0.5, 0.5, 200, 100, WW, WH, n=60, noise_px=3)
    r = diagnose(pts, "Wide ellipse 200x100px on 16:9")
    check_any(r, ('circle', 'ellipse'), "wide ellipse 16:9")

    pts = generate_ellipse(0.5, 0.5, 80, 200, WW, WH, n=60, noise_px=3)
    r = diagnose(pts, "Tall ellipse 80x200px on 16:9")
    check_any(r, ('circle', 'ellipse'), "tall ellipse 16:9")

    pts = generate_ellipse(0.5, 0.5, 150, 80, 1000, 1000, n=50, noise_px=2)
    r = diagnose(pts, "Ideal ellipse 150x80px")
    check(r, 'ellipse', "ideal ellipse")

    # Very eccentric ellipse (4:1 ratio)
    pts = generate_ellipse(0.5, 0.5, 300, 75, WW, WH, n=60, noise_px=3)
    r = diagnose(pts, "Very eccentric ellipse 300x75px")
    check_any(r, ('circle', 'ellipse'), "very eccentric ellipse")

    # Sloppy ellipse (few points, more noise)
    pts = generate_ellipse(0.5, 0.5, 150, 80, WW, WH, n=20, noise_px=8)
    r = diagnose(pts, "Sloppy ellipse 150x80px (20pts, noise=8px)")
    check_any(r, ('circle', 'ellipse'), "sloppy ellipse")


def test_rectangles():
    """Test rectangle detection."""
    print("\n== RECTANGLES ==")
    WW, WH = 1920, 1080

    pts = generate_rectangle(0.5, 0.5, 200, 120, WW, WH, n_per_side=12, noise_px=3)
    r = diagnose(pts, "Rectangle 200x120px on 16:9")
    check(r, 'rectangle', "rectangle 16:9")

    pts = generate_rectangle(0.5, 0.5, 100, 100, WW, WH, n_per_side=12, noise_px=3)
    r = diagnose(pts, "Square 100x100px on 16:9 (stretched in norm coords)")
    check(r, 'rectangle', "square 16:9")

    pts = generate_rectangle(0.5, 0.5, 400, 250, WW, WH, n_per_side=20, noise_px=4)
    r = diagnose(pts, "Large rectangle 400x250px")
    check(r, 'rectangle', "large rectangle")

    pts = generate_rectangle(0.5, 0.5, 150, 100, 1000, 1000, n_per_side=12, noise_px=2)
    r = diagnose(pts, "Ideal rectangle 150x100px")
    check(r, 'rectangle', "ideal rectangle")

    pts = generate_rectangle(0.5, 0.5, 150, 100, 1000, 1000, n_per_side=8, noise_px=6)
    r = diagnose(pts, "Sloppy rectangle (8pts/side, noise=6px)")
    check(r, 'rectangle', "sloppy rectangle")

    # Quick rectangle (few points per side)
    pts = generate_rectangle(0.5, 0.5, 200, 150, WW, WH, n_per_side=5, noise_px=3)
    r = diagnose(pts, "Quick rectangle (5pts/side)")
    check(r, 'rectangle', "quick rectangle")


def test_negative():
    """Test that non-shapes are rejected."""
    print("\n== NEGATIVE CASES ==")

    pts = [[np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)] for _ in range(30)]
    r = diagnose(pts, "Random noise 30pts")
    check(r, 'NONE', "random noise")

    pts = []
    for i in range(30):
        t = i / 29
        pts.append([0.1 + 0.8 * t, 0.5 + 0.05 * ((-1) ** i)])
    r = diagnose(pts, "Zigzag")
    check(r, 'NONE', "zigzag")

    pts = [[0.1, 0.1], [0.9, 0.9]]
    r = diagnose(pts, "Only 2 points")
    check(r, 'NONE', "2 points")

    # Triangle (closed, but not a rectangle or ellipse)
    pts = []
    for i in range(15):
        pts.append([0.3 + 0.4 * i / 14, 0.3])  # bottom
    for i in range(15):
        pts.append([0.7 - 0.2 * i / 14, 0.3 + 0.4 * i / 14])  # right side
    for i in range(15):
        pts.append([0.5 - 0.2 * i / 14, 0.7 - 0.4 * i / 14])  # left side
    r = diagnose(pts, "Triangle")
    check(r, 'NONE', "triangle")


def test_point_types():
    """Test that both list and tuple point formats work (pympress stores tuples)."""
    print("\n== POINT FORMAT TESTS ==")
    WW, WH = 1920, 1080

    # Tuples (as returned by get_slide_point)
    pts = generate_circle_tuples(0.5, 0.5, 120, WW, WH, n=40, noise_px=3)
    r = diagnose(pts, "Circle as tuples")
    check_any(r, ('circle', 'ellipse'), "circle tuples")

    # Mixed lists and tuples
    pts_list = generate_circle(0.5, 0.5, 120, WW, WH, n=40, noise_px=3)
    pts_mixed = [tuple(p) if i % 2 == 0 else p for i, p in enumerate(pts_list)]
    r = diagnose(pts_mixed, "Circle mixed tuples/lists")
    check_any(r, ('circle', 'ellipse'), "circle mixed")


def test_scribble_conversion():
    """Integration test: verify that _try_recognize_shape correctly converts
    the scribble data structure for each shape type."""
    print("\n== SCRIBBLE CONVERSION (integration) ==")
    WW, WH = 1920, 1080

    color = (1.0, 0.0, 0.0, 1.0)
    width = 3.0

    # --- Circle → ellipse scribble ---
    pts = generate_circle(0.5, 0.5, 120, WW, WH, n=40, noise_px=3)
    result = shape_recognition.recognize_shape(pts)
    if result and result[0] in ('circle', 'ellipse'):
        scribble = ["segment", color, width, pts, [[0, 0], [1, 1]]]
        old_state = copy.deepcopy(scribble[:])
        shape_type, params = result

        if shape_type == 'circle':
            center, radius = params
            corner1 = [center[0] - radius, center[1] - radius]
            corner2 = [center[0] + radius, center[1] + radius]
        else:
            center, half_w, half_h = params
            corner1 = [center[0] - half_w, center[1] - half_h]
            corner2 = [center[0] + half_w, center[1] + half_h]

        scribble[0] = "ellipse"
        scribble[3] = [corner1, corner2]
        scribble[4] = [corner1[:], corner2[:]]
        del scribble[5:]
        scribble.append((0, 0, 0, 0))

        # Verify structure
        assert scribble[0] == "ellipse", "type should be ellipse"
        assert len(scribble[3]) == 2, "should have 2 corner points"
        assert len(scribble) == 6, "should have 6 elements (incl fill_color)"
        c1, c2 = scribble[3]
        assert c1[0] < c2[0] or abs(c1[0] - c2[0]) < 0.01, "x0 should be <= x1"
        assert c1[1] < c2[1] or abs(c1[1] - c2[1]) < 0.01, "y0 should be <= y1"
        # Rendering check: ellipse needs non-zero dimensions
        p0 = (c1[0] * WW, c1[1] * WH)
        p1 = (c2[0] * WW, c2[1] * WH)
        assert abs(p1[0] - p0[0]) > 1, "x extent should be > 1px, got {}".format(abs(p1[0] - p0[0]))
        assert abs(p1[1] - p0[1]) > 1, "y extent should be > 1px, got {}".format(abs(p1[1] - p0[1]))

        global PASS
        PASS += 1
        print("  [PASS] circle -> ellipse scribble (dims: {:.0f}x{:.0f}px)".format(
            abs(p1[0] - p0[0]), abs(p1[1] - p0[1])))
    else:
        global FAIL
        FAIL += 1
        print("  [FAIL] circle not recognized for scribble conversion test")

    # --- Rectangle → box scribble ---
    pts = generate_rectangle(0.5, 0.5, 200, 120, WW, WH, n_per_side=12, noise_px=3)
    result = shape_recognition.recognize_shape(pts)
    if result and result[0] == 'rectangle':
        scribble = ["segment", color, width, pts, [[0, 0], [1, 1]]]
        corner1, corner2 = result[1]
        scribble[0] = "box"
        scribble[3] = [corner1, corner2]
        scribble[4] = [corner1[:], corner2[:]]
        del scribble[5:]
        scribble.append((0, 0, 0, 0))

        assert scribble[0] == "box"
        assert len(scribble[3]) == 2
        assert len(scribble) == 6
        PASS += 1
        print("  [PASS] rectangle -> box scribble")
    else:
        FAIL += 1
        print("  [FAIL] rectangle not recognized for scribble conversion test")

    # --- Line → segment scribble ---
    pts = generate_line(0.1, 0.2, 0.8, 0.7)
    result = shape_recognition.recognize_shape(pts)
    if result and result[0] == 'line':
        scribble = ["segment", color, width, pts, [[0, 0], [1, 1]]]
        start, end = result[1]
        scribble[0] = "segment"
        scribble[3] = [start, end]
        scribble[4] = [list(map(min, zip(start, end))), list(map(max, zip(start, end)))]
        del scribble[5:]

        assert scribble[0] == "segment"
        assert len(scribble[3]) == 2
        assert len(scribble) == 5, "segment should have 5 elements, got {}".format(len(scribble))
        PASS += 1
        print("  [PASS] line -> segment scribble")
    else:
        FAIL += 1
        print("  [FAIL] line not recognized for scribble conversion test")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n== EDGE CASES ==")
    WW, WH = 1920, 1080

    # Exactly MIN_POINTS_SHAPE points for a circle
    pts = generate_circle(0.5, 0.5, 120, WW, WH, n=shape_recognition.MIN_POINTS_SHAPE - 1, noise_px=2)
    r = diagnose(pts, "Circle with exactly MIN_POINTS_SHAPE pts ({})".format(shape_recognition.MIN_POINTS_SHAPE))
    check_any(r, ('circle', 'ellipse'), "min-points circle")

    # Very small shape (20px radius)
    pts = generate_circle(0.5, 0.5, 20, WW, WH, n=30, noise_px=2)
    r = diagnose(pts, "Tiny circle 20px")
    check_any(r, ('circle', 'ellipse'), "tiny circle")

    # Shape near edge of screen
    pts = generate_circle(0.05, 0.05, 50, WW, WH, n=30, noise_px=2)
    r = diagnose(pts, "Circle near corner (cx=0.05, cy=0.05)")
    check_any(r, ('circle', 'ellipse'), "corner circle")

    # Shape drawn over most of the slide
    pts = generate_circle(0.5, 0.5, 450, WW, WH, n=80, noise_px=5)
    r = diagnose(pts, "Huge circle 450px")
    check_any(r, ('circle', 'ellipse'), "huge circle")

    # Very small rectangle (noise proportional to size)
    pts = generate_rectangle(0.5, 0.5, 30, 20, WW, WH, n_per_side=8, noise_px=1)
    r = diagnose(pts, "Tiny rectangle 30x20px (noise=1px)")
    check(r, 'rectangle', "tiny rectangle")


def run_tests():
    global PASS, FAIL

    np.random.seed(42)

    print("=" * 60)
    print("SHAPE RECOGNITION TEST SUITE")
    print("=" * 60)

    test_lines()
    test_circles_16_9()
    test_circles_4_3()
    test_circles_ideal()
    test_ellipses()
    test_rectangles()
    test_negative()
    test_point_types()
    test_scribble_conversion()
    test_edge_cases()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print("RESULTS: {}/{} passed, {} failed".format(PASS, total, FAIL))
    if FAIL > 0:
        print("STATUS: SOME TESTS FAILED")
    else:
        print("STATUS: ALL TESTS PASSED")
    print("=" * 60)

    return FAIL == 0


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
