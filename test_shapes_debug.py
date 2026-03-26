"""Debug specific shape recognition failures — now testing fixes."""
import math
import numpy as np
from pympress import shape_recognition

np.random.seed(123)

def test(name, pts):
    result = shape_recognition.recognize_shape(pts)
    status = result[0] if result else 'NONE'
    print("  {}: {}".format(name, status))
    return result

print("=== Shape tests with realistic hand-drawn noise ===\n")

# Quick circle (12pts)
pts = []
for i in range(12):
    t = 2 * math.pi * i / 12
    pts.append([0.5 + 0.12 * math.cos(t) + np.random.uniform(-0.015, 0.015),
                0.5 + 0.12 * math.sin(t) + np.random.uniform(-0.015, 0.015)])
pts.append([pts[0][0] + np.random.uniform(-0.01, 0.01),
            pts[0][1] + np.random.uniform(-0.01, 0.01)])
test("Quick circle (12pts, noise=0.015)", pts)

# Quick circle (20pts)
np.random.seed(42)
pts = []
for i in range(20):
    t = 2 * math.pi * i / 20
    pts.append([0.5 + 0.15 * math.cos(t) + np.random.uniform(-0.02, 0.02),
                0.5 + 0.15 * math.sin(t) + np.random.uniform(-0.02, 0.02)])
pts.append([pts[0][0] + np.random.uniform(-0.02, 0.02),
            pts[0][1] + np.random.uniform(-0.02, 0.02)])
test("Quick circle (20pts, noise=0.02)", pts)

# Sloppy ellipse (25pts)
np.random.seed(123)
pts = []
for i in range(25):
    t = 2 * math.pi * i / 25
    pts.append([0.5 + 0.25 * math.cos(t) + np.random.uniform(-0.025, 0.025),
                0.5 + 0.1 * math.sin(t) + np.random.uniform(-0.025, 0.025)])
pts.append([pts[0][0] + 0.01, pts[0][1] + 0.01])
test("Sloppy ellipse (25pts, noise=0.025)", pts)

# Sloppy rectangle (no explicit closing point)
np.random.seed(123)
pts = []
corners = [(0.2, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7), (0.2, 0.3)]
for i in range(4):
    cx0, cy0 = corners[i]
    cx1, cy1 = corners[i + 1]
    for j in range(10):
        t = j / 10
        pts.append([cx0 + t * (cx1 - cx0) + np.random.uniform(-0.01, 0.01),
                     cy0 + t * (cy1 - cy0) + np.random.uniform(-0.01, 0.01)])
test("Sloppy rect (40pts, noise=0.01)", pts)

# Rectangle starting mid-side
np.random.seed(42)
pts = []
# Start at middle of top side
side_points = [
    [(0.5, 0.2), (0.8, 0.2)],
    [(0.8, 0.2), (0.8, 0.7)],
    [(0.8, 0.7), (0.2, 0.7)],
    [(0.2, 0.7), (0.2, 0.2)],
    [(0.2, 0.2), (0.5, 0.2)],
]
for s0, s1 in side_points:
    for j in range(8):
        t = j / 8
        pts.append([s0[0] + t * (s1[0] - s0[0]) + np.random.uniform(-0.005, 0.005),
                     s0[1] + t * (s1[1] - s0[1]) + np.random.uniform(-0.005, 0.005)])
test("Rect mid-side start (40pts, noise=0.005)", pts)

# Clean line (should still work)
np.random.seed(42)
pts = []
for i in range(30):
    t = i / 29
    pts.append([0.1 + t * 0.7 + np.random.uniform(-0.003, 0.003),
                0.2 + t * 0.5 + np.random.uniform(-0.003, 0.003)])
test("Clean line (30pts)", pts)

# Random noise (should NOT recognize)
np.random.seed(42)
pts = [[np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)] for _ in range(30)]
test("Random noise (30pts)", pts)

print("\nAll tests complete.")
