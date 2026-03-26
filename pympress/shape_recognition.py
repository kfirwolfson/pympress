# -*- coding: utf-8 -*-
#
#       shape_recognition.py
#
#       Copyright 2026 Cimbali <me@cimba.li>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
"""
:mod:`pympress.shape_recognition` -- Detect geometric shapes from freehand strokes
-----------------------------------------------------------------------------------

Pure math module using numpy. No GTK imports.

When the user draws freehand with Shift held, the stroke points are analyzed
and converted to a geometric shape if the fit is good enough.

Supported shapes: line, circle, ellipse, rectangle.
"""

from __future__ import print_function, unicode_literals

import math
import numpy as np


# --- Thresholds ---
LINE_MAX_DEVIATION_RATIO = 0.05
ELLIPSE_RADIAL_STD_RATIO = 0.25
CIRCLE_AXIS_RATIO = 0.90
CLOSURE_DISTANCE_RATIO = 0.20
RECT_ANGLE_TOLERANCE = 25  # degrees
RDP_EPSILON_RATIO = 0.03
MIN_POINTS_LINE = 3
MIN_POINTS_SHAPE = 8


def _is_closed(pts):
    """ Check whether a stroke's start and end are close together.

    Args:
        pts (`numpy.ndarray`): Nx2 array of points

    Returns:
        `bool`: True if the stroke is approximately closed
    """
    d = np.linalg.norm(pts[0] - pts[-1])
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if bbox_diag < 1e-9:
        return False
    return d / bbox_diag < CLOSURE_DISTANCE_RATIO


def fit_line(pts):
    """ Fit a line to 2D points using SVD.

    Args:
        pts (`numpy.ndarray`): Nx2 array of points

    Returns:
        `tuple` or None: (start, end) as [x,y] lists, or None if the fit is poor
    """
    if len(pts) < MIN_POINTS_LINE:
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, s, vt = np.linalg.svd(centered, full_matrices=False)

    # Direction of best-fit line
    direction = vt[0]

    # Project all points onto the line
    projections = centered @ direction
    residuals = centered - np.outer(projections, direction)
    max_deviation = float(np.max(np.sqrt(np.sum(residuals ** 2, axis=1))))

    # Use extent along principal axis (not stroke arc length, which inflates for zigzag paths)
    extent = float(projections.max() - projections.min())
    if extent < 1e-9:
        return None

    if max_deviation / extent > LINE_MAX_DEVIATION_RATIO:
        return None

    # Project first and last points onto line to get clean endpoints
    t_first = float(np.dot(pts[0] - centroid, direction))
    t_last = float(np.dot(pts[-1] - centroid, direction))

    start = centroid + t_first * direction
    end = centroid + t_last * direction

    return [start.tolist(), end.tolist()]


def fit_ellipse(pts):
    """ Fit an ellipse (or circle) to 2D points using PCA-based approach.

    Args:
        pts (`numpy.ndarray`): Nx2 array of points (should form a closed stroke)

    Returns:
        `tuple` or None: (center, semi_a, semi_b, is_circle) where center is [x,y],
        semi_a/semi_b are axis-aligned half-widths, and is_circle is True if axes
        are within 90% of each other. Returns None if the fit is poor.
    """
    if len(pts) < MIN_POINTS_SHAPE:
        return None

    if not _is_closed(pts):
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA: eigendecompose the covariance matrix
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Transform points to principal-axis frame
    transformed = centered @ eigenvectors

    # In the principal frame, estimate semi-axes from the extent of the points.
    # For a uniform ellipse traced by hand, the semi-axis is ~ max extent along each axis.
    # Use a robust estimator: percentile-based half-range (98th to avoid underestimation from noise)
    semi_a = float(np.percentile(np.abs(transformed[:, 0]), 98))
    semi_b = float(np.percentile(np.abs(transformed[:, 1]), 98))

    if semi_a < 1e-9 or semi_b < 1e-9:
        return None

    # Quality check: how well do points lie on the ellipse?
    # Normalized radial distance: (x/a)^2 + (y/b)^2 should be ~1
    radial = (transformed[:, 0] / semi_a) ** 2 + (transformed[:, 1] / semi_b) ** 2
    radial_std = float(np.std(radial - 1.0))

    if radial_std > ELLIPSE_RADIAL_STD_RATIO:
        return None

    # Check if close enough to a circle
    axis_ratio = min(semi_a, semi_b) / max(semi_a, semi_b)
    is_circle = axis_ratio >= CIRCLE_AXIS_RATIO

    if is_circle:
        radius = (semi_a + semi_b) / 2.0
        semi_a = semi_b = radius

    # Convert back to original coordinate frame
    # The axis-aligned bounding box of the ellipse in the original frame
    # For axis-aligned ellipses (which pympress supports), we need the AABB
    # of the rotated ellipse.
    # Parametric: x(t) = a*cos(t)*e1x + b*sin(t)*e2x + cx
    #             y(t) = a*cos(t)*e1y + b*sin(t)*e2y + cy
    # Max x: sqrt((a*e1x)^2 + (b*e2x)^2), similarly for y
    e1 = eigenvectors[:, 0]
    e2 = eigenvectors[:, 1]
    half_w = math.sqrt((semi_a * e1[0]) ** 2 + (semi_b * e2[0]) ** 2)
    half_h = math.sqrt((semi_a * e1[1]) ** 2 + (semi_b * e2[1]) ** 2)

    center = centroid.tolist()
    return (center, half_w, half_h, is_circle)


def _rdp_simplify(pts, epsilon):
    """ Ramer-Douglas-Peucker polyline simplification.

    Args:
        pts (`numpy.ndarray`): Nx2 array of points
        epsilon (`float`): maximum distance threshold

    Returns:
        `numpy.ndarray`: simplified array of points
    """
    if len(pts) <= 2:
        return pts

    # Find the point farthest from the line between first and last
    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        dists = np.sqrt(np.sum((pts - start) ** 2, axis=1))
    else:
        line_unit = line_vec / line_len
        proj = np.dot(pts - start, line_unit)
        proj = np.clip(proj, 0, line_len)
        nearest = start + np.outer(proj, line_unit)
        dists = np.sqrt(np.sum((pts - nearest) ** 2, axis=1))

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = _rdp_simplify(pts[:max_idx + 1], epsilon)
        right = _rdp_simplify(pts[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def _angle_between(v1, v2):
    """ Angle between two 2D vectors in degrees.

    Args:
        v1 (`numpy.ndarray`): first vector
        v2 (`numpy.ndarray`): second vector

    Returns:
        `float`: angle in degrees [0, 180]
    """
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def _remove_collinear(vertices, epsilon):
    """ Remove vertices that are collinear with their neighbours.

    A vertex is considered collinear (i.e. it lies on a straight side rather
    than at a real corner) when its perpendicular distance to the line through
    its two neighbours is less than *epsilon*.

    Args:
        vertices (`numpy.ndarray`): Mx2 array of polygon vertices
        epsilon (`float`): distance threshold

    Returns:
        `numpy.ndarray`: filtered array with collinear vertices removed
    """
    keep = []
    n = len(vertices)
    for i in range(n):
        prev_pt = vertices[(i - 1) % n]
        curr_pt = vertices[i]
        next_pt = vertices[(i + 1) % n]
        edge = next_pt - prev_pt
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            keep.append(i)
            continue
        # Perpendicular distance from curr_pt to the line prev→next
        dist = abs(np.cross(edge, curr_pt - prev_pt)) / edge_len
        if dist > epsilon:
            keep.append(i)
    return vertices[keep] if keep else vertices


def fit_rectangle(pts):
    """ Detect a rectangle from a closed stroke using RDP simplification.

    Args:
        pts (`numpy.ndarray`): Nx2 array of points (should form a closed stroke)

    Returns:
        `tuple` or None: (corner1, corner2) as [x,y] lists for the axis-aligned
        bounding box, or None if the shape is not a rectangle.
    """
    if len(pts) < MIN_POINTS_SHAPE:
        return None

    if not _is_closed(pts):
        return None

    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    epsilon = bbox_diag * RDP_EPSILON_RATIO

    # Explicitly close the polygon before simplification so RDP preserves
    # identical first/last endpoints (RDP always keeps its first and last points).
    pts_closed = np.vstack([pts, pts[0:1]])
    simplified = _rdp_simplify(pts_closed, epsilon)

    # We expect ~5 points (4 corners + closing duplicate).
    n = len(simplified)
    if n < 4 or n > 7:
        return None

    # Drop the closing duplicate (first and last should now be identical)
    if np.linalg.norm(simplified[0] - simplified[-1]) <= epsilon:
        simplified = simplified[:-1]

    # If the user started mid-side, RDP may keep that point as a vertex
    # because it is an endpoint.  Remove any vertex that is collinear with
    # its neighbours (i.e. lies on a side, not at a corner).
    vertices = _remove_collinear(simplified, epsilon)
    n_verts = len(vertices)

    if n_verts != 4:
        return None

    # Check all angles are close to 90°
    for i in range(n_verts):
        v1 = vertices[(i + 1) % n_verts] - vertices[i]
        v2 = vertices[(i - 1) % n_verts] - vertices[i]
        angle = _angle_between(v1, v2)
        if abs(angle - 90.0) > RECT_ANGLE_TOLERANCE:
            return None

    # Return axis-aligned bounding box of the 4 vertices
    mins = vertices.min(axis=0).tolist()
    maxs = vertices.max(axis=0).tolist()
    return (mins, maxs)


def recognize_shape(points):
    """ Analyze a freehand stroke and detect if it matches a geometric shape.

    Args:
        points (`list`): list of [x, y] coordinate pairs (normalized 0-1)

    Returns:
        `tuple` or None: (shape_type, shape_params) where shape_type is one of
        'line', 'circle', 'ellipse', 'rectangle', or None if no shape detected.

        Shape params:
        - 'line': ([x0, y0], [x1, y1])
        - 'circle': ([cx, cy], radius)
        - 'ellipse': ([cx, cy], half_width, half_height)
        - 'rectangle': ([x_min, y_min], [x_max, y_max])
    """
    if not points or len(points) < MIN_POINTS_LINE:
        return None

    pts = np.array(points, dtype=float)

    closed = _is_closed(pts)

    if closed:
        # Try rectangle first (most specific closed shape)
        rect = fit_rectangle(pts)
        if rect is not None:
            return ('rectangle', rect)

        # Try ellipse/circle
        ell = fit_ellipse(pts)
        if ell is not None:
            center, half_w, half_h, is_circle = ell
            if is_circle:
                return ('circle', (center, (half_w + half_h) / 2.0))
            else:
                return ('ellipse', (center, half_w, half_h))

    # Try line (works for open strokes)
    line = fit_line(pts)
    if line is not None:
        return ('line', tuple(line))

    return None
