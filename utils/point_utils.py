# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from plyfile import PlyData, PlyElement
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


def fetchPly(path: str, stride: int = 1):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T

    if stride > 1:
        print(
            f"Subsampling point cloud of length {positions.shape[0]} with stride {stride}..."
        )
        positions = positions[::stride]
        colors = colors[::stride]
        normals = normals[::stride]
        print(f"New point cloud size: {positions.shape[0]}")

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: np.ndarray, rgb: np.ndarray, normals: np.ndarray=None):
    """
    """

    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def project(point3d: np.ndarray, T_w2c: np.ndarray, calibK: np.ndarray, frame_h: int, frame_w: int):
    rot = T_w2c[:3, :3]
    t = T_w2c[:3, 3:]
    point3d_cam = rot @ point3d + t
    point3d_proj = calibK @ point3d_cam 

    u_proj = point3d_proj[0] / point3d_proj[2] 
    v_proj = point3d_proj[1] / point3d_proj[2]
    z = point3d_proj[2]
    
    if point3d.shape[-1] > 1:
        mask = (u_proj > 0) * (u_proj < frame_w) * (v_proj > 0) * (v_proj < frame_h) * (z > 0)
        return u_proj[mask > 0], v_proj[mask > 0], z[mask > 0], mask
    else:
        if u_proj < 0 or u_proj >= frame_w -1 or v_proj < 0 or v_proj >= frame_h - 1 and z > 0: 
            return None, None, None, None
        return u_proj, v_proj, z, None
