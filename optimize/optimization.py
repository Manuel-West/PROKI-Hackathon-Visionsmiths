import torch as torch
import numpy as np

from torch.autograd import Variable as V
from typing import Tuple, Union, Optional, List
from ncopt.functions import ObjectiveOrConstraint
from ncopt.sqpgs import SQPGS



def affine_transform(input_img: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply affine transformation to an image using bilinear interpolation.

    Args:
        input_img: Input image tensor of shape (height, width)
        transform_matrix: 3x3 affine transformation matrix

    Returns:
        Transformed image tensor of same shape as input
    """
    xsize, ysize = input_img.shape
    coor = create_coordinate_grid(xsize,ysize)
    xyprime = torch.tensordot(coor, torch.t(transform_matrix), dims=1)
    return bilinear_interpolate(input_img, xyprime)


def create_coordinate_grid(xsize: int, ysize: int) -> torch.Tensor:
    """
    Create a coordinate grid for the image.

    Args:
        xsize: Height of the image
        ysize: Width of the image

    Returns:
        Tensor of shape (height, width, 3) containing homogeneous coordinates
    """
    coor=np.asarray(np.meshgrid(np.arange(xsize), np.arange(ysize)))
    coor=np.rollaxis(coor,0,3)
    coor=np.rollaxis(coor,0,2)
    onesMatrix = np.ones((xsize,ysize,1))
    coorMatrix = np.concatenate((coor, onesMatrix), axis=2)
    return V(torch.tensor(coorMatrix, dtype=torch.float32), requires_grad=True)


def bilinear_interpolate(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Perform bilinear interpolation on the image using transformed coordinates.

    Args:
        img: Input image tensor
        coords: Transformed coordinates tensor

    Returns:
        Interpolated image tensor
    """
    xsize, ysize = img.shape

    # Extract x and y coordinates
    x_coords = coords[..., 0]
    y_coords = coords[..., 1]

    # Calculate floor coordinates and ensure they're within bounds
    x0 = torch.clamp(torch.floor(x_coords).long(), 1, xsize - 2)
    y0 = torch.clamp(torch.floor(y_coords).long(), 1, ysize - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    # Calculate interpolation weights
    wx = x_coords - x0.to(x_coords.dtype)
    wy = y_coords - y0.to(y_coords.dtype)

    # Gather pixel values for interpolation
    v00 = img[x0, y0]
    v01 = img[x0, y1]
    v10 = img[x1, y0]
    v11 = img[x1, y1]

    # Perform bilinear interpolation
    return (
            (1 - wx) * ((1 - wy) * v00 + wy * v01) +
            wx * ((1 - wy) * v10 + wy * v11)
    )

def get_affine_matrix(center: List[float], angle: torch.Tensor, translate: List[float]) -> torch.Tensor:
    rot = torch.deg2rad(angle)
    cx, cy = center
    tx, ty = translate

    angle_cos = torch.cos(rot)
    angle_sin = torch.sin(rot)

    matrix = torch.stack([
        torch.stack([angle_cos, -angle_sin, angle_cos * (-tx - cx) - angle_sin * (-ty - cy) + cx]),
        torch.stack([angle_sin, angle_cos, angle_sin * (-tx - cx) + angle_cos * (-ty - cy) + cy]),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    ])
    return matrix.to(dtype=torch.float32)  # Ensure float32

class CoordinateObjective(torch.nn.Module):
    def __init__(self, template, target_coords, template_center=None):
        super().__init__()
        self.template = template
        self.target_x, self.target_y = target_coords
        if template_center is None:
            self.template_center = [template.shape[1] // 2, template.shape[0] // 2]
        else:
            self.template_center = template_center

    def forward(self, params):
        if len(params.shape) > 1:
            params = params.squeeze(0)
        angle, tx, ty = params[0], params[1], params[2]
        matrix = get_affine_matrix(
            center=self.template_center,
            angle=angle,
            translate=[tx, ty]
        )
        center_point = torch.tensor([
            [self.template_center[0]],
            [self.template_center[1]],
            [1.0]
        ], dtype=torch.float32)
        transformed = matrix @ center_point
        error_x = (transformed[0, 0] - self.target_x) ** 2
        error_y = (transformed[1, 0] - self.target_y) ** 2
        output = error_x + error_y
        return output.unsqueeze(0).unsqueeze(1)  # Shape [batch_size, 1]

class AlignmentConstraint(torch.nn.Module):
    def __init__(self, template, reference):
        super().__init__()
        self.template = template
        self.reference = reference

    def forward(self, params):
        if len(params.shape) > 1:
            params = params.squeeze(0)
        angle, tx, ty = params[0], params[1], params[2]
        matrix = get_affine_matrix(
            center=[self.template.shape[1] // 2, self.template.shape[0] // 2],
            angle=angle,
            translate=[tx, ty]
        )
        transformed = affine_transform(self.template, matrix)
        overlap = transformed * self.reference
        output = torch.sum(overlap) ** 2
        return output.unsqueeze(0).unsqueeze(1)  # Shape [batch_size, 1]

def optimize_coordinates(template, reference, target_coords, options, max_iter=100, tol=1e-10):
    obj_func = CoordinateObjective(template, target_coords)
    const_func = AlignmentConstraint(template, reference)
    f = ObjectiveOrConstraint(obj_func, dim=3)
    g = ObjectiveOrConstraint(const_func, dim_out=1)
    gI = []  # Inequality constraints
    gE = [g]  # Equality constraints
    x0 = np.array([0.1, 0.1, 0.1])  # Initial guess
    # Setup problem with SQPGS
    problem = SQPGS(f, gI, gE, x0=x0, tol=tol, max_iter=max_iter, verbose=True, options=options)
    x = problem.solve()

    return x, problem.x_hist