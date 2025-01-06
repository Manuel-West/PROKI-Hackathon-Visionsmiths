import torch as torch
import numpy as np

from torch.autograd import Variable as V
from typing import Tuple, Union, Optional, List
from scipy.optimize import minimize
from PIL import Image

class AffineImageTransformer:
    """
    A class for performing affine transformations on images using PyTorch.
    Provides functionality for coordinate grid creation, bilinear interpolation,
    and affine matrix generation.
    """

    @staticmethod
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
        coor = AffineImageTransformer.create_coordinate_grid(xsize,ysize)
        xyprime = torch.tensordot(coor, torch.t(transform_matrix), dims=1)
        return AffineImageTransformer.bilinear_interpolate(input_img, xyprime)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


class OptimizationFunctions:
    def __init__(self, template: torch.Tensor, reference: torch.Tensor, target_coords: Tuple[float, float]):
        self.template = template
        self.reference = reference
        self.target_x, self.target_y = target_coords
        self.template_center = [template.shape[1] // 2, template.shape[0] // 2]

    def _compute_objective_with_grad(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute both objective value and its gradient"""
        params_torch = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        angle, tx, ty = params_torch[0], params_torch[1], params_torch[2]

        matrix = AffineImageTransformer.get_affine_matrix(
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

        # Compute gradient
        output.backward()
        grad = params_torch.grad.numpy()

        return output.item(), grad

    def objective(self, params: np.ndarray) -> float:
        """Objective function for scipy.optimize.minimize"""
        obj_value, _ = self._compute_objective_with_grad(params)
        return obj_value

    def objective_gradient(self, params: np.ndarray) -> np.ndarray:
        """Gradient of the objective function"""
        _, grad = self._compute_objective_with_grad(params)
        return grad

    def _compute_constraint_with_grad(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute both constraint value and its gradient"""
        params_torch = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        angle, tx, ty = params_torch[0], params_torch[1], params_torch[2]

        matrix = AffineImageTransformer.get_affine_matrix(
            center=self.template_center,
            angle=angle,
            translate=[tx, ty]
        )

        transformed = AffineImageTransformer.affine_transform(self.template, matrix)
        overlap = transformed * self.reference
        output = torch.sum(overlap)

        # Compute gradient
        output.backward()
        grad = params_torch.grad.numpy()

        return output.item(), grad

    def constraint(self, params: np.ndarray) -> float:
        """Constraint function for scipy.optimize.minimize"""
        cons_value, _ = self._compute_constraint_with_grad(params)
        return cons_value

    def constraint_gradient(self, params: np.ndarray) -> np.ndarray:
        """Gradient of the constraint function"""
        _, grad = self._compute_constraint_with_grad(params)
        return grad

def optimize_coordinates(template: torch.Tensor,
                         reference: torch.Tensor,
                         target_coords: Tuple[float, float],
                         max_iter: int = 100,
                         tol: float = 1e-10,
                         show: bool = False,
                         output_path: Optional[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Optimize the transformation parameters using scipy.optimize.minimize with SLSQP method.
    """
    opt_funcs = OptimizationFunctions(template, reference, target_coords)

    # Initial guess
    x0 = np.array([0.1, 0.1, 0.1])

    # Define constraint dictionary
    constraint = {
        'type': 'eq',
        'fun': opt_funcs.constraint,
        'jac': opt_funcs.constraint_gradient
    }
    # Calculate bounds for optimization
    half_width = reference.shape[1] / 2
    half_height = reference.shape[0] / 2

    # bounds for [angle, tx, ty]
    # angle can be unbounded (-inf, inf), tx and ty are bounded by image dimensions
    bounds = [
        (-np.inf, np.inf),  # angle: no bounds
        (-half_width, half_width),  # tx: bounded by half width
        (-half_height, half_height)  # ty: bounded by half height
    ]

    # Run optimization
    result = minimize(
        fun=opt_funcs.objective,
        x0=x0,
        method='SLSQP',
        jac=opt_funcs.objective_gradient,
        bounds=bounds,
        constraints=[constraint],
        options={
            'maxiter': max_iter,
            'ftol': tol,
            'disp': True
        }
    )

    if show:
        result_image = apply_solution_overlay(
            template=template,
            reference=reference,
            solution_vector=result.x,
            output_path=output_path  # Optional
        )
    return result.x, result.x_hist if hasattr(result, 'x_hist') else [result.x]


def apply_solution_overlay(template: torch.Tensor,
                           reference: torch.Tensor,
                           solution_vector: np.ndarray,
                           output_path: str = None) -> np.ndarray:
    """
    Applies the solution vector to the template and creates a visualization
    where the transformed template is shown in red over the reference image.

    Args:
        template: Template image tensor
        reference: Reference image tensor
        solution_vector: Solution vector [angle, tx, ty] from optimization
        output_path: Optional path to save the visualization. If None, only returns the array

    Returns:
        numpy array of RGB image with red overlay (height, width, 3)
    """
    # Get image dimensions from reference
    height, width = reference.shape
    center = [width // 2, height // 2]

    # Convert solution vector to appropriate types
    angle = torch.tensor(float(solution_vector[0]), dtype=torch.float32)
    translate = [float(solution_vector[1]), float(solution_vector[2])]

    # Apply transformation to template
    matrix = AffineImageTransformer.get_affine_matrix(
        center=center,
        angle=angle,
        translate=translate
    )
    transformed_template = AffineImageTransformer.affine_transform(template, matrix)

    # Create RGB visualization
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Set reference as grayscale background
    reference_np = reference.detach().cpu().numpy()
    rgb_image[..., 0] = 255-reference_np
    rgb_image[..., 1] = (1-reference_np) * 255
    rgb_image[..., 2] = (1-reference_np) * 255

    # Add red overlay for transformed template
    transformed_np = transformed_template.detach().cpu().numpy()
    rgb_image[transformed_np > 0.5] = [255, 0, 0]  # Red color for template

    # Save if output path is provided
    if output_path:
        Image.fromarray(rgb_image).save(output_path)

    return rgb_image
