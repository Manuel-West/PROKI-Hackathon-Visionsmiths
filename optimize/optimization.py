import torch as torch
import numpy as np
import matplotlib.pyplot as plt

import math
from torch.autograd import Variable as V
from typing import Tuple, Union, Optional, List
from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.max_linear import MaxOfLinear
from ncopt.functions.rosenbrock import NonsmoothRosenbrock
from ncopt.sqpgs import SQPGS

import random


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

def randxy(img,rate):
    """
     Creates a random binary mask with the specified shape and sampling rate.

     Args:
         shape: Tuple of (height, width) or torch.Size
         rate: Fraction of positions to set to 1 (between 0 and 1)
     Returns:
         Binary mask tensor of the specified shape with rate% of 1s
     """
    zeros = torch.zeros(img.shape)
    xsize, ysize = img.shape
    i = 0
    flatten_indexes = [i for i in range(xsize*ysize)]
    choices = random.sample(flatten_indexes, int(rate*len(flatten_indexes)))
    for i in range(len(choices)):
        zeros[choices[i]//ysize][choices[i]%xsize] = 1
    return zeros

def imshow(template, transformed, reference):
    plt.figure(2)
    plt.subplot(1, 3, 1)
    plt.imshow(template)
    plt.title("Template")

    plt.subplot(1, 3, 2)
    plt.imshow(transformed)
    plt.title("Transformed")

    plt.subplot(1, 3, 3)
    plt.imshow(reference)
    plt.title("Reference")
    plt.show()

def gradient_descent(template, reference, iteration=1000, learning_rate=1e-9):
    """
    Performs gradient descent to find optimal affine transformation matrix

    Args:
        template: Source image tensor to transform
        reference: Target image tensor to match
        iteration: Number of optimization iterations
        learning_rate: Step size for gradient updates
    """
    matrix = V(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]]).double(), True)
    losses = []
    rate = 0.01

    for i in range(iteration):
        transformed = affine_transform(template, matrix)
        error = reference - transformed
        loss = torch.sum(error[randxy(template, rate) == 1] ** 2)
        loss.backward(retain_graph=True)
        losses.append(np.sum((transformed.detach().numpy() - reference.detach().numpy()) ** 2))

        with torch.no_grad():
            matrix[:2, :] -= learning_rate * matrix.grad[:2, :]
            matrix.grad.zero_()

    plt.plot(np.arange(iteration), losses)
    imshow(template.data.numpy(), transformed.data.numpy(), reference.data.numpy())
    print("last loss:", losses[-1])
    print("last gradient:", matrix.grad[:2, :])

    return matrix, losses


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

def optimize_coordinates(template, reference, target_coords, max_iter=100, tol=1e-10):
    obj_func = CoordinateObjective(template, target_coords)
    const_func = AlignmentConstraint(template, reference)
    f = ObjectiveOrConstraint(obj_func, dim=3)
    g = ObjectiveOrConstraint(const_func, dim_out=1)
    gI = [g]  # Inequality constraints
    gE = []  # Equality constraints
    x0 = np.array([0.0, 0.0, 0.0])  # Initial guess
    # Setup problem with SQPGS
    problem = SQPGS(f, gI, gE, x0=x0, tol=tol, max_iter=max_iter, verbose=True)
    x = problem.solve()
    # Final transformation using get_affine_matrix
    matrix = get_affine_matrix(
        center=[template.shape[1] // 2, template.shape[0] // 2],
        angle=x[0],
        translate=[x[1], x[2]]
    )
    return matrix, problem.x_hist

def plot_optimization_trajectory(template, reference, target_coords):
    matrix, history = optimize_coordinates(template, reference, target_coords)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot template and reference centers
    template_center = [template.shape[1] // 2, template.shape[0] // 2]
    ax.scatter(template_center[0], template_center[1], marker="o", s=100,
               c="blue", label="Template Center")
    ax.scatter(target_coords[0], target_coords[1], marker="*", s=200,
               c="gold", label="Target")

    # Plot optimization trajectory (tx, ty components)
    ax.plot(history[:, 1], history[:, 2], c="silver", lw=1, ls="--",
            alpha=0.5, zorder=2, label="Translation Path")

    ax.set_xlabel("X Translation")
    ax.set_ylabel("Y Translation")
    ax.legend()
    fig.tight_layout()

    return matrix, fig

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


def create_template(size=100, square_size=20):
    """Create a simple square template"""
    img = np.zeros((size, size))
    start = (size - square_size) // 2
    end = start + square_size
    img[start:end, start:end] = 1
    return img


def create_reference_with_obstacles(size=100, border_width=5, n_obstacles=3, obstacle_size=15):
    """Create a reference image with border and internal obstacles"""
    img = np.zeros((size, size))

    # Add border
    img[0:border_width, :] = 1  # Top border
    img[-border_width:, :] = 1  # Bottom border
    img[:, 0:border_width] = 1  # Left border
    img[:, -border_width:] = 1  # Right border

    # Add random internal obstacles
    np.random.seed(42)  # For reproducibility
    for _ in range(n_obstacles):
        x = np.random.randint(border_width + obstacle_size, size - border_width - obstacle_size)
        y = np.random.randint(border_width + obstacle_size, size - border_width - obstacle_size)
        img[x:x + obstacle_size, y:y + obstacle_size] = 1

    return img


def visualize_results(template, reference, transformed=None, title="Optimization Results"):
    """Visualize the images"""
    n_plots = 3 if transformed is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    axes[0].imshow(template.detach().numpy(), cmap='gray')
    axes[0].set_title('Template')
    axes[0].axis('off')

    axes[1].imshow(reference.detach().numpy(), cmap='gray')
    axes[1].set_title('Reference with Obstacles')
    axes[1].axis('off')

    if transformed is not None:
        axes[2].imshow(reference.detach().numpy(), cmap='gray')
        transformed_np = transformed.detach().numpy()
        # Overlay transformed template in a different color
        overlay = np.zeros((*transformed_np.shape, 4))  # RGBA
        overlay[..., 0] = transformed_np  # Red channel
        overlay[..., 3] = transformed_np * 0.7  # Alpha channel
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Result')
        axes[2].axis('off')

    plt.suptitle(title)
    plt.show()


def main():
    # Create images
    size = 100
    template_img = create_template(size, square_size=20)
    reference_img = create_reference_with_obstacles(size, border_width=5, n_obstacles=3)

    # Convert to torch tensors
    template = V(torch.tensor(template_img).double(), requires_grad=True)
    reference = V(torch.tensor(reference_img).double(), requires_grad=True)

    # Visualize initial setup
    visualize_results(template, reference, title="Initial Setup")

    # Target coordinates (center of the image)
    target_coords = [size // 2, size // 2]

    # Run optimization
    print("Running optimization...")
    matrix, history = optimize_coordinates(
        template=template,
        reference=reference,
        target_coords=target_coords,
        max_iter=200,
        tol=1e-1
    )

    # Transform template with final transformation
    transformed = affine_transform(template, matrix)

    # Visualize results
    visualize_results(template, reference, transformed,
                      "Optimization Results\nRed overlay shows transformed template")

    # Plot optimization trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(history[:, 1], history[:, 2], 'b-', label='Optimization Path')
    plt.scatter(target_coords[0], target_coords[1], c='red', marker='*', s=200, label='Target Center')
    plt.title('Optimization Trajectory')
    plt.xlabel('X Translation')
    plt.ylabel('Y Translation')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final metrics
    collision = torch.sum(transformed * reference)
    distance_to_center = np.sqrt((history[-1, 1]) ** 2 + (history[-1, 2]) ** 2)
    print(f"\nResults:")
    print(f"Final collision metric: {collision.item():.4f}")
    print(f"Final distance to center: {distance_to_center:.4f}")
    print(f"Final angle: {history[-1, 0]:.2f} degrees")


if __name__ == "__main__":
    main()