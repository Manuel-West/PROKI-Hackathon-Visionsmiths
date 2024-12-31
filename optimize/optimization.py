import torch as t
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


def affine_transform(input_img: t.Tensor, transform_matrix: t.Tensor) -> t.Tensor:
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
    xyprime = t.tensordot(coor, t.t(transform_matrix), dims=1)
    return bilinear_interpolate(input_img, xyprime)


def create_coordinate_grid(xsize: int, ysize: int) -> t.Tensor:
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
    return V(t.tensor(coorMatrix),True)


def bilinear_interpolate(img: t.Tensor, coords: t.Tensor) -> t.Tensor:
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
    x0 = t.clamp(t.floor(x_coords).long(), 1, xsize - 2)
    y0 = t.clamp(t.floor(y_coords).long(), 1, ysize - 2)
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
    zeros = t.zeros(img.shape)
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
    matrix = V(t.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]]).double(), True)
    losses = []
    rate = 0.01

    for i in range(iteration):
        transformed = affine_transform(template, matrix)
        error = reference - transformed
        loss = t.sum(error[randxy(template, rate) == 1] ** 2)
        loss.backward(retain_graph=True)
        losses.append(np.sum((transformed.detach().numpy() - reference.detach().numpy()) ** 2))

        with t.no_grad():
            matrix[:2, :] -= learning_rate * matrix.grad[:2, :]
            matrix.grad.zero_()

    plt.plot(np.arange(iteration), losses)
    imshow(template.data.numpy(), transformed.data.numpy(), reference.data.numpy())
    print("last loss:", losses[-1])
    print("last gradient:", matrix.grad[:2, :])

    return matrix, losses


def get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float]) -> t.Tensor:
    rot = math.radians(angle)
    cx, cy = center
    tx, ty = translate

    angle_cos = math.cos(rot)
    angle_sin = math.sin(rot)

    matrix = [
        [angle_cos, -angle_sin, angle_cos*(-tx-cx)-angle_sin*(-ty-cy)+cx],
        [angle_sin, angle_cos, angle_sin*(-tx-cx)+angle_cos*(-ty-cy)+cy],
        [0,0,1]]


    return t.tensor(matrix)

if __name__ == "__main__":
    foreground_img = plt.imread('hand.jpg')
    #foreground_img = np.zeros((100, 100))
    #foreground_img[25:75, 25:75] = 1
    #foreground_img[33:37, 33:37] = 0
    foreground_img = V(t.Tensor(foreground_img).double(), True)
    xsize, ysize = foreground_img.shape
    center1 = xsize /2
    center2 = ysize /2
    transformMatrix2 = V(get_inverse_affine_matrix(center=[35,35],angle=45, translate=[-15,-15]).double(), True)
    background_img = affine_transform(foreground_img, transformMatrix2)
    gradient_descent(foreground_img, background_img, iteration=500)