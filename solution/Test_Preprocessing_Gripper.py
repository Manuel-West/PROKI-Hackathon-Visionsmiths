import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import os

# Assuming the functions are defined in a module named `image_processing`
import Preprocessing_Gripper as pg

class TestImageProcessing(unittest.TestCase):

    @patch('cv2.imread')
    @patch('os.path.exists')
    @patch('cv2.cvtColor')
    @patch('cv2.threshold')
    def test_get_binary(self, mock_threshold, mock_cvtColor, mock_exists, mock_imread):
        # Simulate successful read and conversion
        mock_exists.return_value = True
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))

        binary_image = pg.get_binary_image('valid_path')
        self.assertIsNotNone(binary_image)
        self.assertEqual(binary_image.shape, (100, 100))

    def test_find_center(self):
        n = 5
        # Simulate valid binary image
        binary_image = np.zeros((n, n), dtype=np.uint8)
        binary_image[2, 2] = 255  # Create a white square in the center

        cX, cY = pg.find_center(binary_image)
        self.assertEqual(2, cX)
        self.assertEqual(2, cY)

        n = 4
        # Simulate valid binary image
        binary_image = np.zeros((n, n), dtype=np.uint8)
        binary_image[1:2, 1:2] = 255  # Create a white square in the center

        cX, cY  = pg.find_center(binary_image)
        self.assertEqual(int(n/2)-1, cX) # -1 becourse array starts with 0
        self.assertEqual(int(n/2)-1, cY) # -1 becourse array starts with 0

if __name__ == '__main__':
    unittest.main()