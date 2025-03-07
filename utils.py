import numpy as np
from typing import Tuple, Union, List

def triangular_fuzzy_saaty(scale_value: int) -> Tuple[float, float, float]:
    """
    Returns the corresponding Triangular Fuzzy Number (TFN) for a given Saaty scale value.

    Args:
        scale_value (int): Saaty scale value (1-9)
    Returns:
        Tuple[float, float, float]: TFN representation (l, m, u)
    """
    if scale_value < 1 or scale_value > 9:
        raise ValueError("Scale value must be between 1 and 9")

    if scale_value == 1:
        return (1.0, 1.0, 2.0)
    elif scale_value == 9:
        return (8.0, 9.0, 9.0)
    else:
        return (scale_value - 1, scale_value, scale_value + 1)

def inverse_tfn(tfn: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Calculate the inverse of a triangular fuzzy number.
    For TFN (l,m,u), inverse is (1/u, 1/m, 1/l)

    Args:
        tfn (Tuple[float, float, float]): Input TFN (l, m, u)
    Returns:
        Tuple[float, float, float]: Inverse TFN (1/u, 1/m, 1/l)
    """
    l, m, u = tfn
    epsilon = 1e-10  # Small constant to prevent division by zero
    return (
        1.0 / (u + epsilon),
        1.0 / (m + epsilon),
        1.0 / (l + epsilon)
    )

def calculate_fuzzy_synthetic_extent(matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the fuzzy synthetic extent values from the pairwise comparison matrix.

    Args:
        matrix (np.ndarray): 3D matrix of TFNs
    Returns:
        np.ndarray: Fuzzy synthetic extent values
    """
    epsilon = 1e-10
    # Sum rows
    row_sums = np.sum(matrix, axis=1)

    # Sum all elements
    total_sum = np.sum(row_sums, axis=0) + epsilon

    # Normalize
    return row_sums / total_sum

def adjust_weights_matrix(matrix: np.ndarray, strategic_avg: np.ndarray, level: str) -> np.ndarray:
    """
    Adjusts the comparison matrix based on respondent level and strategic average.

    Args:
        matrix (np.ndarray): Original comparison matrix
        strategic_avg (np.ndarray): Strategic level average matrix
        level (str): Respondent level
    Returns:
        np.ndarray: Adjusted matrix
    """
    if level == "Strategic":
        return matrix

    # Define adjustment factors
    factors = {
        "Tactical": (1.33, 0.67),  # +33% or -33%
        "Operational": (1.66, 0.34)  # +66% or -66%
    }
    increase_factor, decrease_factor = factors[level]

    # Compare and adjust each element in the matrix
    adjusted_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(3):  # For each TFN component (l,m,u)
                if matrix[i,j,k] < strategic_avg[i,j,k]:
                    adjusted_matrix[i,j,k] = matrix[i,j,k] * increase_factor
                else:
                    adjusted_matrix[i,j,k] = matrix[i,j,k] * decrease_factor

    # Ensure values stay within bounds (1-9)
    return np.clip(adjusted_matrix, 1.0, 9.0)

def validate_matrix(matrix: np.ndarray) -> bool:
    """
    Validates the consistency of a pairwise comparison matrix.

    Args:
        matrix (np.ndarray): Pairwise comparison matrix
    Returns:
        bool: True if matrix is valid
    """
    if not np.all(np.isfinite(matrix)):
        return False

    # Check diagonal elements
    if not np.allclose(np.diagonal(matrix), 1.0):
        return False

    return True

def defuzzify_tfn(l: float, m: float, u: float) -> float:
    """
    Defuzzify a triangular fuzzy number using the centroid method.

    Args:
        l (float): Lower bound
        m (float): Middle value
        u (float): Upper bound
    Returns:
        float: Defuzzified (crisp) value
    """
    return (l + m + u) / 3.0

def normalize_crisp_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize an array of crisp weights to sum to 1.

    Args:
        weights (np.ndarray): Array of weights
    Returns:
        np.ndarray: Normalized weights
    """
    return weights / np.sum(weights)