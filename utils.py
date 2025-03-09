import numpy as np
from typing import Tuple, Union, List, Dict

def triangular_fuzzy_saaty(scale_value: int) -> Tuple[float, float, float]:
    """Returns the corresponding Triangular Fuzzy Number (TFN) for a given Saaty scale value."""
    if scale_value < 1 or scale_value > 9:
        raise ValueError("Scale value must be between 1 and 9")

    if scale_value == 1:
        return (1.0, 1.0, 2.0)
    elif scale_value == 9:
        return (8.0, 9.0, 9.0)
    else:
        return (scale_value - 1, scale_value, scale_value + 1)

def inverse_tfn(tfn: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Calculate the inverse of a triangular fuzzy number."""
    l, m, u = tfn
    epsilon = 1e-10  # Small constant to prevent division by zero
    return (1.0 / (u + epsilon), 1.0 / (m + epsilon), 1.0 / (l + epsilon))

def calculate_fuzzy_synthetic_extent(matrix: np.ndarray) -> np.ndarray:
    """Calculates the fuzzy synthetic extent values from the pairwise comparison matrix."""
    epsilon = 1e-10  # Small constant to prevent division by zero
    row_sums = np.sum(matrix, axis=1)  # Shape: (n, 3) for TFNs
    total_sum = np.sum(row_sums, axis=0) + epsilon  # Shape: (3,) for TFN
    return row_sums / total_sum  # Broadcasting will handle the division

def adjust_weights_matrix(matrix: np.ndarray, strategic_avg: np.ndarray, level: str) -> np.ndarray:
    """Adjusts the comparison matrix based on respondent level and strategic average."""
    if level == "Strategic":
        return matrix

    factors = {
        "Tactical": (1.33, 0.67),  # +33% or -33%
        "Operational": (1.66, 0.34)  # +66% or -66%
    }
    increase_factor, decrease_factor = factors[level]

    adjusted_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(3):  # For each TFN component (l,m,u)
                if matrix[i,j,k] < strategic_avg[i,j,k]:
                    adjusted_matrix[i,j,k] = matrix[i,j,k] * increase_factor
                else:
                    adjusted_matrix[i,j,k] = matrix[i,j,k] * decrease_factor

    return np.clip(adjusted_matrix, 1.0, 9.0)

def validate_matrix(matrix: np.ndarray) -> bool:
    """Validates the consistency of a pairwise comparison matrix."""
    if not np.all(np.isfinite(matrix)):
        return False
    if not np.allclose(np.diagonal(matrix[:,:,1]), 1.0):  # Check middle values on diagonal
        return False
    return True

def defuzzify_tfn(l: float, m: float, u: float) -> float:
    """Defuzzify a triangular fuzzy number using the centroid method."""
    return (l + m + u) / 3.0

def normalize_crisp_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize an array of crisp weights to sum to 1."""
    return weights / np.sum(weights)

def calculate_alternative_priorities(alternatives_matrix: Dict[str, Dict[str, Tuple[np.ndarray, str]]], 
                                  criteria_weights: np.ndarray,
                                  adjust_levels: bool = True) -> Tuple[np.ndarray, List[int]]:
    """Calculate final alternative priorities using criteria weights."""
    if not alternatives_matrix:
        return np.array([]), []

    # Get dimensions from first criterion's first respondent's matrix
    first_criterion = next(iter(alternatives_matrix.keys()))
    first_respondent = next(iter(alternatives_matrix[first_criterion].values()))
    num_alternatives = first_respondent[0].shape[0]
    final_weights = np.zeros(num_alternatives)

    # Process each criterion
    for criterion_idx, (criterion, respondents_data) in enumerate(alternatives_matrix.items()):
        # Skip if no respondents evaluated this criterion
        if not respondents_data:
            continue

        # Calculate strategic average if using level adjustment
        if adjust_levels:
            strategic_matrices = [
                matrix for matrix, level in respondents_data.values()
                if level == "Strategic"
            ]
            if strategic_matrices:
                strategic_avg = np.mean(strategic_matrices, axis=0)
            else:
                strategic_avg = np.ones_like(next(iter(respondents_data.values()))[0])

        # Process each respondent's evaluation
        alt_weights_list = []
        for matrix, level in respondents_data.values():
            # Apply level adjustment if needed
            if adjust_levels:
                matrix = adjust_weights_matrix(matrix, strategic_avg, level)

            # Calculate fuzzy synthetic extent values
            fuzzy_weights = calculate_fuzzy_synthetic_extent(matrix)

            # Defuzzify weights
            crisp_weights = np.array([
                defuzzify_tfn(l, m, u) 
                for l, m, u in fuzzy_weights
            ])

            # Normalize weights
            normalized_weights = normalize_crisp_weights(crisp_weights)
            alt_weights_list.append(normalized_weights)

        # Only proceed if we have weights for this criterion
        if alt_weights_list:
            # Average weights across respondents for this criterion
            criterion_weights = np.mean(alt_weights_list, axis=0)

            # Add to final weights
            final_weights += criterion_weights * criteria_weights[criterion_idx]

    # Normalize final weights
    final_weights = normalize_crisp_weights(final_weights)

    # Sort alternatives by final weight
    sorted_indices = np.argsort(final_weights)[::-1]
    sorted_weights = final_weights[sorted_indices]

    return sorted_weights, sorted_indices
