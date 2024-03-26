import numpy as np

def generate_discontinuous_distribution(probabilities_zero, x_thresholds, scales, num_samples, rng):
    """
    Generate events for a toy discontinuous distribution, composed of a delta peak at zero with an exponential tail.

    Parameters:
    probabilities_zero (numpy.ndarray): The probabilities that a sample is zero (Dirac Delta) for each feature.
    x_thresholds (numpy.ndarray): Threshold values for the exponential component for each feature.
    scales (numpy.ndarray): The scales of the exponential component for each feature.
    num_samples (int): The total number of samples to generate.
    rng (numpy.random.Generator): Random number generator.

    Returns:
    numpy.ndarray: A 2D array of shape (num_samples, n_features) containing the generated event samples.

    This function generates a sequence of events by sampling `num_samples` exponentially distributed values
    for each feature and then setting values to zero based on the specified `probabilities_zero`.

    Example usage:
    >>> num_samples = 1000
    >>> n_features = 3
    >>> probabilities_zero = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]])  # Two sets of probabilities
    >>> x_thresholds = np.array([[3.0, 2.0, 1.0], [2.0, 1.0, 0.5]])  # Two sets of thresholds
    >>> scales = np.array([[1.0, 0.5, 0.2], [0.8, 0.4, 0.1]])  # Two sets of scales
    >>> rng = np.random.default_rng(42)
    >>> event_samples = generate_discontinuous_distribution(probabilities_zero, x_thresholds, scales, num_samples, rng)
    """

    # Ensure the arguments are numpy arrays
    probabilities_zero = np.asarray(probabilities_zero)
    x_thresholds = np.asarray(x_thresholds)
    scales = np.asarray(scales)

    # Check if the shapes of input arrays are compatible
    if probabilities_zero.shape != x_thresholds.shape or x_thresholds.shape != scales.shape:
        raise ValueError("Input arrays should have the same shape for num_samples and n_features")

    num_samples, n_features = probabilities_zero.shape

    # Create an output array to store the samples
    samples = np.zeros((num_samples, n_features))

    for feature in range(n_features):
        # Generate `num_samples` exponentially distributed values for the current feature
        expon = rng.exponential(scale=scales[:, feature], size=num_samples) + x_thresholds[:, feature]
        # Set values to zero based on the specified `probabilities_zero` for the current feature
        zero_mask = rng.uniform(size=num_samples) < probabilities_zero[:, feature]
        expon[zero_mask] = 0
        # Assign the generated values to the samples for the current feature
        samples[:, feature] = expon

    return samples