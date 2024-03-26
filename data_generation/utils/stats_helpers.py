import numpy as np

def find_closest_PSD(matrix):
    """
    Compute the closest (according to some subjective definition) Positive Semidefinite (PSD) matrix to the given
    input matrix while ensuring the diagonal elements of the resulting matrix are equal to 1.

    Parameters:
    matrix (numpy.ndarray): The input matrix for which to find the closest PSD matrix.

    Returns:
    numpy.ndarray: A matrix that is the closest PSD approximation of the input matrix, with diagonal elements
    set to 1.

    This function computes the eigenvalues and eigenvectors of the input matrix and replaces any negative eigenvalues
    with the mean of the positive eigenvalues, ensuring that the resulting matrix is PSD (even positive definite).
    Additionally, it sets the diagonal elements of the resulting matrix to 1, which is a requirement for correlation matrices.

    Example:
    >>> import numpy as np
    >>> input_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    >>> psd_matrix = find_closest_PSD(input_matrix)
    >>> print(psd_matrix)
    array([[1.        , 0.5       , 0.3       ],
           [0.5       , 1.        , 0.2       ],
           [0.3       , 0.2       , 1.        ]])

    Note:
    The input matrix should be symmetric, and the output matrix is also guaranteed to be symmetric.

    """
    eigval, eigvec = np.linalg.eig(matrix)
    number_of_negative_eigvals = len(eigval[eigval < 0])
    print('Have to replace ', number_of_negative_eigvals, ' negative eigenvalues with mean...')
    eigval[eigval < 0] = np.mean(eigval[eigval > 0])
    # Ensure the diagonal elements are 1
    matrix = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
    np.fill_diagonal(matrix, 1.0)
    return matrix