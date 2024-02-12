import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    ##-your-code-starts-here-##
    A = np.empty((0, 12))
    for x_w, x_i in zip(x_world, x_im):
        a_1 = np.reshape(np.hstack((np.zeros(4), x_w, -x_i[1]*x_w)), (-1,12))
        a_2 = np.reshape(np.hstack((x_w, np.zeros(4), -x_i[0]*x_w)), (-1,12))
        A = np.vstack((A, a_1, a_2))
    
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    solution_A = np.matmul(A.T, A)
    eivals, eivects = np.linalg.eig(solution_A)
    print(eivects)
    ev = eivects[:, np.argmin(eivals)]
    print(ev)
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    # P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P
