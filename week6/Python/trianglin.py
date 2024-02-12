import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    x1_x = np.array([[     0, -x1[2],  x1[1]],
                     [ x1[2],      0, -x1[0]],
                     [-x1[1],  x1[0],     0]])
    x2_x = np.array([[     0, -x2[2],  x2[1]],
                     [ x2[2],      0, -x2[0]],
                     [-x2[1],  x2[0],     0]])
    x1P1 = x1_x @ P1
    x2P2 = x2_x @ P2
    A = np.vstack((x1P1[:2], x2P2[:2]))
    eivals, eivects = np.linalg.eig(A.T@A)
    X = eivects[:, np.argmin(eivals)]
    ##-your-code-ends-here-##
    # X = np.array([0, 0, 0, 1])  # remove me
    
    return X
