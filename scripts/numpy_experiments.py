import numpy as np

np.set_printoptions(linewidth=125)
# Setup fake input for testing
corners = np.array([
    [0, 0],
    [5, 5],
    [5, 6],
    [5, 7],
    [6, 6],
    [6, 7],
    [7, 6],
    [7, 7],
])
corners = np.expand_dims(corners, axis=1)
vectors = np.swapaxes(corners, 0, 1) - corners
print(vectors.shape)

# Compute |A| * |B|
mag = np.sqrt(np.sum(np.square(vectors), axis=2))
print(mag)
print(mag[:, 0].reshape(9, 1) * mag[:, 0])
mag = mag * mag.T

# Compute A*B
dot = np.outer(vectors[:, :, 0], vectors[:, :, 0]) + np.outer(vectors[:, :, 1], vectors[:, :, 1])
dot = dot.reshape((9, 9, 9, 9))

# Compute arccos(Dot Product / Magnitude)
result = np.arccos(dot / mag)

# Replace Nan with 0
result = np.nan_to_num(result)

# Convert to Degrees
result = np.rad2deg(result)

'''
There should be 9 places where the angle between the two vectors
is 0, as this is the self-angle. The vector compared against itself
'''
print((result == 0).sum())

# Spot-check results. Select the two vectors using: [r1, c1, r2, c2]
print(result[0, 0, :, :])
