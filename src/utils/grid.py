import numpy as np


def cartesian_product(arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def get_grid_int(steps, data_format='channels_last'):
    assert data_format in ['channels_first', 'channels_last']
    shape_indices = [np.arange(x) for x in steps]
    xyz_indices = cartesian_product(shape_indices).astype(np.int32)
    if data_format == 'channels_first':
        return xyz_indices.T
    return xyz_indices


def get_grid(steps, data_format='channels_last'):
    assert data_format in ['channels_first', 'channels_last']
    shape_indices = [np.linspace(0, 1.0, x) for x in steps]
    xyz_indices = cartesian_product(shape_indices)
    if data_format == 'channels_first':
        return xyz_indices.T
    return xyz_indices


def get_batched_grid(steps, batch_size, data_format='channels_last'):
    """
    Returns a batch of grids

    :param steps: list of steps for each dimension
    :param batch_size: int
    :param data_format: str, 'channels_first' or 'channels_last'
    :return: an array of [batch_size, len(steps), np.prod(steps)] grid points
    """
    assert data_format in ['channels_first', 'channels_last']
    grid = get_grid(steps)
    grid_batch = np.tile(np.array(grid).T, batch_size).T
    grid_batch = np.reshape(grid_batch, (batch_size, np.prod(steps), len(steps)))
    if data_format == 'channels_first':
        grid_batch = np.transpose(grid_batch, (0, 2, 1))

    return grid_batch


def parse_grid_steps(grid_steps, n):
    if grid_steps != 'auto':
        return [int(x) for x in grid_steps.split(',')]
    else:
        n2 = np.sqrt(n / 1.618)
        return [int(x) for x in [n2, 1.618 * n2, 1]]


def grid_borders_mask(grid_steps):
    t = np.zeros(grid_steps)
    t[0, :] = 255.0
    t[:, 0] = 255.0
    t[-1, :] = 255.0
    t[:, -1] = 255.0

    return t == 255.0
