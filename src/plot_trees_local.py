import laspy
import numpy as np
import numba

import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree
from mpl_toolkits.mplot3d import Axes3D

CLUSTER_SIZE = 200
MIN_DISTANCE = 4
HIST = False
PLOT3D = False
PLOT2D = True

chunk_size = 1_000_000
fname = 'out0010_pred.las'#"C_64EZ2_pred.las"

def get_ground_level_around_local_max(X, Y, Z, loc_max_X_Y):
    """
    Given X, Y, Z of points and single peak location
    loc_max_X_Y, calculates the ground level around 
    that local maximum
    """
    R_T = 0.5
    mask = np.power(loc_max_X_Y[0] - X, 2) + np.power(loc_max_X_Y[1] - Y, 2) < R_T**2
    ground_level = np.min(Z[mask])
    return ground_level

def locally_extreme_points(coords, data, neighbourhood, lookfor = 'max', p_norm = 2.):
    '''
    Find local maxima of points in a pointcloud.  Ties result in both points passing through the filter.

    Not to be used for high-dimensional data.  It will be slow.

    coords: A shape (n_points, n_dims) array of point locations
    data: A shape (n_points, ) vector of point values
    neighbourhood: The (scalar) size of the neighbourhood in which to search.
    lookfor: Either 'max', or 'min', depending on whether you want local maxima or minima
    p_norm: The p-norm to use for measuring distance (e.g. 1=Manhattan, 2=Euclidian)

    returns
        filtered_coords: The coordinates of locally extreme points
        filtered_data: The values of these points
    '''
    assert coords.shape[0] == data.shape[0], 'You must have one coordinate per data point'
    extreme_fcn = {'min': np.min, 'max': np.max}[lookfor]
    kdtree = KDTree(coords)
    neighbours = kdtree.query_ball_tree(kdtree, r=neighbourhood, p = p_norm)
    i_am_extreme = [data[i]==extreme_fcn(data[n]) and len(neighbours[i]) > CLUSTER_SIZE for i, n in enumerate(neighbours)]
    extrema, = np.nonzero(i_am_extreme)  # This line just saves time on indexing
    return coords[extrema], data[extrema]

@numba.njit
def get_overlap_masks(m1, m2):
    for i in range(chunk_size):
        if m1[i] == 1 and m2[i] == 1:
            m1[i] = 1
        else:
            m1[i] = 0
    return m1        

def get_hist_data(X, Y, Z, maxima, maxima_xy):
    """
    Creates an array of tree heights
    """
    tree_heights = np.zeros((len(maxima)))

    for i, maximum in enumerate(maxima):
        ground_level = get_ground_level_around_local_max(X, Y, Z, maxima_xy[i])
        tree_heights[i] = ground_level + maximum
    return tree_heights

def main():
    heights = []

    with laspy.open(fname) as file:
        i = 0
        for points in file.chunk_iterator(chunk_size):
            mask = np.array(points.classification) > 0
            mask2 = np.array(points.number_of_returns) > 1#np.array(points.z) > 0.3
            mask = get_overlap_masks(mask, mask2)   
            x = np.array(points.x)
            x_mask = x[mask]
            y = np.array(points.y)
            y_mask = y[mask]
            points_array = np.array((x, y)).T
            z = np.array(points.z)
            z_mask = z[mask]

            # Get local max
            local_max_x_y_2, local_max_z_2 = locally_extreme_points(points_array[mask], z[mask], MIN_DISTANCE)
            batch_height = get_hist_data(x, y, z, local_max_z_2, local_max_x_y_2)

            if HIST:    
                heights.append(batch_height)

            #plotting a scatter for example
            if PLOT3D:
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection = "3d")
                    ax.scatter(xs = x[mask], ys = y[mask], label = 'tree', s= 0.5, zs = z[mask])
                    ax.scatter(xs = x[~mask], ys = y[~mask], label = 'no tree', s= 0.005, zs = z[~mask], c = z[~mask], alpha = 0.1)
                    ax.scatter(xs = local_max_x_y_2[:, 0], ys = local_max_x_y_2[:, 1], zs = local_max_z_2, c = 'red', s = 6)
                    plt.show()
            
            print(f'# of trees with mask: {len(local_max_z_2)}')        
            
            if PLOT2D:
                plt.scatter(x[~mask], y[~mask], c = z[~mask] , label = 'no tree', s = 0.005, alpha =  0.5)
                plt.scatter(x[mask], y[mask], label = 'part of tree', s= 0.5, c = z[mask])
                plt.scatter(local_max_x_y_2[:, 0], local_max_x_y_2[:, 1], c = 'red', s = 6, label = 'local max tree')
                plt.legend()
                plt.xlabel('x coordinate [m]')
                plt.ylabel('y coordinate [m]')

                cbar = plt.colorbar()
                cbar.set_label('Height (normalized)', rotation=270, labelpad=15)
                
                plt.show()
    if HIST:
        heights = np.array([val for sublist in heights for val in sublist])
        mask = heights < 30
        heights = heights[mask]
        print(np.mean(heights))
        print(np.std(heights))
        plt.xlim([0, 32])
        plt.hist(heights, bins = 150)
        plt.xlabel('Tree height [m]')
        plt.ylabel('Trees in bin')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    main()