import laspy
import numpy as np
import numba

from scipy.spatial.kdtree import KDTree
from mpl_toolkits.mplot3d import Axes3D

points_per_chunk = 100_000

OUT_PATH = '/dbfs/mnt/group22/txt_pred_old/'
PRED_PATH = '/dbfs/mnt/group22/las_out_pred_2/'

def locally_extreme_points(coords, data, neighbourhood, lookfor = 'max', p_norm = 2.):
    '''
    inspired by: https://stackoverflow.com/questions/27032562/local-maxima-in-a-point-cloud

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
    i_am_extreme = [data[i]==extreme_fcn(data[n]) and len(neighbours[i]) > 200 for i, n in enumerate(neighbours)]
    extrema, = np.nonzero(i_am_extreme) 
    x_y_z_max = np.zeros((len(data[extrema]), 3))
    x_y_z_max[:, 0:2] = coords[extrema]
    x_y_z_max[:, 2] = data[extrema]
    return x_y_z_max

@numba.njit
def get_overlap_masks(m1, m2):
    """
    Calculate the and operation of two mask arrays element-wise

    m1, m2: Masks of which to calculate the and operation

    returns
        overlap of m1 and m2
    """
    for i in range(points_per_chunk):
        if m1[i] == 1 and m2[i] == 1:
            m1[i] = 1
        else:
            m1[i] = 0
    return m1     


def get_filepaths():
    """
    Get a list of filenames where the prediction files reside
    """
    import os
    # Path where prediction files are
    path = PRED_PATH
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # Append the file name to the list
            filelist.append(os.path.join(root,file))
    return filelist

def get_trees(fpath):
    """
    Convert the prediction files to txt files 
    containing the trees
    """
    fname = fpath.split('/')[-1]

    # Where to write the output files
    f_write = OUT_PATH + f"{fname}.txt"
    
    with open(f_write, 'wb') as treefile:

        with laspy.open(fpath) as file:
            
            for points in file.chunk_iterator(points_per_chunk):
              
                mask = np.array(points.classification) > 0
                mask2 = np.array(points.number_of_returns) > 1
                mask = get_overlap_masks(mask, mask2) 
              
                # local max
                x = np.array(points.x)
                x_mask = x[mask]
                y = np.array(points.y)
                y_mask = y[mask]
                points_array = np.array((x, y)).T
                z = np.array(points.z)
                z_mask = z[mask]
                xyz_max = locally_extreme_points(points_array[mask], z[mask], 4)
                
                np.savetxt(treefile, xyz_max, delimiter=",")
    return True

def main():
    PARALLEL = True

    filelist = get_filepaths()
    if PARALLEL:
        rdd = sc.parallelize(filelist)
        df = rdd.map(lambda filename: get_trees(filename)).count()
    else:
        # If parallel doesn't work
        for filename in filelist:
            get_trees(filename)

if __name__ == '__main__':
    main()
