import os
import shutil
import sys
import shutil
import glob

from arcgis.learn import PointCNN

# To use this code, it is assumed a /tmp/ folder is available
PARALLEL = False

BUCKET_LOC = '/dbfs/mnt/group22/'
DATA_LOC = '/dbfs/mnt/lsde/ahn3/'
TEMP_FOLDERS = ['/tmp/laz_temp', '/tmp/las_out','/tmp/files']
TEMP_LAZ_NAME = 'tmp.LAZ'
TEMP_FOLDER = '/tmp/'
OUT_PRED_PATH = 'dbfs:/mnt/group22/files/'
AHN3_PATH = "/dbfs/mnt/lsde/ahn3/*"

# Can be obtained from https://www.arcgis.com/home/item.html?id=58d77b24469d4f30b5f68973deb65599
MODEL_PATH = '/dbfs/mnt/group22/Tree_point_classification-1.dlpk'

# This is where the lasmerge binary file from LAStools should reside
LASMERGE_PATH = '/dbfs/mnt/group22/lasmerge'

FC = PointCNN.from_model(MODEL_PATH)

def renew_folder(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.mkdir(path)
  
def remove_folder(path):
  if os.path.exists(path):
    shutil.rmtree(path)
    
def unzip_split_laz(filename):
    """
    Given a laz file splits the file up in chunks of 250 mb
    and return the folders in which those files reside
    """
    import os
    import shutil
    [renew_folder(x) for x in TEMP_FOLDERS]
    print('renewed folders')
    shutil.copy(DATA_LOC + filename, TEMP_FOLDERS[0] + TEMP_LAZ_NAME)
    
    # Change to directory where file resides
    os.chdir(TEMP_FOLDER)
    
    # Split the file
    os.popen('./lasmerge -i /tmp/laz_temp/tmp.LAZ -o /tmp/las_out/out0000.las -split 10000000')
    
    las_path = TEMP_FOLDERS[1]
    onlyfiles = [f for f in os.listdir(las_path) if os.path.isfile(os.path.join(las_path, f))]
    out_folders = []
    
    for file in onlyfiles:
      out_folder = '/tmp/files/' + str(file.split('.las')[0])
      out_folders.append(out_folder)
      os.mkdir(out_folder)
      shutil.move('/tmp/las_out/' + file, out_folder + '/' + str(file))
      print(out_folder)
    return out_folders

def get_loc_glob_filenames_pred(in_path_las, filename):
    out_file_local = in_path_las + '/results/' + (in_path_las.split('/')[-1]).split('.las')[0] + '_pred.las'
    out_dir = OUT_PRED_PATH + filename
    out_file_s3 = out_dir + '/' + out_file_local.split('/')[-1]
    return (out_file_local, out_dir, out_file_s3)
  
def process_las(in_path_las, filename):
    
    # Get the relevant paths
    out_file_local, out_dir, out_file_s3 = get_loc_glob_filenames_pred(in_path_las, filename)
    
    # Check if prediction has been done before
    if os.path.exists(out_file_s3):
        return True
    
    FC.predict_las(in_path_las)
    
    # If none of the files of a particular LAZ have not been processed
    # we make the folder in S3
    if not os.path.exists(out_dir):
        dbutils.fs.mkdirs(out_dir)
        
    # Copy to S3
    dbutils.fs.cp('file:' + out_file_local, out_file_s3)
    
    print('Single prediction done.')
    return True

def get_filenames(path):
    import glob
    all_laz_files = glob.glob(path + '*.LAZ')
    import os
    return all_laz_files 

def process_single_laz(filename):
    shutil.copy(LASMERGE_PATH, '/tmp/lasmerge')
    output_folders_las = unzip_split_laz(filename)
    for folder in output_folders_las:
          process_las(folder, filename)

def process_all_laz():
    shutil.copy(LASMERGE_PATH, '/tmp/lasmerge')
    filenames = [x.split('/')[-1] for x in get_filenames(AHN3_PATH)]
    if PARALLEL:
        # Under this is the code for having many machines
        para_path = sc.parallelize(out_folders)
        df = para_path.map(lambda x: process_las(x, filename)).toDF()
        df.count()    
    else:
        for filename in filenames:
            out_folders = unzip_split_laz(filename)
            for folder in out_folders:
                process_las(folder, filename)

process_single_laz('C_25GN2.LAZ')