import os 
#get list of files in current directory and all subdirectories
def get_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        yield dirs
        yield files
        
#get current working directory
def get_current_directory

#split file path into parts

def split_path(path):
    path_parts = path.split(os.sep)
    return path_parts

#get list of files and file path in current directory and from all subdirectories and put in dataframe
def get_files_from_directory(directory):
    files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files.append(os.path.join(root, file))
    return files

#zip two lists together in pandas dataframe and name columns
def zip_:
    files = get_files_from_directory(directory)
    file_path = []
    for file in files:
        file_path.append(split_path(file))
    df = pd.DataFrame(file_path, columns=['file_path'])
    return df

# read images and list dimensions in a graph
def images_dimensions(directory):
    files = get_files_from_directory(directory)
    image_dimensions = []
    for file in files:
        img = cv2.imread(file)
        image_dimensions.append(img.shape)
    return image_dimensions



import cv2



# create scatterplot from pandas series of tuples

def scatter_():
    df = pd.DataFrame(image_dimensions, columns=['height', 'width', 'channels'])
    df.plot.scatter(x='height', y='width')
    
#build histogram of image dimensions
def histogram():
    df = pd.DataFrame(image_dimensions, columns=['height', 'width', 'channels'])
    df.hist()
    
#build a boxplot of one or more pandas columns displaying the distribution of the data
def boxplot():
    df = pd.DataFrame(image_dimensions, columns=['height', 'width', 'channels'])
    df.boxplot(column='height')
    

#get information and distrubution of data in pandas dataframe

def get_info