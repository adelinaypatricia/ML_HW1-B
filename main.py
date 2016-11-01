from os import listdir, path
from PIL import Image

from sklearn.decomposition import PCA
import numpy as np
from math import log10


# settting of the files path for analysis
def get_file_list(img_dir='C:/data/ML/training/'):
    files_path = ["{}/{}".format(img_dir, f) for f in listdir(img_dir)
                  if path.isfile(path.join(img_dir, f))]
    return files_path

def main():
    # get the list of image paths
    files_path_list = get_file_list()

    # load each image from files_path_list
    imgs_list = [Image.open(img) for img in files_path_list]

    # convert image to numpy.array
    imgs_array_list = [np.array(img) for img in imgs_list]

    # calculate the average of all image
    avg_img_array = np.average(np.array(imgs_array_list), 0)

    # save the average image
    avg_img = Image.fromarray(np.uint8(avg_img_array))
    avg_img.save("avg.tif")

    # feature matrix generate
    feature_matrix = np.vstack([item.reshape(item.shape[0]*item.shape[1]) for item in imgs_array_list])

    # get ans1: top 5 eigenfaces and their corresponding eigenvalues in a descending order
    pca = PCA(5)
    pca.fit(feature_matrix)
    eigenvalue = pca.explained_variance_[:5]
    for i, eigen_vector in enumerate(pca.components_[:5]):
        Image.fromarray((eigen_vector.reshape(128, 128) - eigen_vector.min()) / (eigen_vector - eigen_vector.min()).max()).save('top{}.tif'.format(i))
    print('eigenvalue: {}'.format(eigenvalue))

    #--------------------ans1----------------------#

    # train the test image(hw01-test.tif), compute top 10 eigenface coefficients
    pca = PCA(10)
    pca.fit(feature_matrix)
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))[0, :10]

    for i, eigen_vector in enumerate(pca.components_[:35]):
        Image.fromarray((eigen_vector.reshape(128, 128) - eigen_vector.min()) / (eigen_vector - eigen_vector.min()).max()).save('e{}.tif'.format(i))

    # get ans2: top 10 eigenface
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))[0, :10]

    #--------------------ans2----------------------#

    # Keep only first K (K=5,10,15,20, and 25) coefficients
    pca = PCA(5)
    pca.fit(feature_matrix)
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))

    # Reconstruct the image in the pixel domain
    test1 = pca.inverse_transform(pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))).reshape((16384,))
    Image.fromarray(np.uint8(((test1 - test1.min()) / (test1 - test1.min()).max() * 255)).reshape(128, 128)).save('test1.tif')

    # When K = 10
    pca = PCA(10)
    pca.fit(feature_matrix)
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))
    test2 = pca.inverse_transform(pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))).reshape((16384,))
    Image.fromarray(np.uint8(((test2 - test2.min()) / (test2 - test2.min()).max() * 255)).reshape(128, 128)).save('test2.tif')

    # When K = 15
    pca = PCA(15)
    pca.fit(feature_matrix)
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))
    test3 = pca.inverse_transform(pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))).reshape((16384,))
    Image.fromarray(np.uint8(((test3 - test3.min()) / (test3 - test3.min()).max() * 255)).reshape(128, 128)).save('test3.tif')

    # When K = 20
    pca = PCA(20)
    pca.fit(feature_matrix)
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))
    test4 = pca.inverse_transform(pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))).reshape((16384,))
    Image.fromarray(np.uint8(((test4 - test4.min()) / (test4 - test4.min()).max() * 255)).reshape(128, 128)).save('test4.tif')

    # When K = 25
    pca = PCA(25)
    pca.fit(feature_matrix)
    pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))
    test5 = pca.inverse_transform(pca.transform(np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape((1, 16384)))).reshape((16384,))
    Image.fromarray(np.uint8(((test5 - test5.min()) / (test5 - test5.min()).max() * 255)).reshape(128, 128)).save('test5.tif')


    # When K = 5,  compare the reconstructed image with the original image by PSNR value
    item = np.array(Image.open('C:/data/ML/hw01-test.tif')).reshape(16384)
    mse = ((item-test1) ** 2).sum()/16384
    psnr1 = log10(255/mse ** 0.5) * 20
    print('psnr1: {}'.format(psnr1))

    # When K = 10,  compare the reconstructed image with the original image by PSNR value
    mse = ((item-test2) ** 2).sum()/16384
    psnr2 = log10(255/mse ** 0.5) * 20
    print('psnr2: {}'.format(psnr2))

    # When K = 15,  compare the reconstructed image with the original image by PSNR value
    mse = ((item-test3) ** 2).sum()/16384
    psnr3 = log10(255/mse ** 0.5) * 20
    print('psnr3: {}'.format(psnr3))

    # When K = 20,  compare the reconstructed image with the original image by PSNR value
    mse = ((item-test4) ** 2).sum()/16384
    psnr4 = log10(255/mse ** 0.5) * 20
    print('psnr4: {}'.format(psnr4))

    # When K = 25,  compare the reconstructed image with the original image by PSNR value
    mse = ((item-test5) ** 2).sum()/16384
    psnr5 = log10(255/mse ** 0.5) * 20
    print('psnr5: {}'.format(psnr5))

    #------------ans3-------------------------#




if __name__ == "__main__":
    main()
