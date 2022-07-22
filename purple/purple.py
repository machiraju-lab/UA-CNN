# Some code used from Adrian Rosebrock's k-means to find dominant colors tutorial
from __future__ import division
from collections import Counter
from scipy.spatial import distance
import numpy as np
from scipy.stats import rankdata
import cv2
import glob
import os
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing
import time

white_all_tiles = []
pink_all_tiles = []
purple_all_tiles = []
total_all_tiles = []
truePurple_all_tiles = []


# ##########Helper Functions##################
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


# pink = [255,192,203]
# purple = [128,0,128]
# white = [255,255,255]
def ClusterIndicesNumpy(arr, clustNum, labels_array):  # numpy
    return arr[np.where(labels_array == clustNum)[0]]


def get_pixel_color_counts(filename):
    image = cv2.imread(filename)
    print("Working on " + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_arr = image.reshape((-1, 3))
    img_arr = np.float32(img_arr)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    temp, labels, (centers) = cv2.kmeans(img_arr, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    showdetails = 0
    # Remove "outliers" that don't belong to any pertinent color, i.e non-specific colors (top 20% most distant
    # pixels from cluster centroids)
    ranks0 = ((rankdata(distance.cdist(ClusterIndicesNumpy(img_arr, 0, labels),
                                       np.array([centers[0]]), 'euclidean')) - 1).astype(int))
    ranks1 = ((rankdata(distance.cdist(ClusterIndicesNumpy(img_arr, 1, labels),
                                       np.array([centers[1]]), 'euclidean')) - 1).astype(int))
    ranks2 = ((rankdata(distance.cdist(ClusterIndicesNumpy(img_arr, 2, labels),
                                       np.array([centers[2]]), 'euclidean')) - 1).astype(int))
    newranks0 = ranks0 < round(len(ranks0) * 0.80)
    newranks1 = ranks1 < round(len(ranks1) * 0.80)
    newranks2 = ranks2 < round(len(ranks2) * 0.80)

    labels0 = labels[np.where(labels == 0)[0]]
    labels1 = labels[np.where(labels == 1)[0]]
    labels2 = labels[np.where(labels == 2)[0]]

    newlabels0 = labels0[newranks0]
    newlabels1 = labels1[newranks1]
    newlabels2 = labels2[newranks2]

    newlabels = np.concatenate([newlabels0, newlabels1, newlabels2])
    # How many pixels in each cluster originally
    # Num_pix = Counter(labels)
    # How many pixels in each cluster after removing non-specific colors
    Num_pix_refined = Counter(newlabels)
    # Debugging print statements
    # print(Counter(labels))
    # print("centers")
    # print(clt.cluster_centers_)
    # Get the sum of R,G,B values for cluster centroids
    myList = [sum(item) for item in np.array(centers)]

    # Rank the sum of R,G,B values for cluster centroids (Higher the value, more proclivity for white, lower value
    # will be dark purple, middle value will be pink/red)
    ranks = ((rankdata(myList) - 1).astype(int))

    if showdetails:
        # Print cluster and member cluster details
        print("White is cluster: " + str(np.where(np.isin(ranks, [2]))[0][0] + 1) + ", With # pixels = " + str(
            Num_pix_refined[np.where(np.isin(ranks, [2]))[0][0]]))
        print("Pink/red is cluster: " + str(np.where(np.isin(ranks, [1]))[0][0] + 1) + ", With # pixels = " + str(
            Num_pix_refined[np.where(np.isin(ranks, [1]))[0][0]]))
        print("Purple is cluster: " + str(np.where(np.isin(ranks, [0]))[0][0] + 1) + ", With # pixels = " + str(
            Num_pix_refined[np.where(np.isin(ranks, [0]))[0][0]]))

        # Show cluster colors and bar histogram
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)
        zipped = zip(hist, clt.cluster_centers_)
        hist, clt.cluster_centers = zip(*zipped)
        i = 0
        for rgb in (clt.cluster_centers_).round():
            plt.title(i + 1)
            plt.imshow([[(rgb / 255)]])
            plt.show()
            i += 1

            plt.imshow(bar)
            plt.show()

    # purpledef = [128,0,128]
    total_pixels = len(labels)
    truepurple = centers[np.where(np.isin(ranks, [0]))[0][0]][1]
    purple = Num_pix_refined[np.where(np.isin(ranks, [0]))[0][0]] / total_pixels
    purple_cluster_center = centers[np.where(np.isin(ranks, [0]))[0][0]]

    return [filename, purple, truepurple, purple_cluster_center]


def main(IMAGE_FOLDER_PATH, OUTPUT_FOLDER_PATH):
    tic = time.time()
    # list_img = glob.glob(glob.glob("tiled1000/*/")[0]+"slide/*/*.jpeg")
    files = glob.glob(IMAGE_FOLDER_PATH + '/*')
    tile_ratios = []
    print(len(files))
    pool = multiprocessing.Pool(processes=1)
    results = pool.map(get_pixel_color_counts, files)
    results = get_pixel_color_counts(
        'data/TCGA-02-0047-01Z-00-DX1.4755D138-5842-4159-848C-4248D6D53DE0/tilesTCGA-DX-A6Z0-01Z-00-DX1.DA09422C-5FB5-4BA2-BBB3-82A0DEAEBE6E.svs_1024_11264.png')
    print("start")
    # purpledef = [128, 0, 128]
    # purple_all_tiles = [item[1] for item in results]
    # truePurple_all_tiles = [item[2] for item in results]
    # purple_center = [item[3] for item in results]
    # tp2 = truePurple_all_tiles.copy()
    # tp2.sort()
    # index = len(tp2) * .50
    # index2 = len(tp2) * .30
    # # threshold3 = tp2[math.floor(index)]
    # # threshold4 = tp2[math.floor(index2)]
    # threshold1 = 25
    # threshold2 = 90
    # purpleAll = [purple_all_tiles[i] for i in range(len(purple_all_tiles)) if
    #              all([abs(purple_center[i][0] - purpledef[0]) < threshold1, purple_center[i][1] < threshold2,
    #                   abs(purple_center[i][2] - purpledef[2]) < threshold1])]
    # out = []
    # for item in results:
    #     filename = item[0]
    #     if (all([abs(item[3][0] - purpledef[0]) < threshold1, item[3][1] < threshold2,
    #              abs(item[3][2] - purpledef[2]) < threshold1])):
    #         tset2, pval2 = stats.ttest_1samp(purpleAll, item[1], alternative='less')
    #         if pval2 < 0.30 and item[1] > 0.15:
    #             out.append(item[1])
    #             basename = os.path.splitext(os.path.basename(filename))[0]
    #             basepath = os.path.join(OUTPUT_FOLDER_PATH, basename)
    #             fileout = basepath + ".jpeg"
    #             img = cv2.imread(filename)
    #             cv2.imwrite(fileout, img)
    #             '''
    #         else:
    #             basename = os.path.splitext(os.path.basename(filename))[0]
    #             basepath = os.path.join(OUTPUT_FOLDER_PATH2, basename)
    #             fileout =basepath +".jpeg"
    #             img = cv2.imread(filename)
    #             cv2.imwrite(fileout, img)
    #             '''
    # print(out)
    # print(max(out))
    # totalmax = max(out)
    # with open('maxs.txt', 'a') as f:
    #     f.write(IMAGE_FOLDER_PATH)
    #     f.write('\n')
    #     f.write(str(totalmax))
    #     f.write('\n')
    # toc = time.time()
    # print("time")
    # print(toc - tic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfolder', help="input folder")
    parser.add_argument('--outputfolder', help="outputfolder")
    parser.add_argument('--processes', help="processes")
    # parser.add_argument('--outputfolder2', help="outputfolder2")
    # parser.add_argument('--filetype', help="file type (png or jpeg)")
    args = parser.parse_args()
    IMAGE_FOLDER_PATH = args.inputfolder
    OUTPUT_FOLDER_PATH = args.outputfolder
    processes = args.processes
    main(IMAGE_FOLDER_PATH, OUTPUT_FOLDER_PATH)
