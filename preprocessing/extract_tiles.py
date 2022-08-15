import os
import gc
import cv2

from tqdm import tqdm
from PIL import Image
import large_image
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
from tissue_detection import DetectTissue

import matplotlib.pyplot as plt

import glob
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization


class WholeSlideImage:
    """
    An application that utilize HistomicsTK to tile SVS images.

    """
    @staticmethod
    def macenko_normalization(tile):
        # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
        # for Macenko (obtained using rgb_separate_stains_macenko_pca()
        # and reordered such that columns are the order:
        # Hematoxylin, Eosin, Null
        W_target = np.array([[0.5807549, 0.08314027, 0.08213795],
                             [0.71681094, 0.90081588, 0.41999816],
                             [0.38588316, 0.42616716, -0.90380025]])
        stain_unmix_params = {'stains': ['hematoxylin', 'eosin'], 'stain_unmixing_method': 'macenko_pca'}
        normalized_tile = deconvolution_based_normalization(tile, W_target=W_target,
                                                            stain_unmixing_routine_params=stain_unmix_params)
        return Image.fromarray(normalized_tile)

    @staticmethod
    def artifact_filter(im, white=True, red=True, green=True):
        # Threshold values
        red_thresh = 256 * 256 * 0.05
        green_thresh = 256 * 256 * 0.015
        white_thresh = 256 * 256 * 0.35

        hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

        # TODO: Maybe change control flow? cascading logic -> True = filter ON = reject tile unless passes threshold
        if white:
            low_white = np.array([0, 0, 227])
            high_white = np.array([255, 28, 255])
            mask = cv2.inRange(hsv_im, low_white, high_white)
            white = cv2.bitwise_and(im, im, mask=mask)
            white = (white > 0).sum(axis=2)
            white = (white > 0).sum() > white_thresh
        # TODO: widen red filter to exclude more RBCs regions?
        if not white and red:
            low_red = np.array([0, 126, 175])
            high_red = np.array([25, 255, 255])
            red_mask = cv2.inRange(hsv_im, low_red, high_red)
            red = cv2.bitwise_and(im, im, mask=red_mask)
            red = (red > 0).sum(axis=2)
            red = (red > 0).sum() > red_thresh
        if not white and not red and green:
            low_green = np.array([40, 0, 0])
            high_green = np.array([113, 255, 227])
            green_mask = cv2.inRange(hsv_im, low_green, high_green)
            green = cv2.bitwise_and(im, im, mask=green_mask)
            green = (green > 0).sum(axis=2)
            green = (green > 0).sum() > green_thresh
        reject = white or red or green
        return not reject

    @staticmethod
    def save_tile_to_disk(im, x, y, pos, file_name, output_dir):
        tile_fname = os.path.basename(file_name) + "_" + str(x) + "_" + str(y) + "_" + str(pos)
        im.save(output_dir + tile_fname + ".png")

    @staticmethod
    def get_tissue_region(file, mag):
        detect_tissue = DetectTissue()
        return detect_tissue.get_wsi_mask(file, mag)

    def tile_wsi(self, file, tile_width, tile_height, mag_out, output_dir=None):
        # read svs image
        ts = large_image.getTileSource(file)
        max_width, max_height = ts.getMetadata()['sizeX'], ts.getMetadata()['sizeY']
        native_mag = ts.getNativeMagnification()['magnification']

        # get tissue region
        tissue_regions = self.get_tissue_region(file, native_mag)
        # TODO: Add an option to use Jason's "get Aline's annotation" function
        # See tissue_detection.py for additional changes to implement

        # Faster search algo - STR Tree
        alpha = native_mag / mag_out
        scaled_width = tile_width * alpha
        scaled_height = tile_height * alpha

        x = np.arange(0 + (scaled_width/2 - 1), max_width, scaled_width)
        y = np.arange(0 + (scaled_height/2 - 1), max_height, scaled_height)
        xx, yy = np.meshgrid(x, y)

        pts = [Point(X, Y) for X, Y in zip(xx.ravel(), yy.ravel())]
        tree = STRtree(pts)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(pts))
        points = [(index_by_id[id(pt)], pt.wkt) for ply in tissue_regions
                  for pt in tree.query(ply['polygon'])
                  if pt.within(ply['polygon'])]
        points.sort(key=lambda i: i[0])

        # Sanity check: makes sure tiles are pulled from expected WSI regions
        # test_width = int(np.round(max_width/(tile_width * alpha)))
        # test_height = int(np.round(max_height/(tile_height * alpha)))
        # arr = np.ones((test_height, test_width), dtype=int)
        # for pt in points:
        #     r, c = int(np.round(pt[0] / test_width)), pt[0] % test_width
        #     arr[r,c] = 0
        # plt.imshow(arr, cmap='Greys_r')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig(file[:-3] + 'tiles.png', format='png', bbox_inches='tight')
        # #plt.show()
        # plt.close()

        pre_fil, post_fil, err = 0, 0, 0
        for pt in tqdm(points):
            try:
                point = pt[1].split()
                x, y = int(point[1][1:]), int(point[2][:-1])
                scaled_x = x - scaled_width/2
                scaled_y = y - scaled_height/2
                im_roi, _ = ts.getRegion(region=dict(left=scaled_x,
                                                     top=scaled_y,
                                                     width=scaled_width,
                                                     height=scaled_height,
                                                     units='base_pixels'),
                                         scale=dict(magnification=mag_out),
                                         format=large_image.tilesource.TILE_FORMAT_NUMPY)
                if im_roi.shape != (256, 256, 4): continue
                tile = im_roi[:, :, :3]
                pre_fil += 1
                if self.artifact_filter(tile, white=True, red=True, green=True):
                    norm_tile = self.macenko_normalization(tile)
                    self.save_tile_to_disk(norm_tile, scaled_x, scaled_y, pt[0], file, output_dir)
                    post_fil += 1
            except (ValueError, IndexError):
                self.save_tile_to_disk(Image.fromarray(im_roi), scaled_x, scaled_y, pt[0], file, 'data/errors/')
                err += 1
                continue
        print(file, '-----> Total tiles:', pre_fil, '-----> Clean tiles:', post_fil, '-----> Error:', err)


if __name__ == '__main__':
    wsi = WholeSlideImage()
    path = 'data/slides/*.svs'
    for f in glob.glob(path):
        wsi.tile_wsi(f, tile_width=256, tile_height=256, mag_out=40, output_dir='data/03notR/')
        # free memory after each WSI
        gc.collect()
