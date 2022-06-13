from PIL import Image, ImageFilter
from shapely.geometry import Polygon, shape, mapping, asPolygon
import large_image
from histomicstk.saliency.tissue_detection import (get_tissue_mask)
import cv2
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import rasterio.features
from shapely.geometry import Polygon, shape, GeometryCollection

class DetectTissue():

    def get_wsi_mask(self, wsi):

        # First we load WSI
        ts = large_image.getTileSource(wsi)
        # get maximum magfnification
        native_mag = int(ts.getNativeMagnification()['magnification'])
        # get the thumbnail of the WSI
        thumbnail_rgb, _ = ts.getRegion(scale=dict(magnification=1),
                                        format=large_image.tilesource.TILE_FORMAT_NUMPY)
        thumbnail = thumbnail_rgb[:, :, 0:3]
        plt.imshow(thumbnail)

        # resize factors
        factors = np.array([native_mag, native_mag])

        # Generate the tissue mask based on the thumbnail
        mask_out, mask = get_tissue_mask(
            thumbnail, deconvolve_first=False,
            n_thresholding_steps=1, sigma=0., min_size=1)

        # convert numpy array to PIL
        pil_mask = Image.fromarray(np.uint8(mask_out * 255))

        # smooth edges to reduce polygon indices
        smth_mask = pil_mask.filter(ImageFilter.ModeFilter(size=50))

        #convert PIL to numpy array
        arr_mask = np.asarray(smth_mask)

        #  mask to shapes
        shp_mask = rasterio.features.shapes(arr_mask)

        poly_mask = []
        for polygon, value in shp_mask:
            if value > 0:
                poly = shape(polygon)
                if poly.geom_type == 'Polygon' and poly.area > 1:
                    clean = poly.simplify(1, preserve_topology=False)
                    #skip if smoothing polygon provides empty list
                    if clean.is_empty:
                        continue
                    clean = clean.buffer(0.0)
                    poly_mask.append(clean.buffer(0.0))
                    x, y = clean.exterior.xy
                    plt.plot(x, y, color='red')
        plt.gca().invert_yaxis()

        #resizing
        resized_mask = []
        for ply in poly_mask:
            ply_arr = array(array(ply.exterior) * factors).astype(int)
            x, y = asPolygon(ply_arr).exterior.xy
            plt.plot(x, y, color='red')
            resized_mask.append({'name': 'tissue', 'color': '#000000', 'polygon': asPolygon(ply_arr)})
        plt.gca().invert_yaxis()
        return resized_mask


    def has_tissue(self, img, tile_width, tile_height):
        if (img.shape[0] < int(tile_width) or img.shape[1] < int(tile_height)):
            return False
        img_area = int(img.shape[0]) * int(img.shape[0])
        MAX_WHITE_SIZE = img_area - img_area * 30 / 100
        # tile is nparray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return self.get_cnt_sum(contours, 2) < MAX_WHITE_SIZE


    def get_cnt_sum(self, contours, topn):
        res = 0
        cnts = sorted(contours, key=lambda x: cv2.contourArea(x))[-topn:]
        return sum([cv2.contourArea(cnt) for cnt in cnts])
