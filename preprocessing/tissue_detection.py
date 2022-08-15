from PIL import Image
from shapely.geometry import Polygon, shape
import large_image
from histomicstk.saliency.tissue_detection import get_tissue_mask
import numpy as np
import rasterio.features

import glob
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class DetectTissue:
    @staticmethod
    def get_img(wsi, mag):
        ts = large_image.getTileSource(wsi)
        # print(ts.getMetadata())
        thumbnail_rgb, _ = ts.getRegion(scale=dict(magnification=mag),
                                        format=large_image.tilesource.TILE_FORMAT_NUMPY)
        return thumbnail_rgb[:, :, 0:3]

    @staticmethod
    def get_tissue_mask(im, largest, deconvolve):
        mask_all, mask_lg = get_tissue_mask(im, deconvolve_first=deconvolve, n_thresholding_steps=1,
                                            sigma=1., min_size=50)
        if largest: mask_all = mask_lg
        pil_mask = Image.fromarray(np.uint8(mask_all * 255))
        return rasterio.features.shapes(np.asarray(pil_mask))

    @staticmethod
    def get_polygons(shapes):
        polygons = []
        for polygon, value in shapes:
            if value > 0:
                poly = shape(polygon)
                if poly.geom_type == 'Polygon' and poly.area > 75.:
                    clean = poly.simplify(1, preserve_topology=False)
                    if clean.is_empty or clean.geom_type == 'MultiPolygon':
                        continue
                    clean = clean.buffer(0.0)
                    polygons.append(clean.buffer(0.0))
        #             Sanity check: does polygon encompass tissue on thumbnail image?
        #             x, y = clean.exterior.xy
        #             plt.plot(x, y, color='blue', linewidth=2)
        # plt.gca().invert_yaxis()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        # plt.close()
        return polygons

    @staticmethod
    def enlarge_polygons(polygons, mag, scale): # , wsi):
        resized_mask = []
        factors = np.array([mag, mag]) * 1/scale
        for ply in polygons:
            ply_arr = np.array(np.array(ply.exterior) * factors).astype(int)
            resized_mask.append({'polygon': Polygon(ply_arr)})
        #     Sanity check: does enlarged polygon match expected dimensions of user-requested magnification?
        #     x, y = Polygon(ply_arr).exterior.xy
        #     plt.plot(x, y, color='red', linewidth=1)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.gca().invert_yaxis()
        # plt.show()
        # plt.close()
        return resized_mask

    def get_wsi_mask(self, file, mag, scale=0.25, largest=False, deconvolve=False):
        deconv = ['TCGA-02-0047-01Z-00', 'TCGA-14-1034-01Z-00-DX2', 'TCGA-06-0882-01Z-00']
        if any(x in file for x in deconv): deconvolve = True
        im = self.get_img(file, mag=scale)
        # scaled_mask should return Aline's notation at her specified magnification
        scaled_mask = self.get_tissue_mask(im, largest=largest, deconvolve=deconvolve)
        poly_mask = self.get_polygons(scaled_mask)
        # enlarge_polygons should return Aline's polygons enlarged to the native magnification of the associated WSI
        return self.enlarge_polygons(poly_mask, mag, scale)  # , wsi)


if __name__ == '__main__':
    mask = DetectTissue()
    path = 'data/slides/TCGA-CS-4942-01Z-00-DX1.67f7928e-e1d9-473b-a317-7c6b4ba7433f.svs'
    for f in glob.glob(path):
        mask.get_wsi_mask(f, 20.)
        gc.collect()
