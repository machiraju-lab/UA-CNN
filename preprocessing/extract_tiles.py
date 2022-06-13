import os
import gc
import cv2
from tqdm import tqdm
from PIL import Image
import large_image
import numpy as np
import argparse
from shapely.geometry import Polygon
from tissue_detection import DetectTissue

class WholeSlideImage():
    """
    An application that utilize HistomicsTK to tile SVS images.

    """
    def get_tissue_region(self, file):
        detect_tissue = DetectTissue()
        return detect_tissue.get_wsi_mask(file)

    def tile_wsi(self, file , magnification, tile_width, tile_height, output_dir):
        #read svs image
        ts = large_image.getTileSource(file)

        # get tissue region
        tissue_regions = self.get_tissue_region(file)

        for tile_info in tqdm(ts.tileIterator(
                scale=dict(magnification=magnification),
                tile_size=dict(width=tile_width, height=tile_height),
                format=large_image.tilesource.TILE_FORMAT_PIL)):

            #img
            im_tile = np.array(tile_info['tile'])

            tile_polygon = Polygon([(tile_info['gx'], tile_info['gy']),
                                    (tile_info['gx'] + tile_info['gwidth'], tile_info['gy']),
                                    (tile_info['gx'] + tile_info['gwidth'], tile_info['gy'] + tile_info['gheight']),
                                    (tile_info['gx'], tile_info['gy'] + tile_info['gheight'])])

            for tissue_region in tissue_regions:
                # check if a tile has a tissue
                if (tissue_region['polygon'].contains(tile_polygon)):
                    self.save_tile_to_disk(im_tile, (tile_info['x'], tile_info['y']), file, magnification, output_dir)
        print("Done tiling: ", file)



    def get_otsu_th(self, tile):
        # need to convert from PIL imave to opencv
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # threshold the S channel using adaptive method(`THRESH_OTSU`)
        th, threshed = cv2.threshold(s, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        return th


    def save_tile_to_disk(self, imtile, coords, file_name , magnification , output_dir):
        """ Saves numpy tiles to .png files (full resolution).
            Meta data is saved in the file name.
            - tile             numpy image
            - coords            x, y tile coordinates
            - file_name         original source WSI name
        """

        # Construct the new PNG filename
        tile_fname = os.path.basename(file_name) + "_" + str(coords[0]) + "_" + str(coords[1]) 
        
        tile = Image.fromarray(imtile).convert("RGB")

        #check otsu threshold before saving
        tile_otsu = self.get_otsu_th(imtile)
        if 40 < tile_otsu < 157:
            tile.save(output_dir + tile_fname + ".png")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', help="WSI folder path")
    parser.add_argument('--wsi', help="WSI file name")
    parser.add_argument("-m", "--magnification", help="magnification level")
    parser.add_argument("-wd", "--wd", help="Tile width")
    parser.add_argument("-ht", "--ht", help="Tile height")
    parser.add_argument("-o", "--output", help="path to output directory")

    args = parser.parse_args()
    slides_dir = args.tile
    wsi_file = args.wsi
    magnification = int(args.magnification)
    width = int(args.wd)
    height = int(args.ht)
    output = args.output

    wsi_dir = os.path.join(slides_dir,wsi_file)

    wsi = WholeSlideImage()
    wsi.tile_wsi(wsi_dir, magnification, width, height, output)
    # free memory after each WSI
    gc.collect()
