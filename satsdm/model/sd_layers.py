import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from qgis.core import (
    QgsRasterLayer, QgsProject, QgsColorRampShader,
    QgsRasterShader, QgsSingleBandPseudoColorRenderer
)
from PyQt5.QtGui import QColor
from satsdm.model.utils import add_layer_with_rendering
from osgeo import ogr, gdal

import os
from osgeo import gdal
from qgis.core import QgsRasterLayer, QgsProject

import os
from osgeo import gdal, ogr
from qgis.core import QgsRasterLayer, QgsProject

def run_proximity_layer(vector_path, field_name, field_value, output_dir):
    """
    Create a proximity raster layer from a vector file,
    using a specified field and field value.
    """

    base_name = f"{field_value}_proximity".replace(" ", "_")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.tif")
    prox_path = os.path.join(output_dir, f"{base_name}.tif")

    # Open vector and get first layer name
    ds = ogr.Open(vector_path)
    if ds is None:
        print(f"[ERROR] Failed to open vector: {vector_path}")
        return

    layer_name = ds.GetLayer(0).GetName()

    # SQL expression for filtering
    expr = f'"{field_name}" = \'{field_value}\''
    print(f"[INFO] Rasterizing where {expr}")

    # Rasterize to create binary mask
    gdal.Rasterize(
        mask_path,
        vector_path,
        options=[
            "-burn", "1",
            "-l", layer_name,
            "-where", expr,
            "-ot", "Byte",
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            "-tr", "30", "30"
        ]
    )

    # Open the mask raster
    src_ds = gdal.Open(mask_path)
    src_band = src_ds.GetRasterBand(1)

    # Prepare output raster
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(
        prox_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Int32
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())

    # Compute proximity from mask
    gdal.ComputeProximity(
        src_band,
        dst_ds.GetRasterBand(1),
        options=[
            "VALUES=1",
            "DISTUNITS=PIXEL"
        ]
    )

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    # Load the result into QGIS
    layer = QgsRasterLayer(prox_path, base_name)
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        print(f"[INFO] Proximity layer added: {base_name}")
    else:
        print(f"[WARNING] Invalid raster layer: {prox_path}")

def run_terrain_indices(dem_path, indices, output_dir):
    """
    Compute terrain indices from a DEM raster and add them to the QGIS project.
    :param dem_path: Path to DEM raster (.tif)
    :param indices: List of indices to compute ['SLOPE', 'ASPECT', 'TPI']
    :param output_dir: Output directory for saving result rasters
    """
    if not os.path.isfile(dem_path):
        print(f"[ERROR] DEM file not found: {dem_path}")
        return

    for index in indices:
        index_lower = index.lower()
        out_path = os.path.join(output_dir, f"{index_lower}.tif")

        try:
            gdal.DEMProcessing(out_path, dem_path, index_lower, format='GTiff')
            layer = QgsRasterLayer(out_path, index.capitalize())
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                print(f"[INFO] {index} added to QGIS")
            else:
                print(f"[WARNING] {index} layer is not valid!")
        except Exception as e:
            print(f"[ERROR] Failed to compute {index}: {str(e)}")

def identify_satellite_type(folder):
    """
    Identify the satellite type based on file naming patterns inside the folder.
    """
    folder_name = os.path.basename(folder).upper()
    if "L2A" in folder_name:
        return "Sentinel-2"
    elif "LC8" in folder_name:
        return "Landsat"
    elif "K3_" in folder_name or "K3A_" in folder_name:
        return "KOMPSAT-3"
    return None

def update_index_checkboxes(ui, satellite):
    """
    Enable and check all spectral index checkboxes by default.
    Disable and uncheck unsupported indices for each satellite type.
    """
    # Enable and check all by default
    for cb in [
        ui.check_ndvi, ui.check_gndvi, ui.check_ndwi_mcfeeters,
        ui.check_ndwi_gao, ui.check_ndbi, ui.check_savi, ui.check_bai
    ]:
        cb.setEnabled(True)
        cb.setChecked(True)

    # Disable and uncheck unsupported indices for KOMPSAT-3
    if satellite == "KOMPSAT-3":
        ui.check_ndwi_gao.setEnabled(False)
        ui.check_ndwi_gao.setChecked(False)

        ui.check_ndbi.setEnabled(False)
        ui.check_ndbi.setChecked(False)

        ui.check_savi.setEnabled(False)
        ui.check_savi.setChecked(False)

        ui.check_bai.setEnabled(False)
        ui.check_bai.setChecked(False)

def get_color_ramp_for_index(index):
    index = index.upper()

    if index in ("NDVI", "GNDVI", "SAVI"):
        return ['#00441b', '#238b45', '#66c2a4', '#ccece6', '#f7fcfd']

    elif index in ("NDWI_MCFEETERS", "NDWI_GAO"):
        return ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#f7fbff']

    elif index == "NDBI":
        return ['#f7f7f7', '#cccccc', '#969696', '#525252', '#252525']

    elif index == "BAI":
        return ['#fff5f0', '#fcbba1', '#fc9272', '#fb6a4a', '#cb181d']

    else:
        # Default to Viridis if unknown
        return ['#440154', '#2d708e', '#fde725']

def run_selected_indices(folder, satellite, indices, output_dir):
    band_paths = detect_band_files(folder, satellite)

    for index in indices:
        path, name = compute_index(index, band_paths, satellite, output_dir)
        if path:
            spectrum = get_color_ramp_for_index(index)
            add_layer_with_rendering(path, name, spectrum)

def detect_band_files(folder, satellite):
    band_files = {}

    if satellite == "Sentinel-2":
        for f in os.listdir(folder):
            name = f.lower()
            full = os.path.join(folder, f)
            if name.endswith("_b04.tif"):
                band_files["RED"] = full
            elif name.endswith("_b08.tif"):
                band_files["NIR"] = full
            elif name.endswith("_b03.tif"):
                band_files["GREEN"] = full
            elif name.endswith("_b11.tif"):
                band_files["SWIR"] = full

    elif satellite == "Landsat":
        for f in os.listdir(folder):
            name = f.lower()
            full = os.path.join(folder, f)
            if name.endswith("_b04.tif"):
                band_files["RED"] = full
            elif name.endswith("_b05.tif"):
                band_files["NIR"] = full
            elif name.endswith("_b03.tif"):
                band_files["GREEN"] = full
            elif name.endswith("_b06.tif"):
                band_files["SWIR"] = full

    elif satellite == "KOMPSAT-3":
        for f in os.listdir(folder):
            name = f.lower()
            full = os.path.join(folder, f)
            if name.endswith("_r.tif"):
                band_files["RED"] = full
            elif name.endswith("_g.tif"):
                band_files["GREEN"] = full
            elif name.endswith("_b.tif"):
                band_files["BLUE"] = full
            elif name.endswith("_n.tif"):
                band_files["NIR"] = full
            elif name.endswith("_p.tif"):
                band_files["PAN"] = full

    return band_files

def compute_index(index, bands, satellite, output_dir):
    # load(): optionally reprojects if match array/profile is provided
    def load(path, match_array=None, match_profile=None):
        with rasterio.open(path) as src:
            src_data = src.read(1).astype('float32')
            src_profile = src.profile

            if match_array is None or match_profile is None:
                return src_data, src_profile

            dst = np.empty_like(match_array, dtype='float32')
            reproject(
                source=src_data,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=match_profile["transform"],
                dst_crs=match_profile["crs"],
                resampling=Resampling.bilinear
            )
            return dst, match_profile

    def save(array, profile, out_path):
        profile.update(dtype=rasterio.float32, count=1, driver='GTiff')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(array.astype(np.float32), 1)

    index = index.upper()
    out_path = os.path.join(output_dir, f"{index.lower()}.tif")

    try:
        if index == "NDVI":
            red, p = load(bands["RED"])
            nir, _ = load(bands["NIR"], red, p)
            result = (nir - red) / (nir + red + 1e-6)

        elif index == "GNDVI":
            green, p = load(bands["GREEN"])
            nir, _ = load(bands["NIR"], green, p)
            result = (nir - green) / (nir + green + 1e-6)

        elif index == "NDWI_MCFEETERS":
            green, p = load(bands["GREEN"])
            nir, _ = load(bands["NIR"], green, p)
            result = (green - nir) / (green + nir + 1e-6)

        elif index == "NDWI_GAO":
            nir, p = load(bands["NIR"])
            swir, _ = load(bands["SWIR"], nir, p)  # SWIR is 20m → reproject
            result = (nir - swir) / (nir + swir + 1e-6)

        elif index == "NDBI":
            swir, p = load(bands["SWIR"])
            nir, _ = load(bands["NIR"], swir, p)  # NIR is 10m → reproject
            result = (swir - nir) / (swir + nir + 1e-6)

        elif index == "SAVI":
            red, p = load(bands["RED"])
            nir, _ = load(bands["NIR"], red, p)
            L = 0.5
            result = ((nir - red) / (nir + red + L)) * (1 + L)

        elif index == "BAI":
            red, p = load(bands["RED"])
            nir, _ = load(bands["NIR"], red, p)
            result = 1.0 / ((red - 0.06) ** 2 + (nir - 0.1) ** 2 + 1e-6)

        else:
            return None, None

        save(result, p, out_path)
        return out_path, index

    except KeyError as e:
        print(f"[ERROR] Missing band for {index}: {str(e)}")
        return None, None





