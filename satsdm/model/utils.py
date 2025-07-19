import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
import pickle
import gzip
import os

from typing import Tuple, Any, Union, List, Dict, Iterable
from pyproj import CRS as CRSType
from numbers import Number
from shapely.geometry import Point

from qgis.core import (
    QgsRasterLayer, QgsProject, QgsColorRampShader,
    QgsRasterShader, QgsSingleBandPseudoColorRenderer
)
from PyQt5.QtGui import QColor
from osgeo import gdal

Vector = Union[gpd.GeoSeries, gpd.GeoDataFrame]

class NoDataException(Exception):
    pass

def to_iterable(var: Any) -> list:
    """Checks and converts variables to an iterable type.

    Args:
        var: The input variable to check and convert.

    Returns:
        A list containing var or var itself if already iterable (except strings).
    """
    if not hasattr(var, "__iter__"):
        return [var]
    elif isinstance(var, str):
        return [var]
    else:
        return var

def validate_gpd(geo: Vector) -> None:
    """Validates whether an input is a GeoDataFrame or a GeoSeries.

    Args:
        geo: an input variable that should be in GeoPandas format

    Raises:
        TypeError: if geo is not a GeoPandas dataframe or series
    """
    if not isinstance(geo, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise TypeError("Input must be a GeoDataFrame or GeoSeries")

def string_to_crs(crs_str: str) -> CRSType:
    """Converts a CRS string to a pyproj CRS object."""
    return CRSType.from_user_input(crs_str)

def crs_match(crs1: Union[CRSType, str], crs2: Union[CRSType, str]) -> bool:
    """Evaluates whether two coordinate reference systems are the same.

    Args:
        crs1: The first CRS (CRS object or string).
        crs2: The second CRS (CRS object or string).

    Returns:
        True if CRS match, else False.
    """
    if isinstance(crs1, str):
        crs1 = string_to_crs(crs1)
    if isinstance(crs2, str):
        crs2 = string_to_crs(crs2)

    return crs1 == crs2

def count_raster_bands(raster_paths: list) -> int:
    """Returns the total number of bands from a list of rasters.

    Args:
        raster_paths: List of raster data file paths.

    Returns:
        n_bands: total band count.
    """
    n_bands = 0
    for path in raster_paths:
        with rio.open(path) as src:
            n_bands += src.count
    return n_bands

def n_digits(number: Number) -> int:
    """Counts the number of significant integer digits of a number.

    Args:
        number: the number to evaluate.

    Returns:
        order: number of digits required to represent a number
    """
    if number == 0:
        return 1
    return int(np.floor(np.log10(abs(number))) + 1)


def make_band_labels(n_bands: int) -> list:
    """Creates a list of band names to assign as dataframe columns.

    Args:
        n_bands: total number of raster bands to create labels for.

    Returns:
        labels: list of column names.
    """
    n_zeros = n_digits(n_bands)
    labels = [f"b{i + 1:0{n_zeros}d}" for i in range(n_bands)]
    return labels

def format_band_labels(raster_paths: list, labels: List[str] = None):
    """Verify the number of labels matches the band count, create labels if none passed.

    Args:
        raster_paths: count the total number of bands in these rasters.
        labels: a list of band labels.

    Returns:
        labels: creates default band labels if none are passed.
    """
    n_bands = count_raster_bands(raster_paths)

    if labels is None:
        labels = make_band_labels(n_bands)

    n_labels = len(labels)
    assert n_labels == n_bands, f"number of band labels ({n_labels}) != n_bands ({n_bands})"

    return labels.copy()

def get_feature_types(return_string: bool = False) -> Union[list, str]:
    feature_types = "lqpht" if return_string else ["linear", "quadratic", "product", "hinge", "threshold"]
    return feature_types

def validate_feature_types(features: Union[str, list]) -> list:
    """Ensures the feature classes passed are maxent-legible

    Args:
        features: List or string that must be in ["linear", "quadratic", "product",
            "hinge", "threshold", "auto"] or string "lqphta"

    Returns:
        valid_features: List of formatted valid feature values
    """
    valid_list = get_feature_types(return_string=False)
    valid_string = get_feature_types(return_string=True)
    valid_features = list()

    # ensure the string features are valid, and translate to a standard feature list
    if type(features) is str:
        for feature in features:
            if feature == "a":
                return valid_list
            assert feature in valid_string, "Invalid feature passed: {}".format(feature)
            if feature == "l":
                valid_features.append("linear")
            elif feature == "q":
                valid_features.append("quadratic")
            elif feature == "p":
                valid_features.append("product")
            elif feature == "h":
                valid_features.append("hinge")
            elif feature == "t":
                valid_features.append("threshold")

    # or ensure the list features are valid
    elif type(features) is list:
        for feature in features:
            if feature == "auto":
                return valid_list
            assert feature in valid_list, "Invalid feature passed: {}".format(feature)
            valid_features.append(feature)

    return valid_features

def repeat_array(x: np.array, length: int = 1, axis: int = 0) -> np.ndarray:
    """Repeats a 1D numpy array along an axis to an arbitrary length

    Args:
        x: the n-dimensional array to repeat
        length: the number of times to repeat the array
        axis: the axis along which to repeat the array (valid values include 0 to n+1)

    Returns:
        An n+1 dimensional numpy array
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)

def validate_boolean(var: Any) -> bool:
    """Evaluates whether an argument is boolean True/False

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not boolean
    """
    assert isinstance(var, bool), "Argument must be True/False"
    return var

def validate_numeric_scalar(var: Any) -> bool:
    """Evaluates whether an argument is a single numeric value.

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not numeric.
    """
    assert isinstance(var, (int, float)), "Argument must be single numeric value"
    return var

def save_object(obj: object, path: str, compress: bool = True) -> None:
    """Writes a python object to disk for later access.

    Args:
        obj: a python object or variable to be saved (e.g., a MaxentModel() instance)
        path: the output file path
    """
    obj = pickle.dumps(obj)

    if compress:
        obj = gzip.compress(obj)

    with open(path, "wb") as f:
        f.write(obj)

def get_raster_band_indexes(raster_paths: list) -> Tuple[int, list]:
    """Counts the number raster bands to index multi-source, multi-band covariates.

    Args:
        raster_paths: a list of raster paths

    Returns:
        (nbands, band_idx): int and list of the total number of bands and the 0-based start/stop
            band index for each path
    """
    nbands = 0
    band_idx = [0]
    for i, raster_path in enumerate(raster_paths):
        with rio.open(raster_path) as src:
            nbands += src.count
            band_idx.append(band_idx[i] + src.count)

    return nbands, band_idx

def create_output_raster_profile(
    raster_paths: list,
    template_idx: int = 0,
    windowed: bool = True,
    nodata: Number = None,
    count: int = 1,
    compress: str = None,
    driver: str = "GTiff",
    bigtiff: bool = True,
    dtype: str = "float32",
) -> Tuple[Iterable, Dict]:
    """Gets parameters for windowed reading/writing to output rasters.

    Args:
        raster_paths: raster paths of covariates to apply the model to
        template_idx: index of the raster file to use as a template. template_idx=0 sets the first raster as template
        windowed: perform a block-by-block data read. slower, but reduces memory use
        nodata: output nodata value
        count: number of bands in the prediction output
        driver: output raster file format (from rasterio.drivers.raster_driver_extensions())
        compress: compression type to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        dtype: rasterio data type string

    Returns:
        (windows, profile): an iterable and a dictionary for the window reads and the raster profile
    """
    with rio.open(raster_paths[template_idx]) as src:
        if windowed:
            windows = [window for _, window in src.block_windows()]
        else:
            windows = [rio.windows.Window(0, 0, src.width, src.height)]

        dst_profile = src.profile.copy()
        dst_profile.update(
            count=count,
            dtype=dtype,
            nodata=nodata,
            compress=compress,
            driver=driver,
        )
        if bigtiff and driver == "GTiff":
            dst_profile.update(BIGTIFF="YES")

    return windows, dst_profile

def check_raster_alignment(raster_paths: list) -> bool:
    """Checks whether the extent, resolution and projection of multiple rasters match exactly.

    Args:
        raster_paths: a list of raster covariate paths

    Returns:
        whether all rasters align
    """
    first = raster_paths[0]
    rest = raster_paths[1:]

    with rio.open(first) as src:
        res = src.res
        bounds = src.bounds
        transform = src.transform

    for path in rest:
        with rio.open(path) as src:
            if src.res != res or src.bounds != bounds or src.transform != transform:
                return False

    return True

def xy_to_geoseries(
    x: Union[float, list, np.ndarray], 
    y: Union[float, list, np.ndarray], 
    crs: CRSType = "epsg:4326"
) -> gpd.GeoSeries:
    """Converts x/y data into a GeoPandas GeoSeries of Points."""
    x = to_iterable(x)
    y = to_iterable(y)

    points = [Point(x, y) for x, y in zip(x, y)]
    return gpd.GeoSeries(points, crs=crs)

def sample_raster(
    raster_path: str, 
    count: int, 
    nodata: float = None, 
    ignore_mask: bool = False
) -> gpd.GeoSeries:
    """Create a random geographic sample of points based on a raster's extent."""
    with rio.open(raster_path) as src:
        if src.nodata is None or ignore_mask:
            if nodata is None:
                xmin, ymin, xmax, ymax = src.bounds
                xy = np.random.uniform((xmin, ymin), (xmax, ymax), (count, 2))
            else:
                data = src.read(1)
                mask = data != nodata
                rows, cols = np.where(mask)
                samples = np.random.randint(0, len(rows), count)
                xy = np.zeros((count, 2))
                for i, sample in enumerate(samples):
                    xy[i] = src.xy(rows[sample], cols[sample])
        else:
            if nodata is None:
                masked = src.read_masks(1)
                rows, cols = np.where(masked == 255)
            else:
                data = src.read(1, masked=True)
                data.mask += data.data == nodata
                rows, cols = np.where(~data.mask)
            samples = np.random.randint(0, len(rows), count)
            xy = np.zeros((count, 2))
            for i, sample in enumerate(samples):
                xy[i] = src.xy(rows[sample], cols[sample])

        return xy_to_geoseries(xy[:, 0], xy[:, 1], crs=src.crs)

def annotate_vector(
    vector_path: str,
    raster_paths: list,
    labels: list = None,
    drop_na: bool = True,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Reads and stores pixel values from rasters using a point-format vector file."""
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    gdf = gpd.read_file(vector_path)
    raster_df = annotate_geoseries(
        gdf.geometry,
        raster_paths,
        labels=labels,
        drop_na=drop_na,
        quiet=quiet
    )

    gdf = pd.concat([gdf, raster_df.drop(columns="geometry", errors="ignore")], axis=1)
    return gdf

def annotate_geoseries(
    points: gpd.GeoSeries,
    raster_paths: list,
    labels: List[str] = None,
    drop_na: bool = True,
    dtype: str = None,
    quiet: bool = False,
) -> (gpd.GeoDataFrame, np.ndarray):
    """Reads and stores pixel values from rasters using point locations."""
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)
    n_rasters = len(raster_paths)

    raster_values = []
    valid_idxs = []
    nodata_flag = False

    for raster_idx, raster_path in enumerate(raster_paths):
        if not quiet:
            print(f"Processing raster {raster_idx + 1} of {n_rasters}")
        with rio.open(raster_path, "r") as src:
            if not crs_match(points.crs, src.crs):
                points = points.to_crs(src.crs)

            if raster_idx == 0 and dtype is None:
                dtype = src.dtypes[0]

            xys = [(point.x, point.y) for point in points]

            n_points = len(points)
            samples_iter = list(src.sample(xys, masked=False))
            if not quiet:
                print(f"Processing {raster_idx + 1} of {n_rasters} rasters, sampling {n_points} points.")
            
            samples = np.array(samples_iter, dtype=dtype)
            raster_values.append(samples)

            if drop_na and src.nodata is not None:
                nodata_flag = True
                valid_idxs.append(samples[:, 0] != src.nodata)

    values = np.concatenate(raster_values, axis=1, dtype=dtype)

    if nodata_flag:
        valid = np.all(valid_idxs, axis=0).reshape(-1, 1)
        values = np.concatenate([values, valid], axis=1, dtype=dtype)
        labels.append("valid")

    gdf = gpd.GeoDataFrame(values, geometry=points.geometry, columns=labels)

    return gdf

def annotate(
    points: Union[str, gpd.GeoSeries, gpd.GeoDataFrame],
    raster_paths: Union[str, list],
    labels: list = None,
    drop_na: bool = True,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Read raster values for each point in a vector and append as new columns."""
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    if isinstance(points, gpd.GeoSeries):
        points = points.reset_index(drop=True)
        gdf = annotate_geoseries(
            points,
            raster_paths,
            labels=labels,
            drop_na=drop_na,
            quiet=quiet,
        )

    elif isinstance(points, (gpd.GeoDataFrame, pd.DataFrame)):
        points = points.reset_index(drop=True)
        gdf = annotate_geoseries(
            points.geometry,
            raster_paths,
            labels=labels,
            drop_na=drop_na,
            quiet=quiet,
        )
        gdf = pd.concat([points, gdf.drop(columns="geometry", errors="ignore")], axis=1)

    elif isinstance(points, str) and os.path.isfile(points):
        gdf = annotate_vector(points, raster_paths, labels=labels, drop_na=drop_na, quiet=quiet)

    else:
        raise TypeError("points arg must be a valid path, GeoDataFrame, or GeoSeries")

    if drop_na:
        try:
            valid = gdf["valid"] == 1
            gdf = gdf[valid].drop(columns="valid").dropna().reset_index(drop=True)
        except KeyError:
            pass

    return gdf

def render_raster(layer, band, spectrum):
    prov = layer.dataProvider()
    src_ds = gdal.Open(layer.source())
    src_band = src_ds.GetRasterBand(band)

    if src_band.GetMinimum() is None or src_band.GetMaximum() is None:
        src_band.ComputeStatistics(0)
    band_min = src_band.GetMinimum()
    band_max = src_band.GetMaximum()

    fcn = QgsColorRampShader()
    fcn.setColorRampType(QgsColorRampShader.Interpolated)

    item_list = [
        QgsColorRampShader.ColorRampItem(
            band_min + (n / (len(spectrum) - 1)) * (band_max - band_min),
            QColor(color), lbl=f"{band_min + (n / (len(spectrum) - 1)) * (band_max - band_min):.2f}"
        )
        for n, color in enumerate(spectrum)
    ]

    fcn.setColorRampItemList(item_list)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(fcn)

    renderer = QgsSingleBandPseudoColorRenderer(prov, band, shader)
    renderer.setClassificationMin(band_min)
    renderer.setClassificationMax(band_max)

    layer.setRenderer(renderer)
    layer.triggerRepaint()

def add_layer_with_rendering(file_path, name, spectrum):
    layer = QgsRasterLayer(file_path, name)
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        render_raster(layer, 1, spectrum)
    else:
        print(f"{name} layer is not valid!")

def create_potential_distribution_map(habitat_suitability_path, output_path):
    """
    Creates a potential distribution map by binarizing habitat suitability values.
    Values greater than 0.5 are set to 1 (suitable habitat), and values less than or equal to 0.5 are set to 0 (unsuitable habitat).
    Only a single band (0 and 1 values) is created.

    Args:
        habitat_suitability_path (str): Path to the input habitat suitability raster file.
        output_path (str): Path to the output potential distribution raster file.
    """
    with rio.open(habitat_suitability_path) as src:
        # Read the habitat suitability values (assuming single band)
        habitat_suitability = src.read(1)

        # Binarize the values: 1 if > 0.5, 0 if <= 0.5
        potential_distribution = np.where(habitat_suitability > 0.5, 1, 0)

        # Copy the profile (metadata) from the original raster
        profile = src.profile
        profile.update(count=1, dtype='uint8')  # Single band (binary) output, using uint8

        # Remove NoData value handling, or set it to None
        profile.pop('nodata', None)  # This removes any nodata value that may cause issues

        # Write the new raster (potential distribution map) to the output file
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(potential_distribution, 1)  # Write the binary map to the first band

    print(f"Potential distribution map saved as {output_path}")
    
from qgis.core import QgsRasterLayer, QgsColorRampShader, QgsSingleBandPseudoColorRenderer, QgsRasterShader
from PyQt5.QtGui import QColor
import rasterio as rio

def add_potential_distribution_layer(file_path, name):
    """
    Adds a potential distribution raster layer with the following coloring scheme:
    - 0 (unsuitable habitat): Transparent
    - 1 (suitable habitat): Green

    Args:
        file_path (str): Path to the input raster file (habitat suitability).
        name (str): Name for the new layer.
    """
    # Create the layer from the given file path
    layer = QgsRasterLayer(file_path, name)
    
    if not layer.isValid():
        print(f"{name} layer is not valid!")
        return
    
    # Add the layer to the map
    QgsProject.instance().addMapLayer(layer)

    # Get the data provider and band information
    prov = layer.dataProvider()
    band = 1  # Assuming we are dealing with the first band

    # Set up the color ramp shader
    fcn = QgsColorRampShader()
    fcn.setColorRampType(QgsColorRampShader.Exact)

    # Define the color items for 0 and 1
    item_list = [
        QgsColorRampShader.ColorRampItem(0, QColor(0, 0, 0, 0), "No Habitat (0)"),  # Transparent for 0
        QgsColorRampShader.ColorRampItem(1, QColor(0, 255, 0), "Suitable Habitat (1)")  # Green for 1
    ]

    fcn.setColorRampItemList(item_list)

    # Apply the shader to the renderer
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(fcn)

    # Create the renderer and set classification min and max
    renderer = QgsSingleBandPseudoColorRenderer(prov, band, shader)
    renderer.setClassificationMin(0)
    renderer.setClassificationMax(1)

    # Apply renderer to the layer
    layer.setRenderer(renderer)
    layer.triggerRepaint()

    print(f"Potential distribution layer '{name}' added successfully!")



