import os
import requests
import tempfile
import uuid
import geopandas as gpd

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog, QFileDialog
from qgis.core import Qgis, QgsVectorLayer, QgsRasterLayer, QgsProject, QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer
from qgis.utils import iface
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox, QComboBox, QSpinBox

from sklearn import metrics
from osgeo import gdal

from satsdm.model.model_input import apply_point_layer_style
from satsdm.model.sd_layers import identify_satellite_type, update_index_checkboxes, run_selected_indices, run_terrain_indices, run_proximity_layer
from satsdm.model.model_report import (
    generate_evaluation_summary,
    save_roc_curve,
    get_jackknife_report
)

from satsdm.model.core import stack_geodataframes, MaxentModel, MaxentConfig, apply_model_to_rasters
from satsdm.model.utils import (
    sample_raster,
    annotate, save_object,
    add_layer_with_rendering,
    create_potential_distribution_map,
    add_potential_distribution_layer
)

FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), "satsdm_dialog.ui"))

class SatSDMDialog(QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setMinimumWidth(500)

        self.browse_occurrence_button.clicked.connect(self.load_occurrence_from_file)
        self.gbif_download_button.clicked.connect(self.download_from_gbif)
        self.add_env_button.clicked.connect(self.add_env_layer)
        self.remove_env_button.clicked.connect(self.remove_env_layer)

        self.env_layers = []
        self.run_button.clicked.connect(self.run_maxent_model)
        self.browse_output.clicked.connect(self.select_output_folder)
        
        # SD Layers
        self.satellite_folder_button.clicked.connect(self.select_satellite_folder)
        self.satellite_run_button.clicked.connect(self.on_satellite_run_clicked)

        self.terrain_folder_button.clicked.connect(self.select_dem_file)
        self.terrain_run_button.clicked.connect(self.on_terrain_run_clicked)

        self.proximity_folder_button.clicked.connect(self.select_proximity_vector)
        self.combo_proximity_field.currentIndexChanged.connect(self.update_proximity_values)
        self.proximity_run_button.clicked.connect(self.on_proximity_run_clicked)

    def select_satellite_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Satellite Folder")
        if not folder:
            return

        self.satellite_folder_input.setText(folder)

        satellite = identify_satellite_type(folder)
        if satellite:
            self.satellite_combo.setCurrentText(satellite)
            self.satellite_combo.setEnabled(False)
            update_index_checkboxes(self, satellite)

    def on_satellite_run_clicked(self):
        
        folder = self.satellite_folder_input.text().strip()
        satellite = self.satellite_combo.currentText()

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        selected_indices = []
        if self.check_ndvi.isChecked(): selected_indices.append("NDVI")
        if self.check_gndvi.isChecked(): selected_indices.append("GNDVI")
        if self.check_ndwi_mcfeeters.isChecked(): selected_indices.append("NDWI_MCFEETERS")
        if self.check_ndwi_gao.isChecked(): selected_indices.append("NDWI_GAO")
        if self.check_ndbi.isChecked(): selected_indices.append("NDBI")
        if self.check_savi.isChecked(): selected_indices.append("SAVI")
        if self.check_bai.isChecked(): selected_indices.append("BAI")

        run_selected_indices(folder, satellite, selected_indices, output_dir)

    def select_dem_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DEM File", "", "TIFF Files (*.tif *.tiff)")
        if file_path:
            self.terrain_folder_input.setText(file_path)

    def on_terrain_run_clicked(self):
        dem_path = self.terrain_folder_input.text()
        if not dem_path or not os.path.isfile(dem_path):
            QMessageBox.warning(self, "Invalid DEM", "Please select a valid DEM (.tif) file.")
            return

        selected = []
        if self.check_slope.isChecked():
            selected.append("SLOPE")
        if self.check_aspect.isChecked():
            selected.append("ASPECT")
        if self.check_tpi.isChecked():
            selected.append("TPI")

        if not selected:
            QMessageBox.information(self, "No Selection", "Please select at least one terrain index.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        run_terrain_indices(dem_path, selected, output_dir)

    def select_proximity_vector(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Land Cover Vector", "", "Vector Files (*.shp *.gpkg)"
        )
        if not file_path:
            return

        self.proximity_folder_input.setText(file_path)
        layer = QgsVectorLayer(file_path, "LandCover", "ogr")

        if not layer.isValid():
            QMessageBox.warning(self, "Invalid File", "Selected file is not a valid vector layer.")
            return

        # Save reference for later
        self.current_proximity_layer = layer

        # Fill field combo box
        self.combo_proximity_field.clear()
        for field in layer.fields():
            if field.typeName().lower() in ["string", "int", "integer"]:
                self.combo_proximity_field.addItem(field.name())

    def update_proximity_values(self):
        layer = getattr(self, 'current_proximity_layer', None)
        if not layer:
            return

        field_name = self.combo_proximity_field.currentText()
        if not field_name:
            return

        unique_values = set()
        for feature in layer.getFeatures():
            val = feature[field_name]
            if val is not None:
                unique_values.add(str(val))

        self.combo_proximity_value.clear()
        self.combo_proximity_value.addItems(sorted(unique_values))

    def on_proximity_run_clicked(self):
        vector_path = self.proximity_folder_input.text()
        field_name = self.combo_proximity_field.currentText()
        field_value = self.combo_proximity_value.currentText()

        if not (vector_path and field_name and field_value):
            QMessageBox.warning(self, "Missing Input", "Please select vector, field, and value.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        run_proximity_layer(vector_path, field_name, field_value, output_dir)

    def load_occurrence_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select GPKG file", "", "GeoPackage (*.gpkg)")
        if file_path:
            self.occurrence_file_input.setText(file_path)
            layer = QgsVectorLayer(file_path, "Presence points", "ogr")
            if layer.isValid():
                qgis_layer = iface.addVectorLayer(file_path, "Presence points", "ogr")
                apply_point_layer_style(qgis_layer, color="red")

    def download_from_gbif(self):
        # Clear the output and start the GBIF data request
        self.gbif_result_output.clear()
        self.gbif_result_output.append("Starting GBIF data request...")

        # Get species and country code input from the user
        species = self.gbif_species_input.text().strip()
        country = self.gbif_country_input.text().strip().upper()

        if not species or not country:
            self.gbif_result_output.setPlainText("Species and country code must be provided.")
            return

        # Construct the URL for the GBIF API request
        url = f"https://api.gbif.org/v1/occurrence/search?scientificName={species}&country={country}&hasCoordinate=true&basisOfRecord=HUMAN_OBSERVATION&limit=10000"
        self.gbif_result_output.append(f"Requesting: {url}")
        resp = requests.get(url)
        self.gbif_result_output.append("Response received.")
        
        # Parse the JSON response
        data = resp.json()
        features = []

        # Extract latitude and longitude from the results
        for record in data.get("results", []):
            if "decimalLatitude" in record and "decimalLongitude" in record:
                lat, lon = record["decimalLatitude"], record["decimalLongitude"]
                features.append((lon, lat))

        if not features:
            self.gbif_result_output.setPlainText("No coordinates found for the given inputs.")
            return

        # Allow the user to select an output directory where the GPKG will be saved
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            self.gbif_result_output.setPlainText("No output directory selected. Operation cancelled.")
            return

        # Construct the full path for the output GeoPackage file
        output_file_path = os.path.join(output_dir, "gbif_presence_points.gpkg")

        # Create a GeoDataFrame from the features, with CRS set to EPSG:4326
        gdf = gpd.GeoDataFrame(features, columns=["lon", "lat"], geometry=gpd.points_from_xy([f[0] for f in features], [f[1] for f in features]), crs="EPSG:4326")

        # Save the GeoDataFrame to the GeoPackage (.gpkg) format
        try:
            gdf.to_file(output_file_path, driver="GPKG")
        except Exception as e:
            self.gbif_result_output.setPlainText(f"Failed to save the GeoPackage: {str(e)}")
            return

        # Update the input field with the path to the GeoPackage
        self.occurrence_file_input.setText(output_file_path)

        # Load the GeoPackage into QGIS with the specified layer name
        layer = QgsVectorLayer(output_file_path, "GBIF presence points", "ogr")  # Set the layer name here
        if layer.isValid():
            qgis_layer = iface.addVectorLayer(layer.source(), "GBIF presence points", "ogr")
            apply_point_layer_style(qgis_layer, color="red")
            self.gbif_result_output.append("Layer added to QGIS.")
        else:
            self.gbif_result_output.append("Failed to load the GeoPackage into QGIS.")

        self.gbif_result_output.append(f"Loaded {len(features)} points from GBIF.")

    def add_env_layer(self):
        # Allow user to select multiple raster files
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Raster Files", "", "Raster files (*.tif)")
        for file_path in file_paths:
            if file_path:
                self.env_layers_list.addItem(file_path)
                self.env_layers.append(file_path)

    def remove_env_layer(self):
        # Remove selected items from the list widget and env_layers list
        selected_items = self.env_layers_list.selectedItems()
        for item in selected_items:
            row = self.env_layers_list.row(item)
            self.env_layers_list.takeItem(row)
            if item.text() in self.env_layers:
                self.env_layers.remove(item.text())

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir_input.setText(folder)

    def run_maxent_model(self):

        # Get presence file path
        presence_path = self.occurrence_file_input.text().strip()

        # Get list of environmental layer paths
        rasters = [self.env_layers_list.item(i).text() for i in range(self.env_layers_list.count())]

        # Check if presence or raster data is missing
        if not presence_path or not rasters:
            return

        # Get output directory path
        output_dir = self.output_dir_input.text().strip()

        # Check if output directory is valid
        if not output_dir or not os.path.exists(output_dir):
            return

        # Load presence points from GeoPackage
        presence = gpd.read_file(presence_path)

        # Sample background points
        background = sample_raster(rasters[0], count=10000)
        background_path = os.path.join(output_dir, "background_points.gpkg")

        # Save background points to GPKG
        try:
            background.to_file(background_path, driver="GPKG")
        except Exception as e:
            return

        # Load the saved background layer into QGIS
        layer = QgsVectorLayer(background_path, "Background points", "ogr")
        if layer.isValid():
            qgis_layer = iface.addVectorLayer(background_path, "Background points", "ogr")
            apply_point_layer_style(qgis_layer, color="black", size=1.0)
        else:
            return

        # Merge presence and background points, and label the classes
        merged = stack_geodataframes(presence, background, add_class_label=True)

        # Annotate covariates from raster values
        labels = [os.path.splitext(os.path.basename(p))[0] for p in rasters]
        annotated = annotate(merged, rasters, labels=labels, drop_na=True, quiet=True)
        annotated_path = os.path.join(output_dir, "training_dataset.gpkg")

        # Save the annotated training dataset to GPKG
        try:
            annotated.to_file(annotated_path, driver="GPKG")
        except Exception as e:
            return
        
        # Split x/y data
        x = annotated.drop(columns=['class', 'geometry'])
        y = annotated['class']

        # Feature Types
        if self.feature_linear.isChecked():
            MaxentConfig.feature_types.append("linear")
        if self.feature_hinge.isChecked():
            MaxentConfig.feature_types.append("hinge")
        if self.feature_product.isChecked():
            MaxentConfig.feature_types.append("product")
        if self.feature_threshold.isChecked():
            MaxentConfig.feature_types.append("threshold")

        # Regularization Parameters
        MaxentConfig.beta_multiplier = self.beta_multiplier.value()
        MaxentConfig.beta_hinge = self.beta_hinge.value()
        MaxentConfig.beta_lqp = self.beta_lqp.value()
        MaxentConfig.beta_threshold = self.beta_threshold.value()
        MaxentConfig.beta_categorical = self.beta_categorical.value()

        # Advanced Options
        MaxentConfig.clamp = self.clamp_checkbox.isChecked()
        MaxentConfig.tau = self.tau_spinbox.value()
        MaxentConfig.transform = self.transform_combo.currentText()
        MaxentConfig.tolerance = self.tolerance_spinbox.value()
        MaxentConfig.use_lambdas = self.lambda_combo.currentText()
        MaxentConfig.n_lambdas = self.lambda_spinbox.value()
        MaxentConfig.class_weights = self.class_weights_spinbox.value()

        # Train Maxent model using training data
        model = MaxentModel(
            feature_types = MaxentConfig.feature_types,
            
            beta_multiplier = MaxentConfig.beta_multiplier,
            beta_hinge = MaxentConfig.beta_hinge,
            beta_lqp = MaxentConfig.beta_lqp,
            beta_threshold = MaxentConfig.beta_threshold,
            beta_categorical = MaxentConfig.beta_categorical,

            clamp = MaxentConfig.clamp,
            tau = MaxentConfig.tau,
            transform = MaxentConfig.transform,
            convergence_tolerance = MaxentConfig.tolerance,
            use_lambdas = MaxentConfig.use_lambdas,    
            n_lambdas = MaxentConfig.n_lambdas,
            class_weights = MaxentConfig.class_weights,
        )
        model.fit(x, y)        

        # Predict probability scores
        y_pred_prob = model.predict(x)
        
        # Generate evaluation summary
        summary = generate_evaluation_summary(y, y_pred_prob, threshold=0.5)

        # Jackknife Variable Importance
        jackknife_text = get_jackknife_report(model, x, y)
        self.results_summary.setPlainText(summary + "\n" + jackknife_text)

        # Save ROC Curve as PNG image
        roc_path = os.path.join(output_dir, "roc_curve.png")
        save_roc_curve(y, y_pred_prob, roc_path)

        # Display ROC Curve image in QLabel
        pixmap = QPixmap(roc_path)
        if not pixmap.isNull():
            self.roc_image.setPixmap(pixmap.scaledToWidth(400, Qt.SmoothTransformation))
        else:
            return

        # Save the trained Maxent model
        try:
            model_path = os.path.join(output_dir, "trained_model.satsdm")
            save_object(model, model_path)
        except Exception as e:
            return

        # Save the habitat_suitability to GeoTIFF
        habitat_suitability_path = os.path.join(output_dir, "habitat_suitability.tif")
        sdm_color_list = [
            '#440154',  # Viridis_1 (RGB: 68, 1, 84)
            '#48186a',  # Viridis_2 (RGB: 72, 40, 120)
            '#433d84',  # Viridis_3 (RGB: 62, 74, 137)
            '#38598c',  # Viridis_4 (RGB: 49, 104, 142)
            '#2d708e',  # Viridis_5 (RGB: 38, 130, 142)
            '#25858e',  # Viridis_6 (RGB: 31, 158, 137)
            '#1e9b8a',  # Viridis_7 (RGB: 53, 183, 121)
            '#6ccd5a',  # Viridis_8 (RGB: 109, 205, 89)
            '#b8de29',  # Viridis_9 (RGB: 180, 222, 44)
            '#fde725'   # Viridis_10 (RGB: 253, 231, 37)
        ]
        apply_model_to_rasters(model, rasters, habitat_suitability_path, quiet=True)
        add_layer_with_rendering(habitat_suitability_path, "Habitat Suitability", sdm_color_list)

        # Create the potential distribution map (calling the function from utils.py)
        potential_distribution_path = os.path.join(output_dir, "potential_distribution.tif")
        create_potential_distribution_map(habitat_suitability_path, potential_distribution_path)
        add_potential_distribution_layer(potential_distribution_path, "Potential Distribution")
