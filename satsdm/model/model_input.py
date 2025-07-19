from qgis.core import QgsVectorLayer, QgsSymbol, QgsSingleSymbolRenderer, QgsProject
from qgis.PyQt.QtGui import QColor

def apply_point_layer_style(layer, color, opacity=1.0, size=2.0):
    """
    Apply a customizable style to the point layer.
    """
    if not layer.isValid():
        print("Invalid layer provided to style.")
        return

    geometry_type = layer.geometryType()
    if geometry_type == 0:  # Point
        symbol = QgsSymbol.defaultSymbol(geometry_type)
        if symbol:
            symbol.setColor(QColor(color))
            symbol.setOpacity(opacity)
            symbol.setSize(size)
            renderer = QgsSingleSymbolRenderer(symbol)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            print(f"Style applied to point layer '{layer.name()}'")
        else:
            print("Failed to create default point symbol.")
    else:
        print("This function is currently for point layers only.")
