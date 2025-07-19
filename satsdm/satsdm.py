import os
import sys
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from .satsdm_dialog import SatSDMDialog

class SatSDM:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None
        self.icon_path = os.path.join(os.path.dirname(__file__), 'logo.png')

    def initGui(self):
        # Initialize the GUI
        icon = QIcon(self.icon_path)
        self.action = QAction(icon, "SatSDM", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("&SatSDM", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        # Remove the plugin from the menu and toolbar
        self.iface.removePluginMenu("&SatSDM", self.action)
        self.iface.removeToolBarIcon(self.action)

    def run(self):
        # Show the plugin dialog
        if not self.dialog:
            self.dialog = SatSDMDialog()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
