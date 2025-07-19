def classFactory(iface):
    from .satsdm import SatSDM
    return SatSDM(iface)
