"""
===================
Lasso Selector
===================

Interactively selecting data points with the lasso tool.

This examples plots a scatter plot. You can then select a few points by drawing
a lasso loop around the points on the graph. To draw, just click
on the graph, hold, and drag it around the points you need to select.
"""


import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        #self.collection = collection

        self.alpha_other = alpha_other

        #self.xys = collection.get_offsets()
        #self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        #self.fc = collection.get_facecolors()
        #if len(self.fc) == 0:
        #    raise ValueError('Collection must have a facecolor')
        #elif len(self.fc) == 1:
        #    self.fc = np.tile(self.fc, (self.Npts, 1))

        lineprops ={'color':'red', 'alpha':0.5}
        self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.select = path.vertices
        #self.ind = np.nonzero(path.contains_points(self.xys))[0]
        #self.fc[:, -1] = self.alpha_other
        #self.fc[self.ind, -1] = 1
        #self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


    def disconnect(self):
        self.lasso.disconnect_events()
        #self.fc[:, -1] = 1
        #self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


