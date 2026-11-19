from .base_anatomy import BaseAnatomy

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display


class MGE(BaseAnatomy):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        super().__init__(path_or_BrukerExp)

        self.TEs = self.get_TEs()
        self.Nechoes = self.get_Nechoes()
        self.Nslices = self.get_Nslices()

    def get_TEs(self):
        return self.method["EffectiveTE"]

    def get_Nechoes(self) -> int:
        return self.method["PVM_NEchoImages"]

    def get_Nslices(self) -> int:
        """Get number of slices from both 2D-multislice and 3D acquisitions."""
        SpatialDim = self.method["PVM_SpatDimEnum"]

        if SpatialDim == "<2D>":
            out = self.method["PVM_SPackArrNSlices"]
        elif SpatialDim == "<3D>":
            out = self.method["PVM_Matrix"][-1]
        else:
            raise NotImplementedError(
                f"PVM_SpatDimEnum has an unknown value: {SpatialDim}."
            )

        return out

    def plot(self, ax=None, parameter_dict=None, display_ui=True):
        """A helper function for an interactive image plot.

        Recommended usage:
            import matplotlib.pyplot as plt
            from hypermri.sequences import MGE
            from IPython.display import display

            # create  MGE object
            mge1 = MGE(...)

            # create figure and axes objects to plot into
            fig, ax = plt.subplots()

            # plot mge data into given axes object, provide custom parameter
            plot_params = {'cmap': 'viridis'}
            mge1.plot(ax, plot_params)

            # note: you can still add stuff to the axes object afterwards
            ax.set_title("MGE Plot")

            # render plot
            plt.show()

        Parameters
        ----------
        ax : Axes, optional
            The axes to draw to. Recommended to generate using the command:
            'fig, ax = plt.subplots()'. If None is specified the axes object
            is created internally.

        param_dict : dict, optional
            Dictionary of keyword arguments to pass to ax.imshow.

        display_ui: bool, optional
            Default is True. Directly calls IPython.display on the interactive
            sliders and renders them. If False, returns the ui object instead of
            displaying it.

        Returns
        ----------
        ui : Hbox, optional
            The ipywidgets HBox containing the interactive sliders is returned
            only if the option 'display_ui' is set to False.
        """
        if ax == None:
            fig, ax = plt.subplots()

        # use default paramter dictionary if none is provided
        if not parameter_dict:
            parameter_dict = {"cmap": "gray"}

        # define image plotting function
        def plot_img(slice_nr, echo_nr):
            return ax.imshow(self.seq2d[..., slice_nr, echo_nr], **parameter_dict)

        # create interactive slider for  slices
        slice_slider = widgets.IntSlider(
            value=self.Nslices // 2, min=0, max=self.Nslices - 1, description="Slice: "
        )

        # create interactive slider for echoes
        echo_slider = widgets.IntSlider(
            value=0, min=0, max=self.Nechoes - 1, description="Echo: "
        )

        # put both sliders inside a HBox for nice alignment  etc.
        ui = widgets.HBox(
            [slice_slider, echo_slider],
            layout=widgets.Layout(display="flex"),
        )

        # connect plotting function with sliders
        # Note: The output is unused here because it is sufficient to display
        #       the HBox (stored in ui variable). Without the ui display you
        #       would have to call the display() function on the output of
        #       widgets.interactive_output().
        sliders = widgets.interactive_output(
            plot_img, {"slice_nr": slice_slider, "echo_nr": echo_slider}
        )

        # This displays the Hbox containing the slider widgets.
        if display_ui:
            display(ui, sliders)
        else:
            return (ui, sliders)
