from ..MGE import MGE
from ...utils import onlycallonce
from ...utils.utils_logging import LOG_MODES, init_default_logger
from ...utils.utils_interpolation import (
    interpolate_3d_array,
    interpolate_masked_3d_array,
    downsample_masked_array,
)
from ...utils.utils_fitting import fit_exponential_T2_relaxation
from ...functions import OMPD5_freq2pH
from ._snr_masking import SNR, createMaskFromThreshold
from ._plot_snr_masking import SNRMaskingPlotter
from ._plot_results2D import UmpirePlotter2D
from ._plot_results3D import UmpirePlotter3D
from ...functions.MEpHISDO import MEpHISDO_Solver
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.optimize import Bounds
from ._plot_results3D import plot_spectrum
from ...utils.utils_anatomical import postprocess_fid_file, combine_multichannel
from ...utils.utils_general import fft

# initialize logger
logger = init_default_logger(__name__)

try:
    from umpire import UMPIRE, phase2frequency
except ModuleNotFoundError:
    # logger.info("UMPIRE package not found.")
    pass

import pickle
import numpy as np
import matplotlib as plt
from skimage.transform import resize


LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468


class umpireMGE(MGE):
    """Analysis class for the umpireMGE sequence.
    """

    def __init__(
        self, path_or_BrukerExp, log_mode="warning", repetition=None, recon_num=-1, external_data=None, self_reco=False, data_scaling_factor = None, remove_zeropadding=False, channel_combination_method = "sum"
    ):
        """Accepts directory path or BrukerExp object as input."""
        super().__init__(path_or_BrukerExp)

        # set up logging level
        logger.setLevel(LOG_MODES[log_mode])


        # save some handy method parameters for easier access
        if self.method["PVM_SpatDimEnum"] == "<2D>":
            logger.warning("2D Acquistion Found! Use at your own risk.")
            slice_thick = self.method["PVM_SliceThick"]
            nslices = self.method["PVM_SPackArrNSlices"]

            fov_rp = self.method["PVM_Fov"]
            self.pvm_fov = [*fov_rp, slice_thick * nslices]

            res_rp = self.method["PVM_SpatResol"]
            self.pvm_res = [*res_rp, slice_thick]

            mat_rp = self.method["PVM_Matrix"]
            self.pvm_matrix = [*mat_rp, nslices]
        else:
            self.pvm_fov = self.method["PVM_Fov"]
            self.pvm_res = self.method["PVM_SpatResol"]
            self.pvm_matrix = self.method["PVM_Matrix"]
        self.pvm_orientation = self.method["PVM_SPackArrSliceOrient"]
        self.pvm_readorientation = self.method["PVM_SPackArrReadOrient"]

        # make sure our echo spacing is equal
        echo_time = self.method["PVM_EchoTime"]
        echo_spacing = self.method["EchoSpacing"]

        if False:#echo_time != echo_spacing:
            raise Warning(
                f"PVM_EchoTime ({echo_time:.2f}ms)!= EchoSpacing ({echo_spacing:.2f}ms)"
            )

        self.repetition = repetition

        # to keep track of different kinds of masks, they get combined in self.mask
        self.mask_dict = {}

        # (try to) load in complex data, order is: Echo Read Phase Slice
        if isinstance(external_data, type(None)):
            self.data = self.load_data(recon_num)

            if self_reco:
                 self.data = self.reconstruct_and_combine_multichannel(data_scaling_factor, remove_zeropadding, channel_combination_method)
        else:
            self.slice_orientations = "coronal", "axial", "sagittal"
            self.data = external_data


        #load k-space 
        self.k_space_data = fft(self.data , axes=(3,2,1), ift=True)
        
        self.rawdatajob1 = self.Load_rawdatajob1_file()

        self._phase = np.angle(self.data)
        self._unwrapped_phase = np.zeros_like(self.data, dtype='float')
        self._magnitude = np.abs(self.data)

        self._fmap = np.zeros_like(self.data[0], dtype='float')
        self._fmap_err = np.zeros_like(self.data[0], dtype='float')
        self._fit_results = np.zeros((*self.data[0].shape, 4), dtype='float')  # m, merr, t, terr
        self._pHmap = np.zeros_like(self.data[0], dtype='float')
        self._multi_comp_fmaps = np.zeros_like(self.data[:1,:,:,:], dtype='float')
        self._multi_comp_amplitudes = np.zeros_like(self.data[:1,:,:,:], dtype='complex') 
        self._multi_comp_pHmaps = np.empty(([3, *self.data[0].shape]), dtype='float')
        self._multi_comp_pHmaps.fill(np.nan)
        self._multi_comp_mean_pH = np.zeros_like(self.data[0], dtype='float')
        self._multi_comp_peak_pH = np.zeros_like(self.data[0], dtype='float')
        self._T2_star_map = np.zeros_like(self.data[0], dtype='float')
        self._fit_cost_map = np.zeros_like(self.data[0], dtype='float')
        self._c1_ampl_map = np.zeros_like(self.data[0], dtype='complex')
        self._f_offset = 0
        self._max_num_comps_MEpHISDO = 0
        self.useNechoes = slice(0, len(self.TEs), None)


        # bool value to keep track if we interpolated the data
        self.interpolated = False

        self._interp_phase = None
        self._interp_unwrapped_phase = None
        self._interp_magnitude = None

        # potentially used for snr scaling of the magnitude
        self._noise_stddev = None

    @property
    def mask(self):
        """Returns a combination of all masks from 'mask_dict' as a single mask."""
        if self.mask_dict:
            return np.logical_or.reduce(list(self.mask_dict.values()))
        return np.zeros_like(self._fmap)

    @property
    def fmap(self):
        return np.ma.masked_array(data=self._fmap, mask=self.mask)

    @property
    def fmap_err(self):
        return np.ma.masked_array(data=self._fmap_err, mask=self.mask)

    @property
    def fmap_err_rel(self):
        """https://qastack.com.de/stats/86708"""
        out = np.abs(self._fmap_err / self._fmap) * 100
        return np.ma.masked_array(data=out, mask=self.mask)

    @property
    def magnitude(self):
        mag = self._interp_magnitude if self.interpolated else self._magnitude

        mask = np.repeat(self.mask[np.newaxis], len(self.TEs), axis=0)

        std = self._noise_stddev or 1

        return np.ma.masked_array(data=mag / std, mask=mask)

    @property
    def phase(self):
        phase = self._interp_phase if self.interpolated else self._phase

        mask = np.repeat(self.mask[np.newaxis], len(self.TEs), axis=0)

        return np.ma.masked_array(data=phase, mask=mask)

    @property
    def unwrapped_phase(self):
        if self.interpolated:
            return self._interp_unwrapped_phase
        return self._unwrapped_phase

    @property
    def pHmap(self):
        out = np.ma.masked_array(data=self._pHmap, mask=self.mask)
        if np.isnan(out).all():
            logger.error("Masking of pH-Map aborted, all entries would be NaN.")
        return out
    
    @property
    def T2_star(self):
        out = np.ma.masked_array(data=self._T2_star_fit_results[0].squeeze(), mask=self.mask)
        if np.isnan(out).all():
            logger.error("Masking of T2*-Map aborted, all entries would be NaN.")
        return out
    
    @property
    def spectral_data(self):
        
        if len(self.rawdatajob1) == 0:
            raise Exception("No RawDataJob1 file found. Make sure you acquired additional spectra!")
        else:
            acq_points = int(self.method["NSpecPoints"])
            num_spectra = int(len(self.rawdatajob1)/acq_points)
            
            complex_spec = np.zeros((num_spectra, acq_points), dtype=complex)

            ffts = lambda x: np.fft.fftshift(np.fft.fft(x))

            for idx in range(num_spectra):
                complex_spec[idx, :] = ffts(self.rawdatajob1[idx * acq_points : acq_points + idx * acq_points])

            # calculate frequency axis
            d = 1/self.method["PVM_DigSw"]
            frequ_axis = np.fft.fftshift(np.fft.fftfreq(acq_points, d=d))

            return complex_spec, frequ_axis

    @property
    def multi_comp_pHmaps(self):
        mask = np.repeat(self.mask[np.newaxis], self._multi_comp_pHmaps.shape[0], axis=0)
        out = np.ma.masked_array(data=self._multi_comp_pHmaps, mask=mask)
        if np.isnan(out).all():
            logger.error("Masking of multi-comp pH-Map aborted, all entries would be NaN.")
        return out

    @property
    def multi_comp_fmaps(self):
        mask = np.repeat(self.mask[np.newaxis], self._multi_comp_fmaps.shape[0], axis=0)
        out = np.ma.masked_array(data=self._multi_comp_fmaps, mask=mask)
        if np.isnan(out).all():
            logger.error("Masking of multi-comp frequency-Map aborted, all entries would be NaN.")
        return out

    @property
    def multi_mean_pHmap(self):
        out = np.ma.masked_array(data=self._multi_comp_mean_pH, mask=self.mask)
        if np.isnan(out).all():
            logger.error("Masking of pH-Map aborted, all entries would be NaN.")
        return out

    @property
    def multi_comp_peak_pH(self):
        out = np.ma.masked_array(data=self._multi_comp_peak_pH, mask=self.mask)
        if np.isnan(out).all():
            logger.error("Masking of pH-Map aborted, all entries would be NaN.")
        return out


    def _useNechoes_input_handler(self, usr_input, verbose=False):
        vprint = print if verbose else lambda *a, **k: None

        if usr_input == False:
            vprint(f"Using all N={self.Nechoes} echoes.")
            usr_input = slice(None)
        elif isinstance(usr_input, int):
            usr_input = slice(0, usr_input, None)
            vprint(f"Using N={np.arange(0, self.Nechoes, 1)[usr_input]} echoes.")
        else:
            vprint(f"Using N={np.arange(0, self.Nechoes, 1)[usr_input]} echoes.")

        return usr_input
    
    
    def load_data(self, default_recon_num=-1):
        """Load complex 2dseq reconstruction.

        WARNING: Assumes the last recon is complex! Could be dangerous,
                 bruker2complex throws an error if dimension != 2 though...
        """
        complex_recon_name = self.recon_names[default_recon_num]

        # load data
        data = self.Load_2dseq_file(recon_num=complex_recon_name)

        assert "complex" in str(data.dtype), "No complex data found!"
        # convert into complex dtype - not necessary anymore see brukerapi@luca
        # data = bruker2complex(data)

        if self.repetition is not None:
            print(f"Found {data.shape[-1]} repetitions. Using idx={self.repetition}.")
            data = data[..., self.repetition]

        if self.method["PVM_SpatDimEnum"] == "<2D>":
            # only works for sag HF!
            data = np.moveaxis(data, 0, -1)
            data = np.moveaxis(data, 0, 1)
            data = np.moveaxis(data, -2, -1)
            self.slice_orientations = "coronal", "axial", "sagittal"
            return data

        # ensure the following axis order: read, phase, slice, echo
        data = self.ensure_RPS_order(data)

        # swap echo axis from last to first (bruker puts the echo dimension at
        # the last axis per default)
        data = np.moveaxis(data, -1, 0)

        return data

    def ensure_RPS_order(self, data):
        """Swap data axis to ensure read, phase, slice order."""
        assert (
            self.acqp["ACQ_patient_pos"] == "Head_Prone"
        ), f"Found {self.acqp['ACQ_patient_pos']} instead of 'Head_Prone'."

        if self.Nslices > 1:
            # Reorient the array, make read always first!
            if self.pvm_orientation == "axial":
                if self.pvm_readorientation == "L_R":
                    self.slice_orientations = "axial", "sagittal", "coronal"
                    return data
                if self.pvm_readorientation in ("V_D", "A_P"):
                    self.slice_orientations = "axial", "coronal", "sagittal"
                    return np.moveaxis(data, 0, 1)

            if self.pvm_orientation == "sagittal":
                if self.pvm_readorientation in ("V_D", "A_P"):
                    self.slice_orientations = "sagittal", "coronal", "axial"
                    return data
                if self.pvm_readorientation == "H_F":
                    self.slice_orientations = "sagittal", "axial", "coronal"
                    return np.moveaxis(data, 0, 1)

            if self.pvm_orientation == "coronal":
                if self.pvm_readorientation == "L_R":
                    self.slice_orientations = "coronal", "sagittal", "axial"
                    return data
                if self.pvm_readorientation == "H_F":
                    self.slice_orientations = "coronal", "axial", "sagittal"
                    return np.moveaxis(data, 0, 1)

        else:
            if self.pvm_orientation == "axial":
                self.slice_orientations = ("axial",)
                if self.pvm_readorientation == "L_R":
                    return data
                if self.pvm_readorientation == "A_P":
                    return np.moveaxis(data, 0, 1)

            if self.pvm_orientation == "sagittal":
                self.slice_orientations = ("sagittal",)
                if self.pvm_readorientation == "A_P":
                    return data
                if self.pvm_readorientation == "H_F":
                    return np.moveaxis(data, 0, 1)

            if self.pvm_orientation == "coronal":
                self.slice_orientations = ("coronal",)
                if self.pvm_readorientation == "L_R":
                    return data
                if self.pvm_readorientation == "H_F":
                    return np.moveaxis(data, 0, 1)

        raise NotImplementedError(
            f"Got unknown orientation: '{self.pvm_orientation}'"
            + f"and '{self.pvm_readorientation}'."
        )

    def umpire(
        self,
        useNechoes=False,
        DPD_filter_func="default",
        magnitude_weighted_omega_star=False,
        noreturn=True,
    ):
        # handle what echoes to use
        self.useNechoes = self._useNechoes_input_handler(useNechoes, verbose=True)

        out = UMPIRE(
            self.data[self.useNechoes],
            self.TEs[self.useNechoes],
            DPD_filter_func=DPD_filter_func,
            magnitude_weighted_omega_star=magnitude_weighted_omega_star,
        )

        # store results
        self._unwrapped_phase = out

        if not noreturn:
            return out
    

    def MEpHISDO(
        self,
        max_num_compartments = 3,
        frequ_inits = np.array([250.0, 200.0, 100.0]),
        frequ_boundaries = Bounds([40,40,40],[ 400,400,400]),
        amplitude_inits = np.array([40+1j, 20+1j, 10+1j]),
        amplitude_boundaries = Bounds([-200,-200,-200,-200,-200,-200],[ 200,200,200,200,200,200]),
        t2_star_init = 0.013,
        t2_star_boundary =  Bounds([0.007],[0.1]),
        threshold_frequency_proximity = 15,
        c1_contrib = -np.inf,
        noreturn=True,
        use_echoes = None,
        f_search_range = 20,
        snr_threshold=15,
        **least_squares_kwargs

    ):
        
        MEpHISDO_solver = MEpHISDO_Solver(
            max_num_compartments,
            uMGE_object=self,
            use_echoes = use_echoes,
        )

        out = MEpHISDO_solver.solve(
            frequ_inits = frequ_inits,
            frequ_boundaries = frequ_boundaries,
            amplitude_inits = amplitude_inits,
            amplitude_boundaries = amplitude_boundaries,
            t2_star_init = t2_star_init,
            t2_star_boundary = t2_star_boundary,
            threshold_frequency_proximity = threshold_frequency_proximity,
            c1_contrib = c1_contrib,
            f_search_range = f_search_range,
            snr_threshold=snr_threshold,
            noise=self._noise_stddev,
            **least_squares_kwargs
           )
        

        # store results
        self._multi_comp_fmaps = out[:max_num_compartments,:,:,:].real
        self._multi_comp_amplitudes = out[max_num_compartments:2*max_num_compartments,:,:,:]
        

        if c1_contrib != -np.inf:
            self._c1_ampl_map = out[-2,:,:,:]
        
        self._fit_cost_map = out[-1,:,:,:].real#/np.max(np.abs(self.data), axis=0)

        
        self._T2_star_map = out[2*max_num_compartments,:,:,:].real

        self._max_num_comps_MEpHISDO = max_num_compartments

        if not noreturn:
            return out


    def phase2frequency(
        self, useNechoes=None, load_bar=True, keep_dims=True, return_fit_results=False, njobs=None
    ):
        # check wether all echoes were used during self.umpire()
        if useNechoes is not None:
            if len(self.data) > len(self.unwrapped_phase):
                AssertionError("useNechoes already specified in self.umpire()!")
            else:
                useNechoes = self._useNechoes_input_handler(useNechoes)

                unwrapped_phase = self.unwrapped_phase[useNechoes]
                TEs = self.TEs[useNechoes]
        else:
            unwrapped_phase = self.unwrapped_phase
            TEs = self.TEs

        print(self.unwrapped_phase.shape)

        return phase2frequency(
            unwrapped_phase,
            TEs,
            load_bar=load_bar,
            keep_dims=keep_dims,
            return_fit_results=return_fit_results,
            njobs=njobs,
        )

    def interpolate(self, interp_fac, method="nearest"):
        """
        interp_fac: int, np.ndarray
        """

        def i3d(arr, method=method):
            return interpolate_3d_array(arr, new_dims, method=method)

        # reset masks for interpolation process
        self.mask_dict = {}

        # calculate new dimensions
        old_dims = np.array(self.data.shape[1:])  # skip echo dimension

        # check if we have an int or a shape as interp_fac argument
        if isinstance(interp_fac, int):
            new_dims = old_dims * interp_fac
        elif isinstance(interp_fac, (tuple, list, np.ndarray)):
            new_dims = np.array(interp_fac)
        else:
            raise AssertionError("Not a valid interpolation factor. See the docs.")

        self.interp_matrix = new_dims

        # interpolate magnitude, phase and unwrapped phase:
        interp_magnitude = np.zeros((self.Nechoes, *new_dims))
        interp_phase = np.zeros((self.Nechoes, *new_dims))
        interp_unwrapped_phase = np.zeros((self.Nechoes, *new_dims))

        interp_fmap = np.zeros(new_dims)
        interp_fmap_err = np.zeros(new_dims)

        for i, _ in enumerate(self.TEs[self.useNechoes]):
            interp_phase[i] = i3d(self._phase[i])
            interp_unwrapped_phase[i] = i3d(self._unwrapped_phase[i])
            interp_magnitude[i] = i3d(self._magnitude[i])

        self._interp_phase = interp_phase
        self._interp_unwrapped_phase = interp_unwrapped_phase
        self._interp_magnitude = interp_magnitude

        # reset all other arrays based on old dimensions/information
        # these arrays, are populated by (re-)calling the appropriate methods
        self._fmap = np.zeros(new_dims)
        self._fmap_err = np.zeros(new_dims)
        self._fit_results = np.zeros((*self.pvm_matrix, 4))  # m, merr, t, terr
        self._pHmap = np.zeros(new_dims)
        self._fmap = np.zeros(new_dims)
        self._multi_comp_fmaps = np.zeros((1,*new_dims))
        self._multi_comp_pHmaps = np.zeros((1,*new_dims))
        self._multi_comp_mean_pH = np.zeros(new_dims)
        self._multi_comp_peak_pH = np.zeros(new_dims)

        self.interpolated = True
        print(f"{self.pvm_matrix} --> {new_dims}")

    def interpolate_afterwards(self, interp_fac, method="nearest"):
        def i3d(arr, method=method):
            return interpolate_3d_array(arr, new_dims, method=method)

        # reset masks for interpolation process
        self.mask_dict = {}

        # calculate new dimensions
        old_dims = np.array(self.data.shape[1:])  # skip echo dimension

        # check if we have an int or a shape as interp_fac argument
        if isinstance(interp_fac, int):
            new_dims = old_dims * interp_fac
        elif isinstance(interp_fac, (tuple, list, np.ndarray)):
            new_dims = np.array(interp_fac)
        else:
            raise AssertionError("Not a valid interpolation factor. See the docs.")

        self.interp_matrix = new_dims

        # interpolate magnitude, phase and unwrapped phase:
        interp_magnitude = np.zeros((self.Nechoes, *new_dims))
        interp_phase = np.zeros((self.Nechoes, *new_dims))
        interp_unwrapped_phase = np.zeros((self.Nechoes, *new_dims))

        interp_fmap = np.zeros(new_dims)
        interp_fmap_err = np.zeros(new_dims)

        for i, _ in enumerate(self.TEs[self.useNechoes]):
            interp_phase[i] = i3d(self._phase[i])
            interp_unwrapped_phase[i] = i3d(self._unwrapped_phase[i])
            interp_magnitude[i] = i3d(self._magnitude[i])

        self._interp_phase = interp_phase
        self._interp_unwrapped_phase = interp_unwrapped_phase
        self._interp_magnitude = interp_magnitude

        # interpolate fmap, fmap_err and pHmap:
        self._fmap = i3d(self._fmap)
        self._fmap_err = i3d(self._fmap_err)
        self._pHmap = i3d(self._pHmap)

        #interpolate _multi_comp_fmaps and _multi_comp_pHmaps
        interp_multi_comp_fmaps = np.zeros((self._multi_comp_fmaps.shape[0], *new_dims))
        interp_multi_comp_pHmaps = np.zeros((self._multi_comp_pHmaps.shape[0], *new_dims))
        interp_multi_comp_amplitudes = np.zeros((self._multi_comp_amplitudes.shape[0], *new_dims), dtype='complex')

        for i in range(self._multi_comp_fmaps.shape[0]):
            interp_multi_comp_fmaps[i] = i3d(self._multi_comp_fmaps[i])
            
        for i in range(self._multi_comp_pHmaps.shape[0]):    
            interp_multi_comp_pHmaps[i] = i3d(self._multi_comp_pHmaps[i])

        for i in range(self._multi_comp_amplitudes.shape[0]):
            interp_multi_comp_amplitudes[i] = i3d(self._multi_comp_amplitudes[i])

        self._multi_comp_fmaps = interp_multi_comp_fmaps
        self._multi_comp_pHmaps = interp_multi_comp_pHmaps
        self._multi_comp_amplitudes = interp_multi_comp_amplitudes
        self._multi_comp_mean_pH = i3d(self._multi_comp_mean_pH)
        self._multi_comp_peak_pH = i3d(self._multi_comp_peak_pH)
        self._T2_star_map = i3d(self._T2_star_map)
        self._fit_cost_map =  i3d(self._fit_cost_map)

        # interpolate fit results (not sure this is the correct way to do it):
        interp_fit_results = np.zeros((*new_dims, 4))  # m, merr, t, terr
        for i in range(4):
            interp_fit_results[..., i] = i3d(self._fit_results[..., i])
        self._fit_results = interp_fit_results

        self.interpolated = True

        print(f"{self.pvm_matrix} --> {new_dims}")

    def calculate_fmap(self, useNechoes=None, load_bar=True, njobs=None):
        # generate frequency map from unwrapped phase images
        fmap_out = self.phase2frequency(
            useNechoes=useNechoes,
            load_bar=load_bar,
            return_fit_results=True,
            njobs=njobs,
        )

        self._fmap = fmap_out[0]
        self._fmap_err = fmap_out[1]
        self._fit_results = fmap_out[2]

    def mask_snr(self, echoNr=2, use_k_space = False):
        """Interactive SNR masking interface.

        Parameters to adjust:
            1. 'Thr. SNR': The SNR threshold. Everything below gets masked.
            2. 'Factor': Controlling the area size of the noise region.

        After adjusting the parameters, click "SAVE MASK" to save the mask.

        For silent masking, without any interaction use mask_slient() instead.
        """
        SNRMaskingPlotter(self, echoNr, use_k_space)

    def mask_snr_silent(self, thr=10, factor=0.05, echoNr=0, use_k_space=False, plot = False):
        """SNR masking without any visualization."""
        # magnitude of first echo only

        if not isinstance(self._interp_magnitude, type(None)):
            mag = self._interp_magnitude[echoNr]
        else:
            mag = self._magnitude[echoNr]

        if use_k_space:
            SNR_scaled_mag, _, stddev = SNR(np.squeeze(mag), factor, self.k_space_data, plot = plot)
        else:
            SNR_scaled_mag, _, stddev = SNR(np.squeeze(mag), factor)

        print(f"Std.Dev.: {stddev:.2f}")
        print(f"SNR Min.: {np.min(SNR_scaled_mag):.2f}")
        print(f"SNR Max.: {np.max(SNR_scaled_mag):.2f}")

        mask_mag = createMaskFromThreshold(SNR_scaled_mag, thr)

        self.mask_dict["snr"] = mask_mag
        self._noise_stddev = stddev

    def calculate_pH(self, titrationcurve=None, multi_comp=True, c1_center=None):
        """Calculate pH from frequency map. Assuming C5 peak.

        titrationcurve : optional
            Either None or 'andre'. Default is None, here the titration curve
            parameter from M.Grashei and P.Wodtke published here
            https://doi.org/10.1038/s41467-023-40747-3 were used.
            If 'andre', the new titrationcurve acquired without Urea mixed in, is
            used (see appendix Master Thesis.)
        """
        # [pK, δHA, δA = δHA + δ_range] - Reference: Urea
        PARAMS_OMPD_C5_Urea = 6.5450, 12.4801, 12.4801 + 3.2975
        # [pK, δHA, δA = δHA + δ_range] - Reference: Urea
        PARAMS_OMPD_C5_C1 = 6.5450, 12.4801 - 8.4919, 12.4801 + 3.2975 - 8.0275
        # [pK, δHA, δA] of ZOMPD C5  - Reference: Urea
        PARAMS_ANDRE_OMPD_Urea_C5 = 6.417854, 12.409269, 15.973497

        if titrationcurve in ("andre", "Andre", "new"):
            PARAMS = PARAMS_ANDRE_OMPD_Urea_C5
        elif titrationcurve is not None:
            raise AssertionError("titrationcurve takes None or 'andre' as input.")
        elif c1_center is not None:
            PARAMS = PARAMS_OMPD_C5_C1
        else:
            PARAMS = PARAMS_OMPD_C5_Urea

        f_0 = self.method["PVM_FrqWork"][0]

        def ppm2pH(ppm, params, c1_center):
            """Henderson-Hasselbach relationship to convert chemical shift to pH."""
            pK, dHA, dA = params
            if c1_center is not None:
                ppm = ppm - (c1_center / f_0)

            log_arg = (ppm - dHA) / (dA - ppm)

            return pK + np.log10(log_arg)

        ppmMap = self._fmap / f_0

        pHmap = ppm2pH(ppmMap, PARAMS, c1_center)

        if np.isnan(pHmap).all():
            logger.error("The pH conversion failed, all entries are NaN.")
        else:
            self._pHmap = pHmap

        if multi_comp and np.count_nonzero(self._multi_comp_fmaps) != 0:
            multi_comp_ppmMap = self._multi_comp_fmaps / f_0

            mutli_comp_pHmap = ppm2pH(multi_comp_ppmMap, PARAMS, c1_center)

            if np.isnan(mutli_comp_pHmap).all():
                logger.error(
                    "The pH conversion for the multi_comp results failed, all entries are NaN."
                )
            else:
                self.sort_pH_by_compartment(mutli_comp_pHmap)

                # calculate mean pH map
                self.calculate_mean_pH_map()

    def calculate_mean_pH_map(self):
        """Amplitude weighted mean pH of all compartments with a valid pH"""
        pH = np.real(self._multi_comp_pHmaps)
        ampl = np.abs(self._multi_comp_amplitudes)
        has_pH = ~np.isnan(pH)

        with np.errstate(invalid="ignore", divide="ignore"):
            weighted_sum = np.sum(np.where(has_pH, pH * ampl, 0), axis=0)
            ampl_sum = np.nansum(np.where(has_pH, ampl, np.nan), axis=0)
            mean_pH = weighted_sum / ampl_sum

        # voxels without any valid compartment
        mean_pH[mean_pH == 0] = np.nan

        self._multi_comp_mean_pH = mean_pH

    def calculate_peak_pH_map(self):
        """pH of the compartment with the largest amplitude"""
        pH = np.real(self._multi_comp_pHmaps)
        ampl = np.abs(self._multi_comp_amplitudes)

        # only consider components with both a valid pH and a valid amplitude
        valid = ~np.isnan(pH) & ~np.isnan(ampl)

        max_index = np.argmax(np.where(valid, ampl, -np.inf), axis=0)
        peak_pH = np.take_along_axis(pH, max_index[np.newaxis], axis=0)[0]
        peak_pH[~valid.any(axis=0)] = np.nan

        self._multi_comp_peak_pH = peak_pH

    def sort_pH_by_compartment(self, pH_map):
        """Sorts the multi compartment fit results into fixed pH compartment slots.

        Components are ordered by ascending pH and assigned to the slot of their pH compartment
        (see pH_compartment_thresholds). If several components fall into the same compartment,
        the colliding ones are moved to the nearest free neighbouring slot while keeping the pH order.
        """
        num_slots = self._max_num_comps_MEpHISDO
        thresholds = self.pH_compartment_thresholds()

    
        pH_values = np.real(pH_map)
        frequencies = np.real(self._multi_comp_fmaps)
        amplitudes = self._multi_comp_amplitudes
        spatial_shape = pH_values.shape[1:]

        # only components with valid pH, frequency and amplitude are sorted
        valid = ~np.isnan(pH_values) & ~np.isnan(frequencies) & ~np.isnan(amplitudes)

        # if more components than slots, keep the ones with the largest amplitudes
        ampl_order = np.argsort(np.where(valid, -np.abs(amplitudes), np.inf), axis=0)
        ampl_rank = np.argsort(ampl_order, axis=0)
        keep = valid & (ampl_rank < num_slots)
        num_valid = keep.sum(axis=0)

        # order by ascending pH, the kept components come first
        pH_order = np.argsort(np.where(keep, pH_values, np.inf), axis=0, kind="stable")

        def take(arr, i):
            return np.take_along_axis(arr, pH_order[i][np.newaxis], axis=0)[0]

        sorted_pH = np.full((num_slots, *spatial_shape), np.nan)
        sorted_freqs = np.full((num_slots, *spatial_shape), np.nan)
        sorted_ampls = np.full((num_slots, *spatial_shape), np.nan, dtype="complex")

        prev_slot = np.full(spatial_shape, -1)
        for i in range(min(num_slots, pH_values.shape[0])):
            active = i < num_valid
            pH_i = take(pH_values, i)

            # preferred slot (0-based) of each component based on its pH compartment
            preferred_slot = np.searchsorted(thresholds, np.where(active, pH_i, 0), side="left")

            # slot has to be above the previous one and leave room for the remaining components
            slot = np.minimum(np.maximum(preferred_slot, prev_slot + 1), num_slots - num_valid + i)
            prev_slot = np.where(active, slot, prev_slot)

            # write the component into its slot (only in voxels where it exists)
            slot = np.where(active, slot, 0)[np.newaxis]
            for out, arr in ((sorted_pH, pH_values), (sorted_freqs, frequencies), (sorted_ampls, amplitudes)):
                current = np.take_along_axis(out, slot, axis=0)[0]
                np.put_along_axis(out, slot, np.where(active, take(arr, i), current)[np.newaxis], axis=0)

        self._multi_comp_pHmaps = sorted_pH
        self._multi_comp_fmaps = sorted_freqs
        self._multi_comp_amplitudes = sorted_ampls

    def pH_compartment_thresholds(self):
        """Returns the upper pH limits of the compartments, depending on the number of fitted compartments."""

        if self._max_num_comps_MEpHISDO == 3:
            return np.array([6.7, 7.15])
        elif self._max_num_comps_MEpHISDO == 2:
            return np.array([7.15])
        else:
            raise ValueError(f"No pH compartment thresholds defined for {self._max_num_comps_MEpHISDO} compartments.")

    def pH_comp_number(self, pH_value):
        """Returns the compartment number (1-based) of the given pH value."""

        return int(np.searchsorted(self.pH_compartment_thresholds(), pH_value, side='left')) + 1

    @onlycallonce
    def apply_frequency_offset(self, offset):
        """Adds a given constant offset onto the frequency map."""
        self._fmap += offset
        self._multi_comp_fmaps += offset
        self._f_offset = offset

    def mask_abs_error(self, abs_error_thr=15):
        """Mask array by a absolute fit error threshold."""
        self.mask_dict["abs_error"] = self._fmap_err > abs_error_thr

    def mask_cost_error(self, cost_error_thr=1.1):
        """Mask array by a absolute fit error threshold."""
        self.mask_dict["cost_error"] = self._fit_cost_map > cost_error_thr

    @onlycallonce
    def apply_1H_ref_map(self, ref_fmap, method="resize", scalefor13c=True):
        """Substracts given reference from fmap. Resizes reference map if necessray."""
        if scalefor13c:
            gyro_ratios = LAMOR_FREQUENCY_H1 / LAMOR_FREQUENCY_C13
        else:
            gyro_ratios = 1

        print(f"Using gyromagnetic ratio of {gyro_ratios:.2f}")
        if ref_fmap.shape != self.fmap.shape:
            if method == "resize":
                logger.warning(f"Resizing... {ref_fmap.shape} -> {self.fmap.shape}")
                ref_fmap = interpolate_masked_3d_array(ref_fmap, self.fmap.shape)
            elif method == "downsample":
                logger.warning(f"Downsampling... {ref_fmap.shape} -> {self.fmap.shape}")
                down_factors = np.array(ref_fmap.shape) / np.array(self.fmap.shape)
                for d in down_factors:
                    if not float(d).is_integer():
                        assert False, f"Downsampling factors: {down_factors}"
                ref_fmap = downsample_masked_array(
                    ref_fmap, tuple(down_factors.astype(int))
                )
            else:
                raise ValueError("Not a valid Method.")

        if hasattr(ref_fmap, "mask") and ref_fmap.mask.any():
            logger.warning(f"Replacing masked values with zero...")
            ref_fmap = ref_fmap.filled(0)
            #self.mask_dict["fmap_mask"] = ref_fmap.mask

        self._fmap -= ref_fmap / gyro_ratios
        
        return ref_fmap / gyro_ratios
    
    
    @onlycallonce
    def apply_1H_ref_map_multi_comp(self, ref_fmap, method="resize", scalefor13c=True, mask=None):
        """Substracts given reference from fmap. Resizes reference map if necessray."""
        if scalefor13c:
            gyro_ratios = LAMOR_FREQUENCY_H1 / LAMOR_FREQUENCY_C13
        else:
            gyro_ratios = 1

        print(f"Using gyromagnetic ratio of {gyro_ratios:.2f}")
        if ref_fmap.shape != self.fmap.shape:
            if method == "resize":
                logger.warning(f"Resizing... {ref_fmap.shape} -> {self.fmap.shape}")
                ref_fmap = interpolate_masked_3d_array(ref_fmap, self.fmap.shape)
            elif method == "downsample":
                logger.warning(f"Downsampling... {ref_fmap.shape} -> {self.fmap.shape}")
                down_factors = np.array(ref_fmap.shape) / np.array(self.fmap.shape)
                for d in down_factors:
                    if not float(d).is_integer():
                        assert False, f"Downsampling factors: {down_factors}"
                ref_fmap = downsample_masked_array(
                    ref_fmap, tuple(down_factors.astype(int))
                )
            else:
                raise ValueError("Not a valid Method.")

        if hasattr(ref_fmap, "mask") and ref_fmap.mask.any():
            logger.warning(f"Replacing masked values with zero...")
            ref_fmap = ref_fmap.filled(0)

        self._multi_comp_fmaps -= ref_fmap / gyro_ratios

        if not isinstance(mask, type(None)):
            # mask all fit results consistently so frequencies, amplitudes and pH stay aligned
            self._multi_comp_fmaps[:,mask == True] = np.nan
            self._multi_comp_amplitudes[:,mask == True] = np.nan
            self._multi_comp_pHmaps[:,mask == True] = np.nan

        return ref_fmap / gyro_ratios

    def plot_results(self, param_dict={}, anatref=None):
        """Warpper for plot_results2D and plot_results3D."""
        if self.Nslices == 1:
            return UmpirePlotter2D(self, parameters=param_dict)
        else:
            return UmpirePlotter3D(self, parameters=param_dict, anatref=anatref)

    def plot_spectrum(self, cut_off = 0):
        
        spectra, f_axis = self.spectral_data
        plot_spectrum(spectra, f_axis)

    def reconstruct_data(self, data_scaling_factor=None, remove_zeropadding=False):
        new_fid = postprocess_fid_file(self)
        
        # get encoding order
        ysz = self.method["PVM_Matrix"][1]
        zsz = self.method["PVM_Matrix"][2]

        y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
        z_indices = self.method["PVM_EncSteps2"] + int(np.floor(zsz / 2.0))
        
        matrixDims = self.method["PVM_Matrix"]
        n_reps = self.method["PVM_NRepetitions"]
        n_chans = self.method["PVM_EncNReceivers"]
        n_echoes = len(self.TEs)
        
        rearanged_k_space = np.empty((matrixDims[0], n_chans, n_echoes, matrixDims[1], matrixDims[2], n_reps), dtype="complex")
        running_index = 0

        for rep in np.arange(n_reps):
            for ph2_idx in np.arange(len(z_indices)):
                for ph1_idx in np.arange(len(y_indices)):
                    for echo_idx in np.arange(n_echoes):
                        for ch_idx in np.arange(n_chans):
                            
                            rearanged_k_space[:,ch_idx, echo_idx, y_indices[ph1_idx], z_indices[ph2_idx], rep] = new_fid[running_index, matrixDims[0]*ch_idx:matrixDims[0]+matrixDims[0]*ch_idx]

                        running_index +=1
                        
        
        reordered_fid_data = np.transpose(rearanged_k_space, (2, 0, 3, 4, 1, 5))
        

        if remove_zeropadding:
            self.method["PVM_Matrix"][0] -= 2
            self.pvm_matrix = self.method["PVM_Matrix"]
            self.method["PVM_SpatResol"] = self.method["PVM_Fov"]/self.method["PVM_Matrix"]
            self.pvm_res = self.method["PVM_SpatResol"]
            reordered_fid_data = reordered_fid_data[:,2:,:,:,:,:]
            
        
        reconstructed_data = fft(input_data=reordered_fid_data, axes=(1, 2, 3))
        data = reconstructed_data[:,:,:,:,:,0]

        #rotate to match bruker reconstructed file
        data = np.flip(data, axis=2)
        data = np.flip(data, axis=1)
        data = np.flip(data, axis=3)
        
        if isinstance(data_scaling_factor, type(None)):
            rescaled_data = data/np.max(np.abs(data))*100
            print("Data scaled by: " + str(np.max(np.abs(data))*100))
        else:
            #if provided, scale data by external scaling factor
            rescaled_data = data/data_scaling_factor
        
        #rescaled_data = np.roll(rescaled_data, 1,  axis= 2) 
        #rescaled_data = np.roll(rescaled_data, 1,  axis= 1)  

        from scipy.ndimage import shift
        rescaled_data = shift(rescaled_data, [0.0,1.0,0.5,0.5,0.0], mode="wrap")
        
        return rescaled_data

    def reconstruct_and_combine_multichannel(self, data_scaling_factor=None, remove_zeropadding= False, channel_combination_method = "sum"):
         
        data = self.reconstruct_data(data_scaling_factor, remove_zeropadding)

        if self.method["PVM_EncNReceivers"] > 1:
            #return data[:,:,:,:,1]
            return combine_multichannel(data, chan_dim=4,combination_method=channel_combination_method)
        else:
            return data.squeeze()


    @classmethod
    def load(cls, filename):
        """
        Load an umpireMGE-object from a file and display the timestamp of the
        saved object.

        Parameters:
            filename (str): The name of the file to load the object from.

        Returns:
            umpireMGE: The loaded object.
        """
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
        #print(f"Loading object saved on {tmp._timestamp}.")
        return tmp

    def save(self, filename):
        """
        Save the umpireMGE-object to a file, updating the timestamp and
        displaying information about the saved object.

        Parameters:
            filename (str): The name of the file to save the object to.
        """
        import os
        import datetime

        if filename.split(".")[-1] != "pickle":
            filename += ".pickle"

        self._filename = filename
        self._timestamp = datetime.datetime.now()

        with open(filename, "wb") as f:
            try:
                pickle.dump(self, f, 4)
            except pickle.PicklingError as perr:
                print(
                    "Note: Sometimes pickle can conflict with the "
                    "%autoreload extension.\n      If you altered a class "
                    "method, try running 'save()' again or restart you kernel.\n"
                )
                raise perr

        nbytes = os.path.getsize(filename)

        if nbytes < 1e6:
            sz = f"{nbytes * 1e-3:.2f} KB"
        elif nbytes < 1e9:
            sz = f"{nbytes * 1e-6:.2f} MB"
        else:
            sz = f"{nbytes * 1e-9:.2f} GB"

        print(f"Object saved to '{filename}' with a size of {sz}.")
