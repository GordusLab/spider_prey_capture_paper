
## Morlet Wavelet Analysis 

Apply the Morlet wavelet function to our side camera videos to study spider's prey capture movements.

###  Pipeline 

1. Run **`run_sidecamera_analysis.py`**: Run the `STFT_sidecamera.py` and `wavelet_sidecamera.py` for the joints data.
	1. **wavelet_sidecamera.py**: Applies a [[#Morlet Wavelet transform]]to the limb tracking datasets from the side camera. 
		* There are totally 20 joints tracked by DeepLabCut.
		* Videos were recorded at 100 Hz.
		* X (vertical axis) is mean-centered; Y(horizontal axis) is centered for centroid)

	2. **STFT_sidecamera.py**: Applies a STFT transform to the limb tracking datasets from the side camera. 
2. Run `cut_wavelet.py` to trim the beginning of the videos to exclude the time before the flies are introduced onto the web.

###Package version:
1. python == 3.6.7
2. openTSNE == 0.4.3

###Conda Environment:
spider-side-behavior

### Morlet Wavelet transform

The wavelet transform described by Berman *et al.*³ was defined as follows:

$$
	W_{s, \tau}[y(t)] = \frac{1}{\sqrt{s}} \int_{-\infty}^{\infty} y(t) \, \psi^*\left( \frac{t-\tau}{s} \right) \, dt,
$$

with

$$
	\psi(\eta) = \pi^{-1/4} e^{i \omega_0 \eta} e^{-\eta^2/2},
$$

where $y(t)$ is the spider’s postural time series, $\omega_0 = 5$ is a non-dimensional parameter, $\tau$ is a point in time, and $s$ is the time scale of interest as a function of frequency $f$:

$$
	s(f) = \frac{\omega_0 + \sqrt{2 + \omega_0^2}}{4\pi f}.
$$

The power spectrum is:
		
$$
		S(k, f; \tau) = \frac{1}{C(s(f))} \left| W_{s(f), \tau} [y_k(t)] \right|,
$$
with the scalar function

$$
		C(s) = \frac{\pi^{-1/4}}{\sqrt{2}s} e^{\frac{1}{4}(\omega_0 - \sqrt{2+\omega_0^2})^2}.
$$

Finally, the frequency range used was between $f_{\text{min}} = 0.1$ Hz and the Nyquist frequency $f_{\text{max}} = 50\ \text{Hz}$, with 50 frequencies spaced as follows:

$$
		f_i = f_{\text{max}} \cdot 2^{-\frac{(i-1)}{N_f-1} \log_2\left( \frac{f_{\text{max}}}{f_{\text{min}}} \right)}.
$$