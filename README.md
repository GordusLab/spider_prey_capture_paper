# Web vibration analysis pipeline

## Fourier transform of web vibration
1. Run `main.py` to perform web vibration analysis:
    * Fast Fourier Transform (FFT) analysis:  `FFT_web_vibration.py`
	    * The Fast Fourier Transform (FFT) of a time series of length \( N \) is defined as:
		    
		    $$ \text{FFT}[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi j kn / N} $$
		    
		    , where \(x[n]\) is the discrete-time signal and \(k\) is the frequency bin index. To investigate whole-web vibration, we computed the mean FFT across all silk lines.
	
     *  Short Time Fourier Transform (STFT) analysis:  `STFT_web_vibration.py`
    
		* The Short-Time Fourier Transform (STFT) is defined as:
       
			$$ \text{STFT}[m, k] = \sum_{n=0}^{N-1} x[n + mR] \cdot \omega[n] \cdot e^{-2\pi j kn / N} $$
			
			,where:
			- \( x [n] \) is the input signal,
			- \( \omega[n] \) is the window function of length \( N=400 (0.4s)\),
			- \( R =20 (0.02s)\) is the hop size,
			- \( m \) is the time frame index,
			- \( k \) is the frequency bin index.
3. Run `plot_FFT_heatmap_ratio.py` to plot Fourier Transform heat map:
	Ratio of frequency power between the experimental and control groups

## Regions of interest (ROIs) analysis of prey vibration on webs

#### Top camera
1. `Partialweb_vibration_analysis.py` : ROI analysis of top camera recordings to extract pure fly signal
    * Two selection windows will open: one for the spider and one for the prey. After selecting the ROIs, the program will automatically track both objects throughout the entire video. To modify the ROIs later, press "f" to reselect the prey (fly) or "s" to reselect the spider.
    * After selecting ROIs, `FFT_partial_web` and `STFT_partial_web` will be called to calculate FFT and STFT of pixel intensity fluctuation within the ROIs. For STFT, we used window = 400 (0.4s) and time step = 20 (0.02s).

2. `plot_roi_movie.py`: visualizing spider and fly peripheral STFT spectrum
3. `plot_roi_normalized_movie.py`: visualizing normalized fly STFT spectrum (fly center normalized by fly peripheral)
4. `plot_roi_normalized_fly_states.py`: plot normalized fly STFT figures for all top videos

#### Side camera
1. `Partialweb_vibration_analysis_side.py` : ROI analysis of side camera recordings to extract pure fly signal
    * Two selection windows will open: one for the spider and one for the prey. After selecting the ROIs, the program will automatically track both objects throughout the entire video. To modify the ROIs later, press "f" to reselect the prey (fly) or "s" to reselect the spider.
    * After selecting ROIs, `FFT_partial_web` and `STFT_partial_web` will be called to calculate FFT and STFT of pixel intensity fluctuation within the ROIs. For STFT, we used window = 40 (0.4s) and time step = 2 (0.02s).

2. `plot_roi_movie_side.py`: visualizing spider and fly peripheral STFT spectrum

#### Other scripts
1. `Partialweb_vibration_analysis_side_auto.py` : This script will automatically pop out a window for fly selection if it fails to track fly in the algorithm
2.  `Partialweb_vibration_analysis_side_piezo.py` : This script is for piezo experiment
3.  `Partialweb_vibration_analysis_side_piezo_auto.py` : This script will automatically pop out a window for piezo selection if it fails to track piezo in the algorithm

## Localization analysis

#### Localization based on pixel fluctuation signal
1. `localization_signalmap_visualization.py` : 
	* It extracts pixel fluctuation signal within spider's peripheral field and save it as `_signal_map_data.npz`
	* It pops out a window for you the select the radii you are interested in. 
	* In the second window, it plots mean pixel signal along the selected radii as a function of time.
2. `localization_results.py`: plot the linear regression results for all the videos
#### Localization based on STFT area under the curve (AUC)
1. `localization_AUC.py` : 
	* It calls `localized_spatial_AUC_web_vibration` to calculate area under STFT power between freq_m1=0 and freq_m2 = 50 for each pixel in spider's peripheral field.
	* We use  freq_m1=0 and freq_m2 = 50 because fly's signal is usually less than 50 Hz.
	* For STFT, we use window size = 40, with time step = 20 Since fly's signal is usually less than 50 Hz, we sacrifice frequency resolution and keep good temporal resolution for localization analysis. 
2. `localization_AUC_visualization.py` : This script is to visualize how AUC signal change over time in all radii
	* It pops out a window for you the select the radii you are interested in. 
	* In the second window, it plots mean AUC signal along the selected radii as a function of time.
