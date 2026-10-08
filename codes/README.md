january 28 2025 

### SHEAR WAVE SPLITTING WORKFLOW IN PYTHON USING SPLITWAVEPY ###

- download waveform data for selected region and magnitude range, can choose between SKS or S phase 
- waveform metadata gets organized and structured into more convenient pandas DataFrames, saved as txt files 
- 

codes are based on prior work done by Barron and Utpal Kumar 
all scripts besides Utpal's download scipts written by Nicolas Valencia 2023 - 2025

1. download_SKS_phase.py or download_S_phase.py

2. 

3. 

4. 

5. 





friday april 3 2026 - UPDATE PARAMETERS FOR EACH BATCH OF MEASUREMENTS

2 main things are changed every time a new batch of splitting measurements is made:
- frequency band
- window size 

I have been doing all my measurements using the 'normal' band for filtering for events, this is the one I have been using since the beginning of time. 
The window size is up for interpretation....
    - my senior thesis window size is mentioned in the writing and I will place it here: 
    - the spring measurements for complete/aus use a 30 second window, 15 seconds on each side (03162026)
    - the spring measurements for complete/swp use a 30 second window, 15 seconds on each side (04012026)

    - the second batch of spring measurements for complete/aus use a 30 second window, 10 seconds to the left and 20 seconds to the right (04032026)
    - the second batch of spring measurements for complete/aus use a 30 second window, 10 seconds to the left and 20 seconds to the right (04032026)


some housekeeping to do and data correlation: 

complete southwest pacific shear wave splitting measurements google slides === measurements/spring_plots/single_station

the most recent batch of group plots made in spring_updates === measurements/spring_plots/april_10
- SI error max of 0.5 
- snr before of 3 
- delta travel max at 25 seconds 
- 10-20 window around sta/lta peak
- 1 < ellipticity < 1000

14 "number 1" stations from analysis of the most recent batch:

ARMA    D" issue at baz 120-150, deep S resolves pattern well, good example of why we cant fully trust SKS and then deep S helps a lot

AS31    SKS is already good, deep S follows pattern, good way to show the "scatter" of SI around the sinusoid, but it still follows the receiver side pattern [Deep S is consistent]

BBOO    SKS good with deep S, shallow S follows pattern really well. another good example of receiver side SI is median, spread is source side or raypath

COEN    SKS good, deep S follows it, shallow S visually helps 

EIDS    SKS good, deep S resolves azimuthal coverage well, shallow S is weird here

MNAI    no SKS, deep S and shallow S agree well in baz 0-150, need to check raypaths

MTN     SKS good, deep S doesnt help resolve curve but helps show where they agree, S shallow is scattered

MULG    only with S shallow does this station look decent. heavily dependent on shallow S and thus those measurements must be studied

NOUC    beautiful SKS, deep S is super helpful as well. heavily scattered S, will be worth investigating

QIS     SKS alright, deep S helps a lot with resolving the SI shape, not too confident with this one compared to rest of batch.

SNZO    SKS and deep S alright, shallow S helps a LOT for this station, along with its strong SI values

STKA    great example of deep S coverage going crazy and helping resolve the curve

TOO     my personal favorite. deep S helps resolve the characteristic SI shape, shallow S helps visually, but reduces dt (this can be discussed)

XMI     shallow S looks best here, so don't trust fully. worth investigating these raypaths and tectonic setting. 

XMIS    same as before, but stronger dt values. need to explore where SKS and deep S and shallow S agree



we prefer using SKS only, deep S is consistent at times, some stations we compelemnent with deep S, rarely we include shallow S 

of course there are uncertainties we cannot account for but thats the way it goes...

BBOO: truncate SI curve to only use 0-180, explain that the deep S are strong outliers and we choose not to use them

COEN: using both SKS and deep S is consistent with only using SKS, we prefer using deep S too for better coverage

MNAI: S consistent with deep S and include it! 

XMI / XMIS: all S consistent, SKS and deep S not enough coverage to resolve SI


basic table 






comprehensive table

station name    network name    lat     lon     phases used     phi     dt  comment                                         # of measurements

ARMA            AU                              SKS, deep S                 no SKS 120-150

AS31            AU                              SKS                         deep S visually consistent

BBOO            AU                              SKS, deep S                 no SKS 145-165, no S > 200

COEN            AU                              SKS, deep S                 see previous comment w/ Jonathan

EIDS            AU                              SKS, deep S                 SI > 1.5 is removed (too strong)

MNAI            GE

MTN             AU                              SKS, deep S                 15 bin width has larger dt than 20 bin width

MULG            AU                              all phases                  15 bin width has larger dt than 20 bin width

NOUC            G                               SKS, deep S                 started tracking map events HERE

QIS             AU                              SKS                         best one for just SKS 

SNZO            IU                              all phases

STKA            AU                              SKS, deep S                 10 bin largest dt, then 15, then 20

TOO             AU                              SKS, deep S                 need to remove deep S outliers still

XMI             AU                              all phases                  bin width 30 best results...    

XMIS            AU                              all phases                  bin width 30 too 










spring_updates.ipynb filters per station in cell 

ARMA: 


Meeting Notes May 5 2026

review student T distribution and how necessary this is 

Barbara suggests a simple inverse std (1/std) instead of my 'margin' , compare the two methods

NOUC: both SKS and deep S roduce the same result, but this shows the larger spread in deep S due to source side anisotropy, yet it can still help azimuthal coverage

quantify removal of SKS scatters: variance reductance

reduce bins to 1 per azimtuhal bin. average the uncertainties after doing it by phase to simplify matters

END OF NOTES



May 7 2026 

changed weighting scheme for binned measurements 

SI err is used for filtering, and also a weighted mean and std 

std of bin is used as sigma value, which is inverse weight for curve fit 

ARMA without scatter cov matrix:    phi 7.2 dt 0.53

[[ 5.68828754e+00 -8.70041385e-03] 
 [-8.70041385e-03  3.97494158e-03]]
[2.38501311 0.06304714]

ARMA with scatter:                  phi 7.3 dt 0.49
[[ 6.69977953e+00 -7.78848433e-03]
 [-7.78848433e-03  3.51463970e-03]]
[2.58839323 0.0592844 ]


BBOO with scatter SKS and S above 200: phi 67.3 dt 0.34 
[[4.64078705e+01 4.29821957e-02]
 [4.29821957e-02 1.18114243e-04]]
[6.81233224 0.01086804]

BBOO without scatter and no S above 200: phi 61.0 dt 0.34
[[ 7.78396944e+01 -1.28783174e-01]
 [-1.28783174e-01  2.92594416e-04]]
[8.82268068 0.01710539]


# Summary of Changes to Binning Schema and Splitting Parameter Inversion

## Overview

We have made significant improvements to both the data binning methodology and the parameter inversion approach for shear-wave splitting analysis. These changes address statistical rigor, computational efficiency, and alignment with standard practices in the field.

## Changes to Binning Statistics

Previously, we calculated 95% confidence intervals for each azimuthal bin using the Student's t-distribution. This approach computed a critical t-value based on the degrees of freedom (n-1) and used it to calculate a margin of error as `margin = t_crit × (std / √n)`. This margin was then used as the uncertainty weight passed to the fitting routine. However, following discussions with advisors, we recognized that this approach did not fully account for the individual measurement uncertainties within each bin and that the confidence interval formulation was unnecessarily complex for our purposes.

We have now transitioned to using weighted statistics that incorporate the measurement errors of individual splitting intensity values. For each bin, we calculate a weighted mean and weighted standard deviation using the `weighted_mean_and_std()` function, which takes both the splitting intensity values and their individual errors as inputs. The weighted standard deviation (`std_w`) directly represents the uncertainty in the bin's mean value while accounting for the varying quality of individual measurements. This weighted standard deviation is now used as the uncertainty metric (`sigma`) passed to the fitting routine, providing a more intuitive and statistically appropriate representation of data quality. The variance is simply the square of this weighted standard deviation, and we weight each bin by inverse variance (1/σ²) during the inversion, which is standard practice in weighted least squares.

## Changes to the Inversion Method

The most significant methodological change involves switching from nonlinear least squares optimization to direct linear least squares inversion. Originally, we used scipy's `curve_fit` function to fit the splitting intensity model `SI(θ) = δt·|sin[2(θ-φ)]| + tilt` through iterative optimization. This nonlinear approach required an initial parameter guess, could potentially get stuck in local minima, and was computationally slower due to its iterative nature. More importantly, after discussions with advisors, we learned that this approach is less standard in the seismological literature for this particular problem.

We have now implemented a linear least squares solution by exploiting the trigonometric identity that allows us to rewrite the splitting intensity model in a linear form. Specifically, we expand `sin[2(θ-φ)] = sin(2θ)·cos(2φ) - cos(2θ)·sin(2φ)`, which can be rearranged as `SI(θ) = A·sin(2θ) + B·cos(2θ)`, where `A = δt·cos(2φ)` and `B = -δt·sin(2φ)`. This formulation is linear in the coefficients A and B, allowing us to construct a design matrix X with columns `[sin(2θ), cos(2θ)]` and solve the system directly using the weighted normal equations `(X^T W X)p = X^T W y`, where W is a diagonal weight matrix containing the inverse variances (1/σᵢ²) and p is the parameter vector [A, B].

Once we obtain A and B from the linear solution, we convert back to the physical parameters using `δt = √(A² + B²)` and `φ = 0.5 · arctan2(-B, A)`. The negative sign in the arctan2 function is critical and arises from the fact that B = -δt·sin(2φ) in the trigonometric expansion. This linear approach requires no initial guess, always finds the global minimum, is computationally faster through direct matrix inversion, and represents the standard approach expected by reviewers in the seismic anisotropy community.

## Model Simplification

We have also simplified the model by removing the tilt parameter. Previously, our model included a baseline offset term (`SI = δt·|sin[2(θ-φ)]| + tilt`), fitting for three parameters simultaneously. However, we found that the tilt parameter could trade off with the delay time, potentially absorbing some of the amplitude and leading to parameter correlation. The current two-parameter model (`SI = δt·sin[2(θ-φ)]`) is simpler, avoids these trade-offs, and is more standard in the splitting literature. Additionally, we removed the absolute value from the sine function, as the linearized form naturally handles the azimuthal variation without requiring this constraint.

## Technical Corrections

Several technical issues were identified and corrected during this transition. First, we discovered a parameter name mismatch between `analyze_station()` (which was passing `weights=weights`) and `bin_measurements()` (which expected `phase_weights=None`), resulting in a TypeError that prevented the binning function from executing. This has been corrected to use consistent naming (`phase_weights`) throughout.

Second, we initially had an incorrect sign convention in converting from linear parameters (A, B) to the physical fast direction (φ). Using `arctan2(B, A)` was giving φ ≈ 173° instead of the expected φ ≈ 7°. After testing, we confirmed that the correct formulation is `arctan2(-B, A)`, which accounts for the negative sign in the relationship `B = -δt·sin(2φ)`.

Third, the Jacobian matrix used for uncertainty propagation from (A, B) to (φ, δt) initially missed the radians-to-degrees conversion factor. The correct Jacobian must include the factor 180/π ≈ 57.3 when computing derivatives of φ with respect to A and B, since φ is reported in degrees while the arctan2 function works in radians. Without this correction, the uncertainty in φ was underestimated by nearly a factor of 60.

Finally, we clarified the use of the `absolute_sigma` parameter in curve_fit. The default setting (`absolute_sigma=False`) rescales the returned covariance matrix based on the reduced chi-squared of the fit, essentially second-guessing the provided uncertainties. Since we carefully calculate realistic uncertainties for each bin, we should use `absolute_sigma=True` to ensure the returned parameter uncertainties directly reflect our stated data quality.

## Error Bar Visualization

For plotting purposes, we recommend using error bars of `2×std` (twice the weighted standard deviation) to represent approximately 95% confidence intervals. The single standard deviation error bars, while statistically accurate, are often too small to be easily visible in figures. Using 2σ provides better visibility while maintaining statistical meaning, as this corresponds to roughly 95% confidence for large sample sizes and is standard practice in many geophysical publications.

## Validation and Results

The linear least squares approach has been validated against the curve_fit method with proper settings (`absolute_sigma=True`). For a representative station, curve_fit yielded φ = 7.0 ± 2.4° and δt = 0.53 ± 0.06 s, while the linear least squares method gives φ = 7.2 ± 2.4° and δt = 0.53 ± 0.06 s. The results are essentially identical, confirming that the linear approach correctly reproduces the nonlinear optimization when both are properly configured. The covariance matrices also show negligible correlation between φ and δt (r ≈ 0.06), indicating that both parameters are independently well-constrained by the data.

## Summary

These changes represent a shift toward more statistically rigorous, computationally efficient, and professionally standard methods for analyzing shear-wave splitting. The weighted binning statistics properly account for measurement quality, the linear least squares inversion is transparent and aligns with field conventions, and the simplified two-parameter model avoids unnecessary complexity. All technical issues have been resolved, and results have been validated against alternative methods. This updated workflow is well-suited for thesis documentation and peer-reviewed publication.




GEBCO Bathymetric Compilation Group 2026(2026). The GEBCO_2026 Grid - a continuous terrain model for oceans and land at 15 arc-second intervals. NERC EDS British Oceanographic Data Centre NOC. doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa



STATION_PHASE_MAP = {
    "ARMA": "sks_deep", 
    "AS31": "sks", 
    "BBOO": "sks_deep",
    "BCON": "sks_deep", 
    "BKB": "all",
    "BKNI": "complex",
    "BLDU": "sks",
    "CAN": "sks_deep",
    "CMSA": "complex",
    "CNB": "sks_deep",
    "COEN": "sks",
    "CTA": "complex",
    "CTAO": "complex",
    "EIDS": "sks_deep",
    "FAKI": "complex",
    "FITZ": "sks_deep",
    "FORT": "all",
    "GIRL": "all",
    "GUMO": "complex",
    "HTT": "complex",
    "INKA": "complex",
    "JAGI": "complex",
    "KDU": "complex",
    "KLGGI": "sks_deep",
    "KMBL": "complex",
    "KNRA": "complex",
    "LCRK": "complex",
    "LHI": "complex",
    "LUWI": "complex",
    "MANU": "complex",
    "MEEK": "complex",
    "MILA": "sks_deep",
    "MMRI": "sks_deep",
    "MNAI": "all",
    "MOO": "all",
    "MORW": "complex",
    "MTN": "sks_deep",
    "MULG": "complex",
    "MUN": "sks_deep",
    "NFK": "complex",
    "NOUC": "sks_deep",
    "NWAO": "complex",
    "OOD": "complex",
    "PMBI": "complex", #potentially a null station?
    "PMG": "complex",
    "QIS": "sks",
    "QLP": "complex",
    "RABL": "complex",
    "RIV": "complex", # another potential null
    "SAUI": "all",
    "SMRI": "complex",
    "SNZO": "all",
    "SOEI": "complex",
    "STKA": "sks_deep",
    "TNTI": "sks_deep",
    "TOO": "sks_deep",
    "TROP": "sks_deep",
    "UGM": "complex",
    "WB2": "complex",   # good null
    "WRKA": "complex", # super good null
    "YNG": "PHASETYPE",
    "XMI": "all",
    "XMIS": "all",
    "YNG": "sks_deep",
    }



July 24 2026 

genuinely forgetting which SNR criteria I used to generate the measurements I currently have

test1: 3.0 and 0.0 

test2: 1.5 and 1.5 

gonna make some plots and check it out!


test3: 3.0 and 1.5 , ~5800 measurements 

test4: 3.0 and 1.0 , ~8000 measurements


some notes on this: test 1 was the criteria used for all of the mapping I have made so far. this completely ignores the coda of the phase or other energy that can contaminate our phase arrival. test 2 has an SNR too little that adds a lot of phases to our already pretty well defined stations. 

test 3 and 4 make more sense, in order to have a prominent phase arrival but also an attempt to remove noise on the back half, test 4 is more relaxed with the amount of contaminated energy allowed on the back half. 

AS OF JULY 28 2026: TEST 4 CRITERIA IS BEST FOR MY PURPOSES

visually everything looks the same except BLDU has a 90° shift in fast axis...check this with Jonathan and see what he thinks

Jul 30 2026 

Number of Seismic Stations in eventData: 115
Number of Stations After Applying waveform filtering criteria: 83 
Number of stations within the station_keys dictionary: 59

83 includes stations after filtering but are not in the dictionary (this is just personal assessment of data quality per station, no filtering criteria here besides me and jonathan)