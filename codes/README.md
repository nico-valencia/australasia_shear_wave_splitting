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

station name    network name    lat     lon     phases used     phi     dt  comment     # of measurements