# IS2_ATL10_processing

Repo with code to: 
  1. re-process IceSat-2 ATL10 data to include lead tie points up to 100 km away (instead of the default of 10 km in the NSIDC product). Using the lead tie points further away increases IceSat-2 freeboard coverage in regions where lead points are sparse (e.g. around fast ice). A 'new' freeboard value is calculated as: freeboard_new = ATL07_height - SSHA_interpolated
  2. find beam-to-beam spatial intersections for a specified time window.
  3. statistical analysis of the intersections to verify that the interpolated SSHA values obtained using lead tie points up to 100 km away provide robust freeboard estimates.


The scripts have to be run in consecutive order:
1. reprocess_ATL10_freeboards.py -> obtain 'new' freeboards (only where there was no original ATL10 freeboard)
2. IS2_find_intersections.py -> find beam-to-beam intersections (where there are freeboard values)
3. IS2_gather_stats_all_intersections.py -> gather statistics about the intersections (e.g. delta_freeboard) and save to pickle dictionary
4. IS2_analyse_all_intersections.py -> analyse the statistics from the pickle dictionary


