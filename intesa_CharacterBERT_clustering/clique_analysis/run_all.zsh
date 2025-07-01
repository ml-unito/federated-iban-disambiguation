
# ibans errati clustering-kernel-S9046_2025-06-16_15-17-16
# ibans="iban114 iban116 iban132 iban153 iban159 iban189 iban219 iban248 iban250 iban323 iban337 iban348 iban360 iban361 iban377 iban380 iban391 iban429 iban464 iban472 iban475 iban507 iban519 iban548 iban573 iban584 iban590 iban636 iban645 iban65 iban654 iban682 iban700 iban704 iban710 iban728 iban731 iban766 iban771 iban775 iban83 iban834 iban877 iban882 iban9 iban940 iban967"

# ibans errati clustering-kernel-S23517_2025-06-16_15-17-38
#ibans="iban1008 iban1019 iban116 iban170 iban172 iban180 iban197 iban206 iban295 iban345 iban348 iban360 iban377 iban380 iban387 iban415 iban419 iban443 iban454 iban511 iban519 iban560 iban571 iban584 iban590 iban600 iban630 iban645 iban65 iban685 iban700 iban704 iban731 iban775 iban821 iban824 iban830 iban882 iban883 iban9 iban922 iban933 iban967 iban976 iban998"

# ibans errati clustering-kernel-S67895_2025-06-17_16-16-28
ibans="iban1007 iban197 iban201 iban273 iban297 iban32 iban321 iban345 iban384 iban454 iban465 iban472 iban473 iban475 iban476 iban489 iban506 iban507 iban527 iban546 iban560 iban587 iban601 iban609 iban618 iban686 iban704 iban715 iban723 iban771 iban883 iban984 iban993 iban998"

for iban in $ibans; do
    uv run clique-analysis.py $iban --fname ../out/clustering/clustering-kernel-S67895_2025-06-17_16-16-28/labeled_couple_dataset.csv
done
