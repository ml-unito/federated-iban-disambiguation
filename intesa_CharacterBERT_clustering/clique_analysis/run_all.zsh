
# ibans errati clustering-kernel-S67895_2025-06-17_16-16-28
ibans67895="iban1007 iban197 iban201 iban273 iban297 iban32 iban321 iban345 iban384 iban454 iban465 iban472 iban473 iban475 iban476 iban489 iban506 iban507 iban527 iban546 iban560 iban587 iban601 iban609 iban618 iban686 iban704 iban715 iban723 iban771 iban883 iban984 iban993 iban998"

# ../out/clustering/clustering-kernel-S81789_2025-06-17_16-16-05/labeled_couple_dataset.csv
ibans81789="iban1007 iban1019 iban161 iban219 iban286 iban287 iban295 iban327 iban361 iban384 iban400 iban429 iban430 iban439 iban443 iban461 iban476 iban489 iban506 iban520 iban548 iban609 iban633 iban636 iban65 iban704 iban824 iban847 iban861 iban996 iban998"

# ../out/clustering/clustering-kernel-S47874_2025-06-17_16-15-42/labeled_couple_dataset.csv
ibans47874="iban1007 iban159 iban180 iban196 iban229 iban245 iban312 iban327 iban356 iban387 iban425 iban430 iban475 iban479 iban481 iban501 iban506 iban507 iban513 iban546 iban572 iban584 iban601 iban609 iban653 iban686 iban704 iban761 iban782 iban835 iban850 iban863 iban867 iban883 iban908 iban91 iban925 iban933"

# ../out/clustering/clustering-kernel-S23517_2025-06-17_16-15-18/labeled_couple_dataset.csv
ibans23517="iban1008 iban1019 iban116 iban170 iban172 iban180 iban197 iban206 iban295 iban345 iban348 iban360 iban380 iban387 iban419 iban443 iban454 iban511 iban519 iban560 iban571 iban584 iban590 iban600 iban630 iban645 iban65 iban685 iban700 iban704 iban731 iban775 iban821 iban824 iban830 iban882 iban883 iban9 iban922 iban933 iban967 iban976 iban998"

# ../out/clustering/clustering-kernel-S9046_2025-06-17_16-14-57/labeled_couple_dataset.csv
ibans9046="iban114 iban116 iban132 iban153 iban159 iban189 iban219 iban248 iban250 iban323 iban337 iban348 iban360 iban361 iban380 iban391 iban429 iban464 iban472 iban475 iban507 iban519 iban548 iban573 iban584 iban590 iban636 iban645 iban65 iban654 iban682 iban700 iban704 iban710 iban728 iban731 iban766 iban771 iban775 iban83 iban834 iban877 iban882 iban9 iban940 iban967"


for iban in $ibans23517; do
    uv run clique-analysis.py $iban --fname ../out/clustering/clustering-kernel-S23517_2025-06-17_16-15-18/labeled_couple_dataset.csv
done
