ibans=(iban1007 iban197 iban198 iban309 iban331 iban338 iban377 iban380 iban464 iban473 iban474 iban491 iban505 iban538 iban546 iban573 iban582 iban585 iban587 iban590 iban618 iban631 iban633 iban646 iban685 iban743 iban764 iban793 iban908 iban933 iban976 iban978 iban998)

for iban in $ibans; do
    uv run clique-analysis.py $iban
done

