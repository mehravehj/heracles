#!/bin/bash
my_dir=$(cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd)

echo "[*] Hydra..."
for x in $( find "$my_dir/hydra-nonuniform" -name *.log )
do
	echo -n "[>] "
	echo "$x" | rev | cut -d'/' -f2-3 | rev

	grep -P "(BALANCED|STANDARD) ACC" "$x" | sed -e 's/^/      /'
done

echo ""
echo "[*] Robust ADMM..."
for x in $( find "$my_dir/r-admm-nonuniform" -name *.log )
do
	echo -n "[>] "
	basename "$x" | sed -e 's/.log$//'

	grep -P "(Resuming from checkpoint:|STANDARD ACC|BALANCED ACC)" "$x" | sed -e 's/^/      /'
done
