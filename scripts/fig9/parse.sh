
SIZES=( 256 1024 4096 16384 $(( 64 * 1024 )) $(( 256 * 1024 )) $(( 1024 * 1024 )) )
configs=( "noacc" "dummy" "syncdemote" "syncpref" "syncboth" )

BINS=( "./ufh" "./mmp" )

for bin in "${BINS[@]}"; do
  prefix="logs/$(basename $bin)"


  echo "$prefix"

  for size in "${SIZES[@]}"; do
    for config in "${configs[@]}"; do
      grep -a -v -e get -e info ${prefix}_${config}_$size.log  > tmp && mv tmp ${prefix}_${config}_$size.log
    done
  done

  for size in "${SIZES[@]}"; do
    echo "$size"
    echo ""
    paste ${prefix}_noacc_$size.log ${prefix}_dummy_$size.log ${prefix}_syncdemote_$size.log ${prefix}_syncpref_$size.log  \
    ${prefix}_syncboth_$size.log | tee ${prefix}_$size.log
    echo ""
    echo ""
  done

done