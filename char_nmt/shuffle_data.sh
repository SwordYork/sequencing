#!/bin/bash

set -e

src=$1
trg=$2

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# shuffle
random_source=$(date +%N)
echo $random_source
shuf --random-source=<(get_seeded_random $random_source) ${src} > ${src}.shuf
shuf --random-source=<(get_seeded_random $random_source) ${trg} > ${trg}.shuf

