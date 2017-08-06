#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Author: Sword York
#   GitHub: https://github.com/SwordYork/sequencing
#   No rights reserved.
#
max_src = 50
max_trg = 100
src_file_name = 'data_en2zh/en.tok.shuf'
trg_file_name = 'data_en2zh/zh.tok.shuf'

src_file_out = open('data_en2zh/en.tok.shuf.filter', 'w')
trg_file_out = open('data_en2zh/zh.tok.shuf.filter', 'w')

with open(src_file_name, 'r') as src_file, open(trg_file_name, 'r') as trg_file:
    while True:
        sl = src_file.readline()
        tl = trg_file.readline()

        if (not sl) and (not tl):
            break

        if len(sl.split(' ')) > max_src or len(tl) > max_trg:
            continue

        src_file_out.write(sl)
        trg_file_out.write(tl)

src_file_out.close()
trg_file_out.close()
