source ~/.tf14/bin/activate

export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))

CUDA_VISIBLE_DEVICES=0 python nmt_infer.py --config=config_en2fr_large --test-src='data/newstest2014.en.tok'

ls -t dev_outputs  | head -n1 |awk '{printf("newest file: %s\t",$0)}' >> bleu.hist
perl multi-bleu.perl data/newstest2014.fr.tok < test.out >> bleu.hist
