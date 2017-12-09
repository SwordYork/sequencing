source ~/.tf14/bin/activate

export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))

CUDA_VISIBLE_DEVICES=1 python nmt_train.py --config=config_en2fr_large
