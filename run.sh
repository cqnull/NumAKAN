#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -J nattsyn_bl
#BSUB -o %J.out
#BSUB -e %J.err

python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset semeval_post
python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset semeval_headline
python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset fiqa_post
python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset fiqa_headline

python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset semeval_post --pretrained_bert_name ./plm/finbert
python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset semeval_headline --pretrained_bert_name ./plm/finbert
python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset fiqa_post --pretrained_bert_name ./plm/finbert
python ./run_baseline.py --model_name bert_spc --batch_size 64 --dataset fiqa_headline --pretrained_bert_name ./plm/finbert

python ./run_baseline.py --model_name dualgcn --batch_size 64 --dataset semeval_post
python ./run_baseline.py --model_name dualgcn --batch_size 64 --dataset semeval_headline
python ./run_baseline.py --model_name dualgcn --batch_size 64 --dataset fiqa_post
python ./run_baseline.py --model_name dualgcn --batch_size 64 --dataset fiqa_headline

python ./run_baseline.py --model_name mgfn --batch_size 64 --dataset semeval_post
python ./run_baseline.py --model_name mgfn --batch_size 64 --dataset semeval_headline
python ./run_baseline.py --model_name mgfn --batch_size 64 --dataset fiqa_post
python ./run_baseline.py --model_name mgfn --batch_size 64 --dataset fiqa_headline

python ./run_baseline.py --model_name ssegcn --batch_size 64 --dataset semeval_post
python ./run_baseline.py --model_name ssegcn --batch_size 64 --dataset semeval_headline
python ./run_baseline.py --model_name ssegcn --batch_size 64 --dataset fiqa_post
python ./run_baseline.py --model_name ssegcn --batch_size 64 --dataset fiqa_headline