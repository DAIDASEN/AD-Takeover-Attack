CUDA_VISIBLE_DEVICES=0 python main.py \
                            --epsilon 0.032 \
                            --step_size 0.0039 \
                            --iter 1000 \
                            --seed 256 \
                            --gpu 0 \
                            --root_path ./output \
                            --dataset coco_test
