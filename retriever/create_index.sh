#intfloat/e5-base-v2
#raco:--retriever_path /home/user10/DRAGIN/Retreiver_training/wandb/offline-run-20250225_084709-ntnt3pk2/files/checkpoints/retriever/question_encoder \
#qe2e : /home/user10/DRAGIN/Retreiver_training/wandb/qe2e_v2/files/checkpoints/retriever/question_encoder
#coconut : /home/user10/DRAGIN/Retreiver_training/wandb/coconut/files/checkpoints/retriever/question_encoder
#zebra: sapienzanlp/zebra-retriever-e5-base-v2

retriever_path=/home/user10/DRAGIN/Retreiver_training/wandb/coconut/files/checkpoints/retriever/question_encoder

for i in 1 2 3
do
  echo "Processing index $i"

  CUDA_VISIBLE_DEVICES=0 python create_index.py \
  --retriever_path $retriever_path \
  --data_path /home/user10/DRAGIN/data/commonsense_retrieval/Commonsense20M_$i.jsonl \
  --output_dir /home/user10/RALM_CSQA/documents/explanation_embeddings$i \

done

CUDA_VISIBLE_DEVICES=0 python create_index.py \
  --retriever_path $retriever_path \
  --data_path "../data/Generated_Knowledge.jsonl" \
  --output_dir /home/user10/RALM_CSQA/documents/explanation_embeddings4 \

