MODEL=# MODEL_WEIGHT_PATH
DATA=# TEST_DATA_PATH
SAVE=# WHERE_TO_SAVE
PRED_SUFFIX=# NAME_FOR_SAVING_DATA
BATCH_SIZE=32
DATA_TYPE=test

# Although we don't use pusedo-labels and user information during inference time, 
# we need these configs (e.g., uid, dom, and dom_cls) for code simplification

python translate.py -model ${MODEL} -src ${DATA}test.srcmt.tok -uid ${DATA}test.USER -dom ${DATA}test.10  -output ${SAVE}${DATA_TYPE}.${PRED_SUFFIX}.unprocessed  -beam_size 5 -min_length 1 -batch_size ${BATCH_SIZE} -report_time -length_penalty wu -gpu 0 -block_ngram_repeat 22 -max_length 76 -dom_cls

cat ${SAVE}${DATA_TYPE}.${PRED_SUFFIX}.unprocessed | sed 's/ \#\#//g' > ${SAVE}${DATA_TYPE}.${PRED_SUFFIX}
cat ${SAVE}${DATA_TYPE}.${PRED_SUFFIX} | sacrebleu ${DATA}test.pe.tok
cat ${SAVE}${DATA_TYPE}.${PRED_SUFFIX} | sacrebleu ${DATA}test.pe.tok --m ter