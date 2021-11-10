#cd ${LEAF_ROOT}/data/celeba
#echo 'Creating new LEAF dataset split.'
#./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.8

# Format for Flower experiments. Val/train fraction set to 0.25 so validation/total=0.20
cd ${LOADER_ROOT}/celeba
python split_json_data.py \
--save_root ${SAVE_ROOT}/celeba \
--leaf_train_jsons_root ${LEAF_ROOT}/data/celeba/data/train \
--leaf_test_jsons_root ${LEAF_ROOT}/data/celeba/data/test \
--val_frac 0.25 \
--leaf_celeba_image_dir ${LEAF_ROOT}/data/celeba/data/raw/img_align_celeba
echo 'Done'
