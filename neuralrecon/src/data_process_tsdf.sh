# Change PATH_TO_SCANNET and OUTPUT_PATH accordingly.
# For the training/val split:
python tools/tsdf_fusion/generate_gt.py --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
# For the test split
python tools/tsdf_fusion/generate_gt.py --test --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9

# For train/val
python tools/tsdf_fusion/generate_gt.py --n_proc 2 --n_gpu 1 --save_name all_tsdf_9 --window_size 9

# For test
python tools/tsdf_fusion/generate_gt_ms.py --test --n_proc 2 --n_gpu 1 --save_name all_tsdf_9 --window_size 9