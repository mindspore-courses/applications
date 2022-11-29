# cpp
cd SensReader/c++
make
./ScanNet/SensReader/c++/scens

# python
python ./SensReader/python/reader.py --filename ./data/scannet/raws/scene0000_00/scene0000_00.sens --output_path ./data/scannet/scans/scene0000_00/ --export_depth_images --export_color_images --export_poses --export_intrinsics