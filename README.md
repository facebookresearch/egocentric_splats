# Photoreal Scene Reconstruction from an Egocentric Device [SIGGRAPH 2025]

<a href="#" target="_blank" rel="noopener noreferrer">
  <img src="#" alt="Paper PDF">
</a>
<a href="#"><img src="#" alt="arXiv"></a>

```
@inproceedings{lv2025egosplats,
    title={Photoreal Scene Reconstruction from an Egocentric Device},
    author={Lv, Zhaoyang and Monge, Maurizio and Chen, Ka and Zhu, Yufeng and Goesele, Michael and Engel, Jakob and Dong, Zhao and Newcombe, Richard},
    booktitle={ACM SIGGRAPH}
    year={2025}
}
```

## Overview


## Quick start

```
conda create -n ego_splats python=3.10
conda activate ego_splats
pip install -r requirements.txt
```

## Run on a Project Aria recording

### Run data preprocess
The following python script is transform the Aria VRS file into raw individual streams with rectification.

``` bash
# the full command
python scripts/extract_aria_vrs.py \
    --input_root $DATA_INPUT_DIR \
    --output_root $DATA_PROCESSED_DIR \
    --vrs_file $vrs_file \
    --rectified_rgb_focal 1200 \
    --rectified_rgb_size 2000 \
    --rectified_monochrome_focal 180 --rectified_monochrome_height 480 \
    --online_calib_file $VIBA_DIR/$vrs_file/online_calibration.jsonl \
    --trajectory_file $VIBA_DIR/$vrs_file/closed_loop_trajectory.csv \
    --semi_dense_points_file $VIBA_DIR/$vrs_file/semidense_points.csv.gz \
    --semi_dense_observation_file $VIBA_DIR/$vrs_file/semidense_observations.csv.gz \
    --visualize
    #--aws_cluster \
    # --use_factory_calib
    # --overwrite

# To check an example script, check
bash scripts/bash_local/run_aria_preprocessing.sh

```
In addition to the provided input path, it also assumes the location toolbox data for the corresponding VRS file being generated at the following path which includes
``` bash
$VIBA_DIR/$vrs_file
- closed_loop_trajectory.csv
- semidense_points.csv.gz
- semidense_observations.csv.gz
- online_calibration.jsonl
```

With "--visualize" flag on, the script will stream the processed output in each stage to a rerun visualizer.

## File format of the preprocessed scripts

Running the script will automatically generate a nerfstudio like transform json file for each of the raw camera stream, which in the following data structure in the output folder:
```
$DATA_PROCESSED_DIR
- camera-rgb-images
- camera-rgb-transforms.json
- camera-slam-left-images
- camera-slam-left-transforms.json
- camera-slam-right-images
- camera-slam-right-transforms.json
```
where 'camera-rgb-images', 'camera-slam-left-images', 'camera-slam-right-images' are the raw images folders for each, and each transform json '*-transforms.json' file, it encodes the meta data for **each camera raw data** as following:
``` json
    "camera_model": "FISHEYE624",
    "frames": [
        {
            "fl_x": 1216.7952880859375,
            "fl_y": 1216.7952880859375,
            "cx": 1459.875,
            "cy": 1441.5284423828125,
            # The fisheye distortion parameters
            "distortion_params": [...],
            "w": 2880,
            "h": 2880,
            "file_path": "camera-rgb-images/camera-rgb_9188421039900.0.png",
            "camera_modality": "rgb",
            # 4x4 matrix camera2world. For rolling-shuter (RGB) camera, this is the transformation corrseponds to the center row
            "transform_matrix": [[...]],
            # 4x4 camera2world. This is the first row pose of a rolling shutter camera.
            "transform_matrix_read_start": [[...]],
            # 4x4 camera2world. This is the last row pose of a rolling shutter camera.
            "transform_matrix_read_end": [[...]],
            # 4x4 matrix camera2world. This is the pose of the center-row rolling shutter camera after the full exposure time.
            "transform_matrix_read_center_exposure_end": [[...]],
            "device_linear_velocity": [...],
            "device_angular_velocity": [...],
            # The Aria sensor timestamp (in nanosecond)
            "timestamp": 9188421039900.0,
            # the exposure time in seconds
            "exposure_duration_s": 0.001464,
            # the sensor analog gain
            "gain": 3.002932548522949,
            "camera_name": "camera-rgb"
        }
        # all the rest frame list in the same format
        # ......
        # ......
    ],
    "camera_label": "camera-rgb",
    # The 4x4 transform matrix (camera2world) of device CPF coordinate.
    "transform_cpf": [[]],
```

### The raw data pre processing script will do the following work under the hood:
* Read the correct (online-calibrated) camera intrinsics.
* Estimate the correct timestamp for each camera stream, the extrinsics and transform that into the camera to world transform matrix.
* For a rolling shutter (RGB) camera, we will generate the transformation matrix at the start, center, and end time respectively. This could be the sufficient information needed to compensate the rolling shutter camera in downstream applications.
* Acquire corresponding device linear&angular velocity according to the timestamp. This has not been tested in downstream app though.
* Read the exposure time and analogy gain for each frame.

### Use factory calibration instead of online calibration
In the script, there is an option to used factory calibration instead of online calibrated camera. We don't recommend this in the downstream reconstruction since we have concluded such online device calibration is crucial in particular using RGB camera as input. This was used when doing the ablation study.

To generate such calibration as well as the following rectification. Run with the mode "--use_factory_calib" in the [preprocessing script](#aria-preprocess-script). It will generate all the results with a post-fix called "-factory-calib" as following:
```
$DATA_PROCESSED_DIR
- camera-rgb-images-factory-calib
- camera-rgb-transforms-factory-calib.json
- camera-slam-left-images-factory-calib
- camera-slam-left-transforms-factory-calib.json
- camera-slam-right-images-factory-calib
- camera-slam-right-transforms-factory-calib.json
```

### Generate rectified images and sparse visible point cloud

In the second stage of the method, according to the provided the pinhole camera model, the script will further generate the rectified images for each stream. The pinhole camera model parameters were set in the [preprocessing script](#aria-preprocess-script) with the following config
```
--rectified_rgb_focal 1200 \
--rectified_rgb_size 2000 \
--rectified_monochrome_focal 180 \
--rectified_monochrome_height 480 \
```
which set the focal and image size for rectified image streams. Note for RGB camera, there are two image resolution options, the above focal length and image size are for the full image resolution size (2880x2880). For half resolutoin RGB image (1408x1408), a reasonable estimate to retain the original FoV and size of the image can be
```
--rectified_rgb_focal 600 \
--rectified_rgb_size 1000 \
```

It will generate the following rectified camera stream output with an explanation of each file in the comments
```
- camera-rgb-rectified-1200-h1600
----images                      # The rectified RGB images
----transforms.json             # The transform json file
----vignette.png                # A rectified lens shading model for RGB image
----mask.png                    # A rectified binary mask region for the valid pixels in the lens shading model for RGB image.
----semidense_points.csv.gz     # a symbolic link to the input Aria MPS semi-dense points
- camera-slam-left-rectified-180-h480
----images                      # The rectified SLAM images
----sparse_depth                # A sparse set of depth calculated from the visible semi-dense points in each frame. Only applicable to SLAM images.
----transforms.json             # The transform json file
----vignette.png                # A rectified lens shading model for the SLAM monochrome image
----semidense_points.csv.gz     # a symbolic link to the input Aria MPS semi-dense points
- camera-slam-right-rectified-180-h480
```

### The rectification script will do the following work under the hood:
* Rectify the raw image as well as the vignette image for RGB and SLAM cameras. Note the two cameras models have different lens shading model. The RGB camera recorded at half resolution (1408x1408) is not simply a resize of full resolution image. The rectification will take care of this as well.
* For RGB image, there is a seperate rectified mask image indicating the valid pixels regions while SLAM cameras do not.
* For SLAM cameras, we also compute the a sparse depth map, given the semi-dense point cloud and semi-dense observations. It is based on reprojecting the visible tracked points in each slam view.

### Skip preprocessing a camera modality

When setting the rectified_*_focal number smaller than 0, it will skip preprocessing the target modality. For example, when chose
```
--rectified_rgb_focal -1
```
It will skip generate rectified image data for RGB stream.


### Streamlined preprocess in AWS cluster

We have prepared the scripts to streamline the jobs within AWS cluster.

For example
``` bash
# This is the input vrs directory
DATA_INPUT_DIR=/source_1a/data/DTCDataset/experimental/sun_08_06
# This is the output of preprocessed files
DATA_PROCESSED_DIR=/source_1a/data/DTCDataset/experimental/sun_08_06
# This is the vrs file to be used
vrs_file=library_max_exp_2ms_lux_800.vrs
# Run a preprocessing script
bash scripts/aws/submit_preprocess_single.sh $DATA_INPUT_DIR $DATA_PROCESSED_DIR $vrs_file
```

To run the batch processing at scale, you can find the examplar scripts at
``` bash!
bash scripts/aws/submit_preprocess_batch.sh
```
which will automatically launch multiple slurm jobs to batch process all vrs files.


## Run Gaussian-splatting algorithm

### Preprocessed dataset

We have preprocessed a few golden dataset in the above preprocessed data format. You can download them from these manifold path, and set up the following training script from these path.

#### The DTC golden birdhouse object (The object instance BirdHouse_A79785120_SimpleWoodWalls)
``` bash
# The full sequence
manifold://dtc_dataset/tree/aria_scenes/golden_test_recording

# When used in training, we used following start & end timemstamp.
scene.start_timestamp_ns=1522795043112
scene.end_timestamp_ns=1544195043612
```

#### The Aria home golden dataset (Used for the initial HDR reconstruction)
``` bash
# The full sequence
manifold://dtc_dataset/tree/aria_scenes/aria_home_hdr/dtc_jpg_10fps_e2ms_max_gain

# When used in training, we used following start & end timestmap
scene.start_timestamp_ns=134719106550
scene.end_timestamp_ns=1465291068062
```

#### The MPK large scale scene dataset (Used for evaluating large-scale scene reconstruction)
The data is not uploaded to manifold yet. (TBD) It is currenlty hosted in AWS cluster in the following path:
```
# The raw vrs location
/source_1a/data/DTC/zhaoyang/experimental/mpk_classic_06_02/
# The preprocessed files
/source_1a/data/DTCDataset/experimental/mpk_classic_06_02/
```

### Run Gaussian Splatting

### Train on a general purpose Aria scene recording

We provided a few scripts to reproduce a few common settings in our training, which we will enumerate below.

To prepare the data processing, download the Aria home golden dataset under data
``` bash
cd data
manifold getr manifold://dtc_dataset/tree/aria_scenes/aria_home_hdr
```

During training, the model wil launch an online visualizer at "http://0.0.0.0:8080". Open browser to check the reconstruction results interactively.

#### Using SLAM camera only
``` bash
# Run 3D GS reconstruction using SLAM cameras only
# --train_model: choose gsplats or 2dgs.
# --strategy: default or MCMC.
# example:
bash scripts/local/run_aria_slam_cameras_3DGS.sh --train_model gsplats --strategy default
```

#### Using RGB cameras only
``` bash
# Run 3D GS reconstruction using RGB cameras only
# --train_model: choose gsplats or 2dgs.
# --strategy: default or MCMC.
# example:
bash scripts/local/run_aria_rgb_cameras_3DGS.sh --train_model gsplats --strategy default
```

#### Using both modalities together
``` bash
# Run 3D GS reconstruction using both RGB and SLAM cameras combined
# --train_model: choose gsplats or 2dgs.
# --strategy: default or MCMC.
# example:
bash scripts/local/run_aria_all_cameras_joint_3DGS.sh --train_model 2dgs --strategy MCMC
```

#### Using both modalities but sequentially
``` bash
# has not been checked-in or tested
bash scripts/local/run_aria_all_cameras_sequential_3DGS.sh
```

### Train on Aria sequence for DTC object
To be added


### Run batched reconstruction on AWS cluster

After process all the files, an example to run the Gaussian Splatting algorithm

``` bash
project_folder=sun_08_06
vrs_name=library_full_tour_max_exp_2ms_lux_500_4K
version=1200-h2000
# check additional arguments within this script
bash scripts/aws/submit_run_gs_single.sh --project_folder "$project_folder" --vrs_name $vrs_name --rgb_version "$version"
```

## Visualize the reconstruction via interactive viewer

We provide an interactive viewer to visualize the trained models. For example, after launching the [training scripts](#Using-RGB-cameras-only), you can visualize all the models using
``` bash
python launch_viewer.py model_root=output/aria_home_hdr/
```

In default, it will show the visualizer at "http://0.0.0.0:8080". Open browser to check the results interactively.

### Port forwarding in AWS cluster

On AWS cluster, we can launch a slurm job to host the viewer
```
sbatch scripts/aws/run_viewer.sh outputs/aria/aria_sun_08_06/
```

There will be slurm log file indicating which machine the job is running on in the "DigitalTwinGen/slurm_logs/" folder. Check out the forward port and device machine name. Then on local computer, checkout

``` bash
# Template
ssh surreal -L <PORT_ON_LOCAL>:<COMPUTE_NODE_NAME>:<PORT_ON_COMPUTE>
#For example:
ssh frl-surreal-research -L 8080:cr1-p4de24xlarge-18:8080
```

Then you can open the port http://localhost:8080/ in local machine to checkout the interactive visualizer.

### Render a video from the trained model

#### Render all Aria views video from an existing reconstruction

```
ply_file=output/aria_home_test_online/dtc_jpg_10fps_e2ms_max_gain_monochrome/point_cloud/iteration_30000/point_cloud.ply
json_file=/home/zhaoyang/data/aria/scene/aria_home_05_02_2024/dtc_jpg_10fps_e2ms_max_gain/camera-rgb-rectified-1200-h1600/transforms.json

python render_lightning.py \
scene.load_ply=$ply_file \
render.render_json=$json_file
```

We can adjust the configuration to support render an interpolated frame video from an existing lower fps pose format.

```
python render_lightning.py
```

We can also overwrite the original resolution or fov from the loaded trajectory files to render a video
```
python render_lightning.py render.render_height=960 render.zoom=2.0 render.aspect_ratio=1.0
```
