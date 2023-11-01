# Face_Forensics_videos_preprocessing_with_multiprocessing
This repository contains an optimized code for preprocessing videos from the FaceForensics++ (F++) dataset. By leveraging multiprocessing, the code efficiently extracts target faces from videos using corresponding masked videos as guides.

In the left image, there are two faces present within a video frame. However, by utilizing the masks provided by the FaceForensics++ dataset (as shown on the right), we can precisely extract the target face.
![unmasked_frame_vs_masked_frame](https://github.com/noureldinalaa/Face_Forensics_videos_preprocessing_with_multiprocessing/blob/main/mask_vs_unmasked.PNG)


Please follow the steps provided in the [FaceForensics++ repository](https://github.com/ondyari/FaceForensics/tree/master/dataset) to obtain the videos and their corresponding masks. Note that you should download only the manipulated deepfake masked videos. These masked videos can be used to detect the target face in all other videos.

## Instructions:
Please install the required packages using the requirement file :

```
    pip install -r requirements.txt
```

To run the code:

1. Specify the input video folder. Choose from one of the following folders: `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, or `NeuralTextures`.
2. Provide the path to the deepfake mask video.
3. Finally, specify the output path where the results will be saved

Example:
```
    python .\main.py --input_folder "\archive\manipulated_sequences\NeuralTextures" --masked_input_folder "masks\manipulated_sequences\Deepfakes\masks\videos" --output_folder_base "\test"

```

**Please be aware that you may need to adjust the batch size based on your computer's resources. The current batch size is set to 12.**
