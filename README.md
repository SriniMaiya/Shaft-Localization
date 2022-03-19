# Shaft-Localization
## Aim:

The aim of the project is to localize the centre and orientation of pickable shafts from a bin of randomly oriented shafts as in the image below.

![sample image](/readme_files/image_00031.bmp)

The [instance segmentation model](https://arxiv.org/abs/1703.06870) is purely trained on synthetically generated data, which closely replicates the real world scenario. The synthetic data and annotations are generated using [Blender](https://www.blender.org/) and the workflow of image and annotation generation are automated using [Blender-Python](https://docs.blender.org/api/current/info_overview.html)


The project proposes a deep-learning methodology to solve the problem of [Bin-Picking](https://www.ipa.fraunhofer.de/en/expertise/robot-and-assistive-systems/intralogistics-and-material-flow/separation-processes-using-robots-bin-picking.html) in industries, a core problem in the computer vision domain. 

## Workflow: 
 

1. [Synthetic data and annotation generation by Blender (Click me)](/readme_files/Synthetic_Data.md)
2. [Training of Mask-RCNN model on the generated synthetic data (Click me)](/readme_files/training.md)
3. Prediction on real images

