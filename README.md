# Shaft-Localization

This is the reimplementation of my [Master's project -> (Click Me for project report)]("readme_files/../readme_files/report/SriniPrakashMaiya__923123__ProjectReport-Final.pdf") using PyTorch.

The original project was carried out using [matterport maskrcnn]("https://github.com/matterport/Mask_RCNN).
## Aim:


The aim of the project is to localize the centre and orientation of pickable objects from a bin of randomly oriented shafts as in the images below. 

The shiny, textureless surface of the shafts set an additional hurdle as the traditional image processing techniques fail to preform an effective edge detection or pattern matching as the surface texture induces false edges.

<table>
<tr>
    <th><b style="font-size:20px; text-align: center;" > Original Image </b> </th>
    <th><b style="font-size:20px; text-align: center;"> Predictions of pickable shafts </b> </th>
</tr>

<tr>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006.bmp"  width="100%"></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006_op.bmp" width="100%"></img> </td>
</tr>

<tr>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0004.bmp"  width="100%"></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0004_op.bmp" width="100%"></img> </td>
</tr>

</table>


The [instance segmentation model](https://arxiv.org/abs/1703.06870) is purely trained on synthetically generated data, which closely replicates the real world scenario. The synthetic data and annotations are generated using [Blender](https://www.blender.org/) and the workflow of image and annotation generation are automated using [Blender-Python](https://docs.blender.org/api/current/info_overview.html)


The project proposes a deep-learning methodology to solve the problem of [Bin-Picking](https://www.ipa.fraunhofer.de/en/expertise/robot-and-assistive-systems/intralogistics-and-material-flow/separation-processes-using-robots-bin-picking.html) in industries, a core problem in the computer vision domain. 

## Workflow: 
 
<p align="center">
  <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/Images/workflow.png" />
</p>

1. #### Synthetic data and annotation generation by Blender
   ----
    
    A brief visualization of synthetic images and annotations can be seen [here](/readme_files/Synthetic_Data.md). For a detailed explanation can be found in the [project report]("readme_files/../readme_files/report/SriniPrakashMaiya__923123__ProjectReport-Final.pdf").

2. #### Training of Mask-RCNN model on the generated synthetic data 
   ----
   
   The training code, custom MaskRCNN Model creation methods are explained [here](/readme_files/training.md).

3. #### Prediction on Synthetic images
   ----
   
   The prediction results of synthetic images are visualized in this [readme file](readme_files/prediction_syn.md).

   
    

