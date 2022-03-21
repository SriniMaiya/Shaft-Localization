# Shaft-Localization
## Aim:

The aim of the project is to localize the centre and orientation of pickable shafts from a bin of randomly oriented shafts as in the image below.

<table>
<tr>
    <th><b style="font-size:20px; text-align: center;" > Original Image </b> </th>
    <th><b style="font-size:20px; text-align: center;"> Predictions of pickable shafts </b> </th>
</tr>

<tr>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006.bmp"  width="100%"></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006_op.bmp" width="100%"></img> </td>
<tr>

<tr>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006.bmp"  width="100%"></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0004_op.bmp" width="100%"></img> </td>
<tr>

</table>

<!-- <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006.bmp"  width="45%"></img> | <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006_op.bmp" width="45%"></img>   -->




The [instance segmentation model](https://arxiv.org/abs/1703.06870) is purely trained on synthetically generated data, which closely replicates the real world scenario. The synthetic data and annotations are generated using [Blender](https://www.blender.org/) and the workflow of image and annotation generation are automated using [Blender-Python](https://docs.blender.org/api/current/info_overview.html)


The project proposes a deep-learning methodology to solve the problem of [Bin-Picking](https://www.ipa.fraunhofer.de/en/expertise/robot-and-assistive-systems/intralogistics-and-material-flow/separation-processes-using-robots-bin-picking.html) in industries, a core problem in the computer vision domain. 

## Workflow: 
 

1. [Synthetic data and annotation generation by Blender (Click me)](/readme_files/Synthetic_Data.md)
2. [Training of Mask-RCNN model on the generated synthetic data (Click me)](/readme_files/training.md)
3. [Prediction on real images](readme_files/prediction.md)



