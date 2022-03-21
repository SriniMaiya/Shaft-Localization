# Synthetic-data creation with Belnder and Blender-Python(bpy)

3D model of shafts were created with suitable texture to replicate the real-world scenario.

    Creation of Blender-Python script to automate the data creation. The images were created with:

    1. Blender Physics simulation for realistic drop of the shafts.
    2. Varying number of shafts.
    3. Variyng illumination conditions.
    4. Randomized orientations of shafts for each new image.

    The annotations were generated only for the shafts on the top layer, thus creating a
    dataset of only pickable shafts as training data.

- Below are the visualizations of some of the generated images, and the overlaid masks. 
  

<table>
<tr>
    <th> <b colspan="2" style="font-size:30px; text-align: center;"> Visualization of training data</b></th>
<tr>

<tr>
    <th><b style="font-size:20px; text-align: center;" > Synthetic image </b> </th>
    <th><b style="font-size:20px; text-align: center;"> Generated mask-data </b> </th>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_data/Img_0001.png"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_data/1_annotated.jpg"  ></img></td>
</tr>

<tr>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_data/Img_0009.png"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_data/9_annotated.jpg" ></img> </td>
</tr>

<tr>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_data/Img_00012.png"  ></img></td>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_data/12_annotated.jpg" ></img> </td>
</tr>

</table>