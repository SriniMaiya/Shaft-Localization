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
  
                 Synthetic Data:               |              Masks   
    ----
    
    <img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/Img_0001.png" width="45%" height="45%" ></img>   <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/1_annotated.jpg" width="45%" height="45%" ></img>

    <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/Img_0009.png" width="45%" height="45%" ></img>    <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/9_annotated.jpg" width="45%" height="45%"></img>

    <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/Img_00012.png" width="45%" height="45%" ></img>    <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/12_annotated.jpg" width="45%" height="45%"></img>

