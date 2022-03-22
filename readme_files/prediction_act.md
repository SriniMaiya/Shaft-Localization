# Prediction of Synthetic and Real Images

The trained model was tested real images to check the prediction quality. 

On the predicted masks, PCA was preformed to get center and orientations of individual masks. And the PCA results were written to a txt file in descending order of prediction confidence.

## Prediction on real images
----
The trained network is able to detect atleast one pickable shaft, as visualized in the table below.   
<table>
<tr>
    <th colspan="3" style="font-size:30px; text-align: center;"> Prediction on Synthetic images </th>
<tr>

<tr>
    <th><b style="font-size:20px; text-align: center;" > Predicted Masks </b> </th>
    <th><b style="font-size:20px; text-align: center;"> PCA analysed Image  </b> </th>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0004_pred.bmp"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/004.png"  ></img></td>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_0006_pred.bmp"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/006.png"  ></img></td>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_00028_pred.bmp"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/0028.png"  ></img></td>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_000128_pred.bmp"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/00128.png"  ></img></td>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/image_000149_pred.bmp"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/00149.png"  ></img></td>
</tr>
</table>


