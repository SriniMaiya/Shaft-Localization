# Prediction of Synthetic and Real Images

The trained model was tested on both synthetic and real images to check the prediction quality. 

On the predicted masks, PCA was preformed to get center and orientations of individual masks. And the PCA results were written to a .txt file in descending order of prediction confidence.

## Prediction on Synthetic images
----
The trained network does an excellent job in detecting the pickable shafts, as visualized in the table below.  
<table>
<tr>
    <th colspan="3" style="font-size:30px; text-align: center;"> Prediction on Synthetic images </th>
<tr>

<tr>
    <th><b style="font-size:20px; text-align: center;" > Synthetic image </b> </th>
    <th><b style="font-size:20px; text-align: center;"> Prediction  </b> </th>
</tr>

<tr>
    <td><img src = "https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Img_0001047.png"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Results/Img_0001047_pred.png"  ></img></td>
</tr>

<tr>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Img_0001049.png"  ></img></td>
    <td><img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Results/Img_0001049_pred.png" ></img> </td>
</tr>

<tr>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Img_0001048.png"  ></img></td>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Results/Img_0001048_pred.png" ></img> </td>
</tr>

<tr>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Img_00015.png"  ></img></td>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Results/Img_00015_pred.png" ></img> </td>
</tr>

<tr>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Img_00011.png"  ></img></td>
    <td> <img src="https://github.com/SriniMaiya/Shaft-Localization/blob/main/readme_files/synthetic_test/Results/Img_00011_pred.png" ></img> </td>
</tr>
</table>


