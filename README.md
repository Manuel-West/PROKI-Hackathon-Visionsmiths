# Hackathon 2024 - Submission of Group Visionsmiths

Team members:
    - Meret Götzmann
    - Jonas Kyrion
    - Jonas Ludwig
    - Manuel Westermann

## Description
// A very brief description of your solution, e.g., five sentences //
The task is to find a good position and rotation for a robotic arm to grab a piece of cut sheet metal.
<img src="data/dummy/part_1/part_1.png" alt="An example input image" width="200px" /> 
<img src="data/dummy/part_1/visualisation_1.png" alt="An example solution" width="200px" />

### Preprocessing Part-Image & Gripper 

The sheet metal part will be fed into a preprocess pipeline containing the follwoing steps:

    1. pre-segementation using the ultalytic fast-sam

    [More info here](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/fast-sam.md)

    2. Gaussian blur 

    3. Determine dynymic thresholds for a canny detector via otsu's method
    
    4. morphological operations to close edegs and remove noise from the image 

Afterwards find the shape & the image center via the open-CV `moments` method. This will be used to apped the processed gripper to the right dimensions.



### Optimization 

TODO

## How to Run
- Navigate to the solution dierctory ../solution/

- Run the code in bash:
    ->  `python csvBuilder.py input.csv output.csv`

- An example input.csv could look like this and the default delimiter is ';'

    ```csv
    part;gripper
    ../data/dummy/part_1/part_1.png;../data/dummy/part_1/gripper_2.png
    ../data/dummy/part_2/part_2.png;../data/dummy/part_2/gripper_1.png
    ../data/Validierungsdaten/1/part_1.png;../data/Validierungsdaten/1/gripper_2.png
    ../data/Validierungsdaten/2/part_2.png;../data/Validierungsdaten/2/gripper_1.png
    ../data/Validierungsdaten/3/part_3.png;../data/Validierungsdaten/6/gripper_4.png
    ../data/Validierungsdaten/4/part_4.png;../data/Validierungsdaten/4/gripper_5.png
    ../data/Validierungsdaten/5/part_5.png;../data/Validierungsdaten/5/gripper_3.png


## ... and other things you want to tell us



## License

All resources in this repository are licensed under the MIT License. See the [LICENSE](LICENSE) for more information.

We expect you to also license your code under the MIT License when submitting it.

## Acknowledgments

<img src="doc/logos-all.png" alt="Logos" width="600px" />

This project is partially funded by the German Federal Ministry of Education and Research (BMBF) within the “The Future of Value Creation – Research on Production, Services and Work” program (funding number 02L19C150) managed by the Project Management Agency Karlsruhe (PTKA).
The authors are responsible for the content of this publication.
