# Practical Machine Learning (Sensor Data)

Coursera course: Practical Machine Learning

Useful links:

Github Page For Project: 
<https://github.com/marcomarchetti/Pml-Sensor-Data>  

gh-page branch for Project Report: <https://marcomarchetti.github.io/Pml-Sensor-Data/Practical_Machine_Learning_Sensor.html>

## Data

The training data for this project are available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>  

The test data are available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>  

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>.

## References 
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.
Read more: <http://groupware.les.inf.puc-rio.br/har#ixzz4pC5JdFEz> 

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:  
* Class A - exactly according to the specification  
* Class B - throwing the elbows to the front  
* Class C - lifting the dumbbell only halfway  
* Class D - lowering the dumbbell only halfway  
* Class E - throwing the hips to the front  

More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

## Project Goal
The goal of the  project is to predict the manner in which participant did the exercise ("classe" variable in the training set). We will use prediction model to predict 20 different test cases.
