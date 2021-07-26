# BUET_Endgame Folder Description
The folder should contain the followings files and folders.
	Files:
		1. BUET_Endgame_report.pdf
		2. Intelligent real-time IMU sensor and camera basedanomaly detector and describer for autonomousvehicles using neural networks.pdf
		3. readme.md
		4. requirement.txt
		5. run.py
	Folders:
		1. anomaly_explainer
		2. data
		3. result
					
# Requirement
We've used Matlab(2018a) to extract all the necessary data from '.bag' files and have used python for further work.
So, having Matlab in your workstation is a must. 

For the python part ,you're requested to see the 'requirement.txt' file to know if you're up to date. To ensure you're up to date, run:
		'pip install -r requirement.txt'
		
# Get The Data 

1. Put all the all data('.bag files') defining the normal regime (if you want to retrain the model) in 'BUET_Endgame/data/train' folder to train the network.
2. Put Put all data ('.bag files') for testing purpose in 'BUET_Endgame/data/test' folder. 

		N B : This 'Get The Data' section is already done by us with the provided files. You may change it accordingly.

# Train The Network

After the successful completation of previous two sections you may train the network once again. You can 'SKIP' this portion as it's already trained.
Team BUET_Endgame encourage you to 'retrain' the model if the '.bag' files of 'BUET_Endgame/data/train' are changed.  

To retrain the network do the followings:
			1. Open command window in 'BUET_Endgame' folder.
			2. Type 'python run.py --retrain' and hit 'Enter'.



# Test The Network 
 
At this point the following files should be at the particular folders.
		1. 'test_extracted.mat' at ''BUET_Endgame/data/test' folder 
		2. 'img' folder with some subfolders named '1,2,3...' at 'BUET_Endgame/data/test' folder
		
Before going further ahead 'MAKE SURE' these files are present at the right place.

Now we're good to go for testing !! 

To test the network do the folowings:
			1. Open command window in 'BUET_Endgame' folder.
			2. Type 'Python run.py --noextract' and hit 'Enter'.

However, if you have changed the test dataset, you need to extract the data from the test bag files again. Please omit '--noextract' in that case.
			

# The Result 

A folder named 'result' can be found inside of the 'BUET_Endgame' folder.
The 'result' folder should contain four(4) subfolders.
			1. abnormality_score_&_thresholding
			2. output_video
			3. reconstructed_signals
			4. scaled_abnormality_score
			
1. abnormality_score_&_thresholding:
	Nine (9) '.png' (feature_0,feature_1 ,... ) files should be found here. Each image should've same number of subplots as the number of '.bag' files at 'BUET_Endgame/data/test' directory.
	This images show per feature the abnormality scores and corresponding thresholdings.
	Abnormality scores below the 'Green' line have been marked as 'Normal'.
	Abnormality scores over the 'Red' line have been marked as 'Highly Abnormal'
	Abnormality scores in between the 'Green' & 'Red' lines have been marked as 'Slightly Abnormal'
	
2. output_video:
	Some video files named 'OutputFinal_0.mp4,OutputFinal_1.mp4,..' should be found at 'output_video' folder. Number of '.mp4' files should be equal to the number of '.bag' file at 'BUET_Endgame/data/test' directory. 
	These video files show two images and the sensored data with anomaly score and comments( Normal/Slightly abnormal/ Highly abnormal) in between.
	For the illustration pupose of the result shown in the video files, you're requested to read 'report_BUET_Endgame.pdf' file from 'BUET_Endgame' folder where it's explained for 2 randomly picked timestamps.
	The image on the left side and the anomaly comments are shown in real-time based on the latest time-stamp.
	The image on the right side is the upcoming image for ease of visualization only (i.e. to understand the possible changes happening in-between).

3. reconstructed_signals:
	Some '.png' (reconstructed_signals_0,reconstructed_signals_1,.....) files should be found here. Number of '.png' files should be equal to the number of '.bag' files at 'BUET_Endgame/data/test' directory.
	Each '.png' file contains Nine (9) subplots.
	In each subplot three (3) signals.
			1. 'Normal'--->'.bag' files data taken from 'BUET_Endgame/data/train' in green.
			2. 'Test' --->'.bag' files data taken from 'BUET_Endgame/data/test' in orange.
			3. 'Reconstructed'---> Predicted signals by our 'LAGME' network.
			
4. scaled_abnormality_score:
	Some '.csv' files named(file_0.csv,file_1.csv,...) should be found here. Number of '.csv' files is equal to the number if '.bag' files at 'BUET_Endgame/data/test' directory.
	Each '.csv' file should contain ten(10) columns.
	First nine (9) columns represents scaled(between 0 to 1) abnormality scores per feature. Tenth column does the same for Images.
	First fifteen(15) rows of each '.csv' file should show 'nan' as no prediction can be made untill fifteenth sample from IMU is received.
	'nan' at tenth columns indecates the absence of image at that timestamp.
	Headers of the columns are:
			|| Pitch || Roll || Yaw || w_x || w_y || w_z || a_x || a_y || a_z || image ||
			
		
# For Any Query
You're welcome to contact us:
	1. subrata.biswas@ieee.org
	2. shouborno@ieee.org
	3. tisbuet@gmail.com
