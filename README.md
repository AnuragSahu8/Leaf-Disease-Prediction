# web application for farmers, enabling them to swiftly identify diseases affecting their crops.
Allow the farmers to directly upload the image of the crop and get the result in the form of text based solution  and video solution.

# technology used- fastapi (framework) , deep leaning(CNN architecture ) , python 

following are the steps performed during the process:

1.Collect the dataset from kaggle and divide it into batches to enhance computational efficiency.

2.Perform preprocessing task-:
          1- Resizing
          2- Rescaling
          3- horizontal vertical flip 
          4- rotation

3.Split the dataset into train test and validation.

4.Used a multi-layer CNN(Convolutional Neural Network) architecture
          CONV2D layer 
          DROPOUT layer 
          MAXPOOLING2D layer 
          FLATTEN layer 
          Dense layer 

5.Run it for 25 epochs and plot the train val accuracy loss graph.

6.Predict the output on an unseen image with disease, confidence level.
