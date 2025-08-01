## Network Intrusion Detection System
This project is a Network Intrusion Detection System (NIDS) that uses a deep learning model to classify network traffic as either 'normal' or 'anomaly'. The model is built with PyTorch and is deployed as an interactive web application using Gradio and Hugging Face Spaces.


🚀 Live Demo
You can access the live demo of the project on Hugging Face Spaces:
https://huggingface.co/spaces/Tesarac13/Edunet_project_intrusion-detector



📖 Dataset
The model was trained on the NSL-KDD dataset, a benchmark dataset for network intrusion detection. This dataset contains various network connection features and is labeled as either normal or an attack.
https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection 



🛠️ Technologies Used
This project utilizes the following technologies:
Python: The core programming language.
PyTorch: The deep learning framework used to build and train the neural network. 
Pandas & NumPy: For data manipulation and numerical operations. 
Scikit-learn: For data preprocessing, including scaling numerical features and one-hot encoding categorical features. 
Joblib: For saving and loading the preprocessor. 
Seaborn & Matplotlib: For data visualization, including the confusion matrix.
Gradio: To create and deploy the interactive web-based demo. 
Hugging Face Spaces: For hosting the deployed Gradio application.
Flask: A micro web framework for Python. 



🧠 Model Architecture
The core of this project is a feedforward neural network built with PyTorch. The architecture is as follows:

Input Layer: Takes in 117 features, which is the dimension of the preprocessed data (after one-hot encoding the categorical features).

Hidden Layer: A linear layer with 64 neurons, followed by a ReLU activation function and a dropout layer with a probability of 0.5 to prevent overfitting.

Output Layer: A linear layer that outputs 2 values, corresponding to the two classes: 'normal' and 'anomaly'.

The model is trained using the Adam optimizer and the cross-entropy loss function.

📈 Performance
The model achieves a high accuracy of 98% on the validation set. The classification report and confusion matrix from the training process are shown below, indicating strong performance in identifying both normal and anomalous network traffic.

Classification Report
precision	recall	f1-score	support
anomaly (0)	0.99	0.96	0.97	2349
normal (1)	0.97	0.99	0.98	2690
accuracy			0.98	5039
macro avg	0.98	0.98	0.98	5039
weighted avg	0.98	0.98	0.98	5039
