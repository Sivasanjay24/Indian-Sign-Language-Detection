import pickle
import numpy as np
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect the first few elements of data_dict['data']
for i, item in enumerate(data_dict['data'][:5]):
    print(f"Item {i}: Type = {type(item)}, Content = {item}")

# Convert lists to NumPy arrays
data_dict['data'] = [np.array(img) for img in data_dict['data']]

# Define the target size for all images
target_size = (100, 100)  # Example: 100x100 pixels

# Resize all images to the target size
data_dict['data'] = [resize(img, target_size) for img in data_dict['data']]

# Flatten each image into a 1D vector
data_dict['data'] = [img.flatten() for img in data_dict['data']]

# Convert to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check the shapes of data and labels
print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)