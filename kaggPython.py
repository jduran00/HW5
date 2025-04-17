import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from PIL import Image 
import os 

# Read files and add paths to each image
df = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')
#df = df.head(1000)
df['path'] = df['id'].apply(lambda x: f"histopathologic-cancer-detection/train/{x}.tif")

# Get paths to tests .tifs
test = "histopathologic-cancer-detection/test"
test_files = [f for f in os.listdir(test)]
test_paths = [os.path.join(test,f) for f in test_files]
test_df = pd.DataFrame({'path': test_paths})
#test_df = test_df.head(1000)
test_df['id'] = test_df['path'].apply(lambda x: os.path.basename(x).replace('.tif', ''))


# Drop rows with missing parts
df = df.dropna(subset=['label', 'path'])

# Pie chart of train_label.csv results
df['label'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Cancer', 'Cancer'], startangle=90)
plt.title("Class Balance in Train Labels")
plt.ylabel('')
plt.show()

# Load images into NumPy array
def load_image(path, size=(64,64)):
    img = Image.open(path)
    img = np.array(img) / 255.0        
    return img.flatten()
df['image_array'] = df['path'].apply(load_image) # Apply to all rows

# Prep X and y 
X = np.stack(df['image_array'].values)  # shape: (num_samples, features)
y = df['label'].values                  # shape: (num_samples,)


# Train model with scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# Test accuracy on train
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))

##changed to batch testing since I received a memory...
# ..error tyring to get full sample for my submission.csv

batch_size = 1000  # Tune based on your memory
predictions = []
ids = []
for i in range(0, len(test_paths), batch_size):
    batch_paths = test_paths[i:i+batch_size]
    batch_imgs = [load_image(p) for p in batch_paths]
    X_batch = np.stack(batch_imgs)
    test_preds = model.predict(X_batch)
    predictions.extend(test_preds)
    # Make sure ids remain
    batch_ids = [os.path.basename(p).replace('.tif', '') for p in batch_paths]
    ids.extend(batch_ids)

# Save predictions to submissions.csv
submission = pd.DataFrame({
    'id': ids,
    'label': predictions
})
submission.to_csv('submission.csv', index=False)
print("Predictions saved")


# Apply model to test paths
#test_df['image_array'] = test_df['path'].apply(load_image)
#X_test = np.stack(test_df['image_array'].values)
#test_preds = model.predict(X_test)



#Save predictions to submissions.csv
#test_df['label'] = test_preds
#submission = test_df[['id', 'label']]
#submission.to_csv('submission.csv', index=False)

