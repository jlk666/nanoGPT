import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('losses_data_q2.csv')

# Filter the DataFrame for each sliding window size
window_size_3 = df[df['Sliding window size'] == 3]
window_size_10 = df[df['Sliding window size'] == 10]
window_size_100 = df[df['Sliding window size'] == 100]

# Plot training loss for each sliding window size
plt.figure(figsize=(10, 6))
plt.plot(window_size_3['Iteration Number'], window_size_3['Training Loss'], label='Window Size 3 - Train Loss', color='blue')
plt.plot(window_size_10['Iteration Number'], window_size_10['Training Loss'], label='Window Size 10 - Train Loss', color='green')
plt.plot(window_size_100['Iteration Number'], window_size_100['Training Loss'], label='Window Size 100 - Train Loss', color='red')
plt.title('Training Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('q2_train_loss.png')
plt.close()

# Plot validation loss for each sliding window size
plt.figure(figsize=(10, 6))
plt.plot(window_size_3['Iteration Number'], window_size_3['Validation Loss'], label='Window Size 3 - Val Loss', color='blue')
plt.plot(window_size_10['Iteration Number'], window_size_10['Validation Loss'], label='Window Size 10 - Val Loss', color='green')
plt.plot(window_size_100['Iteration Number'], window_size_100['Validation Loss'], label='Window Size 100 - Val Loss', color='red')
plt.title('Validation Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('q2_val_loss.png')
plt.close()
