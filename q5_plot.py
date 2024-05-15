import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('losses_data_q5.csv')

# Filter the DataFrame for each tested condition
condition_31 = df[df['Tested condition'] == 31]
condition_1 = df[df['Tested condition'] == 1]
condition_3 = df[df['Tested condition'] == 3]

# Plot training loss for each tested condition
plt.figure(figsize=(10, 6))
plt.plot(condition_31['Iteration Number'], condition_31['Training Loss'], label='3 sliding window size + 1 register token', color='blue')
plt.plot(condition_1['Iteration Number'], condition_1['Training Loss'], label='1 register token', color='green')
plt.plot(condition_3['Iteration Number'], condition_3['Training Loss'], label='3 sliding window size', color='red')
plt.title('Training Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('q5_train_loss.png')
plt.close()

# Plot validation loss for each tested condition
plt.figure(figsize=(10, 6))
plt.plot(condition_31['Iteration Number'], condition_31['Validation Loss'], label='3 sliding window size + 1 register token', color='blue')
plt.plot(condition_1['Iteration Number'], condition_1['Validation Loss'], label='1 register token', color='green')
plt.plot(condition_3['Iteration Number'], condition_3['Validation Loss'], label='3 sliding window size', color='red')
plt.title('Validation Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('q5_val_loss.png')
plt.close()
