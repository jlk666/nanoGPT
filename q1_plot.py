import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('losses_data_q1.csv')

# Filter the DataFrame for each interest ratio
interest_ratio_8 = df[df['Interest Ratio'] == 8]
interest_ratio_32 = df[df['Interest Ratio'] == 32]
interest_ratio_64 = df[df['Interest Ratio'] == 64]

# Plot training loss for each interest ratio
plt.figure(figsize=(10, 6))
plt.plot(interest_ratio_8['Iteration Number'], interest_ratio_8['Training Loss'], label='Interest Ratio 8', color='blue')
plt.plot(interest_ratio_32['Iteration Number'], interest_ratio_32['Training Loss'], label='Interest Ratio 32', color='green')
plt.plot(interest_ratio_64['Iteration Number'], interest_ratio_64['Training Loss'], label='Interest Ratio 64', color='red')
plt.title('Training Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('q1_train_loss.png')
plt.close()

# Plot validation loss for each interest ratio
plt.figure(figsize=(10, 6))
plt.plot(interest_ratio_8['Iteration Number'], interest_ratio_8['Validation Loss'], label='Interest Ratio 8', color='blue')
plt.plot(interest_ratio_32['Iteration Number'], interest_ratio_32['Validation Loss'], label='Interest Ratio 32', color='green')
plt.plot(interest_ratio_64['Iteration Number'], interest_ratio_64['Validation Loss'], label='Interest Ratio 64', color='red')
plt.title('Validation Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('q1_val_loss.png')
plt.close()
