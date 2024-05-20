import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('losses_data_q3.csv')

# Filter the DataFrame for each Revised MLP
revised_mlp_false = df[df['Revised MLP'] == False]
revised_mlp_true = df[df['Revised MLP'] == True]

# Plot training loss for each Revised MLP
plt.figure(figsize=(10, 6))
plt.plot(revised_mlp_false['Iteration Number'], revised_mlp_false['Training Loss'], label='Revised MLP - False - Train Loss', color='blue')
plt.plot(revised_mlp_true['Iteration Number'], revised_mlp_true['Training Loss'], label='Revised MLP - True - Train Loss', color='green')
plt.title('Training Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('revised_mlp_train_loss.png')
plt.close()

# Plot validation loss for each Revised MLP
plt.figure(figsize=(10, 6))
plt.plot(revised_mlp_false['Iteration Number'], revised_mlp_false['Validation Loss'], label='Revised MLP - False - Val Loss', color='blue')
plt.plot(revised_mlp_true['Iteration Number'], revised_mlp_true['Validation Loss'], label='Revised MLP - True - Val Loss', color='green')
plt.title('Validation Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('revised_mlp_val_loss.png')
plt.close()
