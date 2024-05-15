import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('losses_data_q4.csv')

# Filter the DataFrame for each register token number
register_token_5 = df[df['Register token number'] == 5]
register_token_1 = df[df['Register token number'] == 1]
register_token_0 = df[df['Register token number'] == 0]

# Plot training loss for each register token number
plt.figure(figsize=(10, 6))
plt.plot(register_token_5['Iteration Number'], register_token_5['Training Loss'], label='Register Token 5 - Train Loss', color='blue')
plt.plot(register_token_1['Iteration Number'], register_token_1['Training Loss'], label='Register Token 1 - Train Loss', color='green')
plt.plot(register_token_0['Iteration Number'], register_token_0['Training Loss'], label='Register Token 0 - Train Loss', color='red')
plt.title('Training Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('q4_train_loss.png')
plt.close()

# Plot validation loss for each register token number
plt.figure(figsize=(10, 6))
plt.plot(register_token_5['Iteration Number'], register_token_5['Validation Loss'], label='Register Token 5 - Val Loss', color='blue')
plt.plot(register_token_1['Iteration Number'], register_token_1['Validation Loss'], label='Register Token 1 - Val Loss', color='green')
plt.plot(register_token_0['Iteration Number'], register_token_0['Validation Loss'], label='Register Token 0 - Val Loss', color='red')
plt.title('Validation Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('q4_val_loss.png')
plt.close()
