import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('losses_data_q6.csv')

# Filter the DataFrame for each revised softmax
revised_softmax_61 = df[df['Revised softmax'] == 61]
revised_softmax_62 = df[df['Revised softmax'] == 62]

# Plot training loss for each revised softmax
plt.figure(figsize=(10, 6))
plt.plot(revised_softmax_61['Iteration Number'], revised_softmax_61['Training Loss'], label='Abs - Train Loss', color='blue')
plt.plot(revised_softmax_62['Iteration Number'], revised_softmax_62['Training Loss'], label='Vanilla Softmax - Train Loss', color='green')
plt.title('Training Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('q6_train_loss.png')
plt.close()

# Plot validation loss for each revised softmax
plt.figure(figsize=(10, 6))
plt.plot(revised_softmax_61['Iteration Number'], revised_softmax_61['Validation Loss'], label='Abs - Val Loss', color='blue')
plt.plot(revised_softmax_62['Iteration Number'], revised_softmax_62['Validation Loss'], label='Vanilla Softmax - Val Loss', color='green')
plt.title('Validation Loss vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('q6_val_loss.png')
plt.close()
