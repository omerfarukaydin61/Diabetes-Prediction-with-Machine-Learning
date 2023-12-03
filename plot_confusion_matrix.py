import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, title):
    plt.figure(figsize=(10,6))
    confusion_matrix[0][0], confusion_matrix[1][1] = confusion_matrix[1][1], confusion_matrix[0][0]
    confusion_matrix[0][1], confusion_matrix[1][0] = confusion_matrix[1][0], confusion_matrix[0][1]
    sns.heatmap(confusion_matrix, annot=True ,fmt='d', cmap='Blues',annot_kws={"size": 32})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

