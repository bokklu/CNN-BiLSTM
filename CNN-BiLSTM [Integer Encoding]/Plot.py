import matplotlib.pyplot as plt

def plot_graph_training(loss_list,acc_list,epochs):
    epoch_list = list(range(epochs))
    plt.xlabel('Epoch Count')
    plt.ylabel('Range [Loss/Accuracy]')
    plt.axis([0,1000,0,1])
    plt.title('LSTM Training Statistics')
    plt.plot(epoch_list,loss_list,'r',label='Training Loss')
    plt.plot(epoch_list,acc_list,'g',label='Training Acc')
    plt.legend(loc='upper left')
    plt.show()

def plot_graph_validation(loss_list,acc_list,epochs):
    epoch_list = list(range(epochs))
    plt.xlabel('Epoch Count')
    plt.ylabel('Range [Loss/Accuracy]')
    plt.axis([0, 1000, 0, 1])
    plt.title('LSTM Validation Statistics')
    plt.plot(epoch_list, loss_list, 'r', label='Validation Loss')
    plt.plot(epoch_list, acc_list, 'g', label='Validation Acc')
    plt.legend(loc='upper left')
    plt.show()

def plot_bothAccuracies(acc_list_training,acc_list_validation,epochs):
    epoch_list = list(range(epochs))
    plt.xlabel('Epoch Count')
    plt.ylabel('Accuracy')
    plt.axis([0,1000,0,1])
    plt.title('LSTM Training and Validation Accuracies')
    plt.plot(epoch_list, acc_list_training, 'b', label='Training Acc')
    plt.plot(epoch_list, acc_list_validation, 'g', label='Validation Acc')
    plt.legend(loc='upper left')
    plt.show()

def plot_bothLosses(loss_training,loss_validation,epochs):
    epoch_list = list(range(epochs))
    plt.xlabel('Epoch Count')
    plt.ylabel('Accuracy')
    plt.axis([0, 1000, 0, 1])
    plt.title('LSTM Training and Validation Losses')
    plt.plot(epoch_list, loss_training, 'b', label='Training Loss')
    plt.plot(epoch_list, loss_validation, 'g', label='Validation Loss')
    plt.legend(loc='upper left')
    plt.show()