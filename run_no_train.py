import model as model
import Dataset as dataset
import torch
import plot
import map 

train_dataset = dataset.ECGDataset("primoecg.csv", "archive/", split='first_half')
validation_dataset = dataset.ECGDataset("primoecg.csv", "archive/", split='second_half')
test_dataset = dataset.ECGDataset("secondoecg.csv", "archive/", split='first_half')

signal_train, id_train = train_dataset[:10]
signal_test, id_test = test_dataset[:10]

input_shape = signal_train.shape[1]  # channel of ecg
output_shape = signal_train.shape[0] # number of classes as output
hidden_units = 32 

model=model.ModelV0(input_shape, hidden_units, output_shape)

with torch.inference_mode():
    test_pred_logit = model(signal_test) # logit
    test_pred = torch.softmax(test_pred_logit, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
    mapped=map.map_positions_to_ids_in_order(test_pred, id_test)
    plot.show_signal(signal=signal_test, id=id_test, pred=mapped)
    print("##################################################################\n")
    print(test_pred)
    print("\n##################################################################\n\n") 
    print("##################################################################\n")
    print(test_pred_logit)
    print("\n##################################################################\n\n") 
