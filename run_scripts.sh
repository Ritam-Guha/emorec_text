#python -m emorec_text.code.training_script.lstm_training_script --device cpu --type_loss mse --lr 0.0001 --n_epochs 10
#python -m emorec_text.code.training_script.gru_training_script --device cpu --loss mse --lr 0.0001 --n_epochs 10
#python -m emorec_text.code.training_script.rnn_training_script --device cpu --loss mse --lr 0.0001 --n_epochs 10
python -m emorec_text.code.training_script.linear_classifier_training_script --device cpu --type_loss mse --lr 0.0001 --n_epochs 10