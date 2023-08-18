# Session 13 - ERA Phase I - Assignment 

1. Class accuracy is more than 75%
2. No Obj accuracy of more than 95%
3. Object Accuracy of more than 70% 
4. Training for 40 epochs
5. Implement Mosaic Augmentation only 75% of the times
6. PL Trainer with precision float16 
7. Use Gradio to create UI & build app on HuggingFace
8. Implement GradCAM
9. Add multi-resolution training [Not finished]

## Usage 
1. S13.ipynb shows the PyTorch Lightning code for training Yolov3 from scratch.
2. S13-HuggingFace.ipynb shows the gradio app tested on colab to be shifted to app.py in HuggingFace Spaces (App is here: https://huggingface.co/spaces/LN1996/S13-ERA-Phase-I-Yolov3-Pascal)

## Results
1. Logs (from PyTorch Lightning)

100%|██████████| 518/518 [01:26<00:00,  5.98it/s]
Train Metrics
Epoch: 39
Loss: 3.4298393726348877
Class Accuracy: 87.459534%
NoObject Accuracy: 98.114128%
Object Accuracy: 80.751427%
100%|██████████| 155/155 [00:27<00:00,  5.64it/s]
Test Metrics
Class Accuracy: 90.091438%
No Object Accuracy: 99.018707%
Object Accuracy: 73.546692%
100%|██████████| 155/155 [08:48<00:00,  3.41s/it]
MAP Value:  0.48722711205482483
Epoch 39: 100%|██████████| 518/518 [13:59<00:00,  1.62s/it, v_num=14, train_loss_step=2.740, train_loss_epoch=3.430]`Trainer.fit` stopped: `max_epochs=40` reached.
Epoch 39: 100%|██████████| 518/518 [14:01<00:00,  1.62s/it, v_num=14, train_loss_step=2.740, train_loss_epoch=3.430]


Contributors
-------------------------
Shashank Gupta sgupta19.ai@gmail.com

Lavanya Nemani lavanyanemani96@gmail.com