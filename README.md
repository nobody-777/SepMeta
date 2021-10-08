# SepMeta
The PyTorch implementation of "Self-paced Meta-learning" (SepMeta). The curriculum in SepMeta is effectively integrated as a regularization term into the objective so as to the meta-learner can measure the hardness of tasks adaptively (or dynamically), according to what the model has already learned (i.e., the computed task-level losses).
![avatar](https://github.com/nobody-777/SepMeta/blob/master/framework.png)

## Prerequisites
- Python 3.5
- PyTorch >= 1.2
- TorchVision >= 0.2
- tqdm

## Dataset Preparation
### mini-ImageNet
- Training set: 64 classes (600 images per class)
- Val set: 16 classes
- Test set: 20 classes

### tiered-ImageNet
- Training set: 351 classes (600 images per class)
- Val set: 97 classes
- Test set: 160 classes

After downloading the dataset, please create a new folder named "images" under the folder "miniimagenet" or "tieredimagenet", and put all images in this folder. The provided data loader will read images from the "images" folder by default. Of course, it is also OK to change the read path. For example, for the miniimagenet dataset, please change the line 10 of "./dataloader/mini_imagenet.py" as the path of the downloaded images.

## Meta-training

### Meta-training using our Curriculum-Based Meta-learning method
Since our method is a two-layer recursive version of the designed BrotherNet module, we give the implementation of the BrotherNet in trainer_ensemble package. 
For example, you can dirrectly perform the Demo.sh  in the package to train and test a specific meta-learner. 
> Demo.sh

## Meta-test
You can using the following script to test your trained model using tasks sampled from test set.
> python trainer_ensemble/test_fsl.py   --shot 5 --eval_shot 5 --num_test_episodes 3000   --test_model .your_trained_model_path --gpu 5

## Acknowledgement
Our implementations use the source code from the following repository:
- [Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions](https://github.com/Sha-Lab/FEAT), CVPR2020

Thanks for their valuable work.

## Contact
If you have any questions about this implementation, please do not hesitate to contact with me. 


