model="resnet"
# Change the model name to vgg,wrn_28_10,resnet50,efficientnet
# echo "python -u trainer.py  --save-dir=save$_{model} |& tee -a log_$model" 
# python -u trainer_cifar.py  --save-dir=save_${model} --epochs 200 |& tee -a log_$model # Use this line for CIFAR10/CIFAR100 dataset


echo "python -u trainer.py  --save-dir=save$_{model} |& tee -a log_$model" 
python -u trainer_efficientnet.py  --save-dir=save_${model} --epochs 200 |& tee -a log_$model # Use this line for CIFAR10/CIFAR100 dataset and efficientnet


# echo "python -u trainer_tiny_imagenet.py --save-dir=save$_{model} |& tee -a log_$model"
# python -u trainer_tiny_imagenet.py  --save-dir=save_${model} --epochs 300 |& tee -a log_$model 

