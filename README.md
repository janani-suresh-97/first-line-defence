## First line of defense: A robust first layer mitigates adversarial attacks.

### Abstract:

Adversarial training (AT) incurs significant computational overhead, leading to growing interest in designing inherently robust architectures. We demonstrate that a carefully designed first layer of the neural network can serve as an implicit adversarial noise filter (ANF). This filter is created using a combination of large kernel size, increased convolution filters, and a maxpool operation. We show that integrating this filter as the first layer in architectures such as ResNet, VGG, and EfficientNet results in adversarially robust networks. Our approach achieves higher adversarial accuracies than existing natively robust architectures without AT and is competitive with adversarial-trained architectures across a wide range of datasets. Supporting our findings, we show that (a) the decision regions for our method have better margins, (b) the visualized loss surfaces are smoother, (c) the modified peak signal-to-noise ratio (mPSNR) values at the output of the ANF are higher, (d) high-frequency components are more attenuated, and (e) architectures incorporating ANF exhibit better denoising in Gaussian noise compared to baseline architectures.

### Usage:

* Prerequisites : Python>=3.6
```bash
pip install -r requirements.txt
chmod +x run.sh or chmod +x run_imagenet.sh
```
* Modify the run.sh scripts for cifar10/cifar100/Tiny Imagenet architecture. The user can also modify the architecture in trainer_cifar.py. 

* All the scripts in the model folder is only for our prposed architecture.
