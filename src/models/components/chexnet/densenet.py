from torch import nn 
import torchvision 

""" 
Taken from open source implementation of CheXnet 
@misc{rajpurkar2017chexnet,
      title={CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning}, 
      author={Pranav Rajpurkar and Jeremy Irvin and Kaylie Zhu and Brandon Yang and Hershel Mehta and Tony Duan and Daisy Ding and Aarti Bagul and Curtis Langlotz and Katie Shpanskaya and Matthew P. Lungren and Andrew Y. Ng},
      year={2017},
      eprint={1711.05225},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Reimplementation by author Xinyu Weng, available at https://github.com/arnoweng/CheXNet
"""


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


