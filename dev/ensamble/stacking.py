# https://www.kaggle.com/c/herbarium-2020-fgvc7/discussion/154351
KERNEL_SIZE = 9
NUM_CHANNELS = 128

class StackingCNN(nn.Module):
    def __init__(self, num_models, num_channels=NUM_CHANNELS):
        super(StackingCNN, self).__init__()
        self.base = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, num_channels,
                kernel_size=(num_models, 1),
                padding=(int((num_models - 1) / 2), 0))),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(num_channels, num_channels,
                                kernel_size=(1, KERNEL_SIZE), padding=(0, int((KERNEL_SIZE - 1) / 2)))),
            ('relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, images, labels=None):
        with autocast():
            x = self.base(images)
            x = x.permute(0, 3, 1, 2)
            x = fast_global_avg_pool_2d(x)
            if self.training:
                losses = {
                    "avg_loss": F.cross_entropy(x, labels)
                    }
                return losses
            else:
                outputs = {"logits": x}
                return outputs