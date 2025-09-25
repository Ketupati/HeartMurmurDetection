from common import *

class MurmurOutcomeNet(nn.Module):
    def __init__(self):
        super(MurmurOutcomeNet, self).__init__()

        self.atrous_conv1 = nn.Conv1d(1, 64, kernel_size=5, dilation=2, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.atrous_conv2 = nn.Conv1d(64, 32, kernel_size=5, dilation=2, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv1d(32, 2, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.spec_conv1 = nn.Conv2d(1, 16, kernel_size=(20, 2), stride=1)
        self.spec_bn1 = nn.BatchNorm2d(16)
        self.spec_pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.spec_conv2 = nn.Conv2d(16, 8, kernel_size=(5, 5), stride=1)
        self.spec_bn2 = nn.BatchNorm2d(8)
        self.spec_pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.dropout = nn.Dropout(0.3)
        self.fc_joint = nn.LazyLinear(256)
        self.murmur_fc = nn.Linear(256, 3)
        self.outcome_fc1 = nn.Linear(256, 30)
        self.outcome_fc2 = nn.Linear(30, 1)

    def forward(self, pcg, spec, patient_metadata):
        pcg = F.elu(self.pool1(self.bn1(self.atrous_conv1(pcg))))
        pcg = F.elu(self.pool2(self.bn2(self.atrous_conv2(pcg))))
        pcg = F.elu(self.pool3(self.bn3(self.conv1(pcg))))
        pcg = torch.flatten(pcg, start_dim=1)

        spec = F.elu(self.spec_pool1(self.spec_bn1(self.spec_conv1(spec))))
        spec = F.elu(self.spec_pool2(self.spec_bn2(self.spec_conv2(spec))))
        spec = torch.flatten(spec, start_dim=1)

        x = torch.cat([pcg, spec], dim=1)
        x = self.dropout(x)
        x = torch.cat([x, patient_metadata], dim=1)
        x = F.elu(self.fc_joint(x))

        murmur_logits = self.murmur_fc(x)
        murmur_probs = F.softmax(murmur_logits, dim=1)
        outcome_hidden = F.elu(self.outcome_fc1(x))
        outcome_prob = torch.sigmoid(self.outcome_fc2(outcome_hidden))

        return murmur_probs, outcome_prob
