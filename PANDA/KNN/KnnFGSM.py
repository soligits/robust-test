from KNN.KnnAttack import Attack
import torch
import torch.nn as nn

class FGSM_KNN(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model,  target, eps=0.007):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.target = target

    def forward(self, images, labels):
        """
        Overridden.
        """
        self.model.zero_grad()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        normal_images = images[labels == 0]
        anomaly_images = images[labels == 1]

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.MSELoss()

        images.requires_grad = True
        normal_images.requires_grad = True
        anomaly_images.requires_grad = True

        _, normal_outputs = self.model(normal_images)
        _, anomaly_outputs = self.model(anomaly_images)

        # Calculate loss
        cost = loss(normal_outputs, self.target.repeat(normal_outputs.shape[0], 1))
        # Update adversarial images
        grad = torch.autograd.grad(cost, normal_images,
                                   retain_graph=False, create_graph=False)[0]
        
        adv_images_normal = normal_images + self.eps*grad.sign()
        adv_images_normal = torch.clamp(adv_images_normal, min=0, max=1).detach()

        cost = - loss(anomaly_outputs, self.target.repeat(anomaly_outputs.shape[0], 1))

        grad = torch.autograd.grad(cost, anomaly_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images_anomaly = anomaly_images + self.eps*grad.sign()
        adv_images_anomaly = torch.clamp(adv_images_anomaly, min=0, max=1).detach()

        adv_images = torch.cat((adv_images_normal, adv_images_anomaly))
        targets = torch.cat((torch.zeros((adv_images_normal.shape[0])), torch.ones((adv_images_anomaly.shape[0]))))

        return adv_images, targets, adv_images_normal, adv_images_anomaly