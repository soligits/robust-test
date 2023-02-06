from KNN.KnnAttack import Attack
import torch
import torch.nn as nn
class PGD_KNN(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, target, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.target = target

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        normal_images = images[labels == 0]
        anomaly_images = images[labels == 1]

        loss = nn.MSELoss()

        images.requires_grad = True
        normal_images.requires_grad = True
        anomaly_images.requires_grad = True

        # adv_images = images.clone().detach()
        adv_normal_images = normal_images.clone().detach()
        adv_anomaly_images = anomaly_images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point

            adv_normal_images = adv_normal_images + torch.empty_like(adv_normal_images).uniform_(-self.eps, self.eps)
            adv_normal_images = torch.clamp(adv_normal_images, min=0, max=1).detach()

            adv_anomaly_images = adv_anomaly_images + torch.empty_like(adv_anomaly_images).uniform_(-self.eps, self.eps)
            adv_anomaly_images = torch.clamp(adv_anomaly_images, min=0, max=1).detach()


        for _ in range(self.steps):
            adv_normal_images.requires_grad = True
            adv_anomaly_images.requires_grad = True

            _, outputs_normal = self.model(adv_normal_images)
            _, outputs_anomaly = self.model(adv_anomaly_images)

         
            cost = loss(outputs_normal, self.target.repeat(outputs_normal.shape[0], 1))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_normal_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_normal_images = adv_normal_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_normal_images - normal_images, min=-self.eps, max=self.eps)
            adv_normal_images = torch.clamp(normal_images + delta, min=0, max=1).detach()

            cost = -loss(outputs_anomaly, self.target.repeat(outputs_anomaly.shape[0], 1))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_anomaly_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_anomaly_images = adv_anomaly_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_anomaly_images - anomaly_images, min=-self.eps, max=self.eps)
            adv_anomaly_images = torch.clamp(anomaly_images + delta, min=0, max=1).detach()
            
        adv_images = torch.cat((adv_normal_images, adv_anomaly_images))
        targets = torch.cat((torch.zeros((adv_normal_images.shape[0])), torch.ones((adv_anomaly_images.shape[0]))))
        
        return adv_images, targets, adv_normal_images, adv_anomaly_images