from KNN.KnnAttack import Attack
import torch
import torch.nn as nn
import gc
import faiss
import numpy as np

class PGD_KNN_ADVANCED(Attack):
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
        - images: :math:(N, C, H, W) where N = number of batches, C = number of channels,        H = height and W = width. It must have a range [0, 1].
        - labels: :math:(N) where each value :math:y_i is :math:0 \leq y_i \leq number of labels.
        - output: :math:(N, C, H, W).
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, train_embeddings, k = 2, eps=0.3,
                 alpha=2/255, steps=40, random_start=True, 
                 randomized_smoothing=False, sigma=0.1, n=5):
        super().__init__("PGD", model)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.train_embeddings = train_embeddings
        self.k = k
        self.randomized_smoothing = randomized_smoothing
        self.sigma = sigma
        self.n = n

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        normal_images = images[labels == 0]
        anomaly_images = images[labels == 1]

        normal_images_np = self.train_embeddings.reshape(self.train_embeddings.shape[0], -1)
        index = faiss.IndexFlatL2(normal_images_np.shape[1])
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(normal_images_np.astype(np.float32))

        images.requires_grad = True

        def adv_attack(target_images, attack_anomaly):
          target_images.requires_grad = True
          
          adv_images = target_images.clone().detach()

          if adv_images.numel() == 0:
            return adv_images

          if self.random_start:
              # Starting at a uniformly random point

              adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
              adv_images = torch.clamp(adv_images, min=0, max=1).detach()

          # print(adv_images.shape)
          for _ in range(self.steps):
              adv_images.requires_grad = True

              outputs = None
              if self.randomized_smoothing:
                  augmented_images = adv_images.repeat(self.n, 1, 1, 1)
                  noise = torch.randn_like(augmented_images) * self.sigma
                  augmented_images = augmented_images + noise
                  augmented_images = augmented_images.clamp(0, 1)
                  outputs = self.model(augmented_images)
              else:
                  outputs = self.model(adv_images)

          
              adv_images_np = outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1)
              _, indices = index.search(adv_images_np.astype(np.float32), self.k)

              # Compute the distance to the K-nearest neighbors
              knn_distances_list = [torch.norm(outputs[i] - torch.tensor(self.train_embeddings[indices[i]], device=self.device), dim=1, keepdim=True) for i in range(outputs.shape[0])]
              knn_distances = torch.cat(knn_distances_list, dim=1).to(self.device)
              if self.randomized_smoothing:
                  knn_distances = knn_distances.view(self.k, self.n, -1)
                  knn_distances = knn_distances.mean(1)

              # Compute the cost function as the mean of the distances to the K-nearest neighbors
              cost = knn_distances.mean()
              
              if attack_anomaly:
                cost = -cost

              # Update adversarial images
              grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

              adv_images = adv_images.detach() + self.alpha*grad.sign()
              delta = torch.clamp(adv_images - target_images, min=-self.eps, max=self.eps)
              adv_images = torch.clamp(target_images + delta, min=0, max=1).detach()
              del grad

          return adv_images
        
        adv_normal_images = adv_attack(normal_images, False)
        adv_anomaly_images = adv_attack(anomaly_images, True)

        if normal_images.numel():
          adv_images = adv_normal_images
          adv_images_in = adv_normal_images
          adv_images_out = normal_images
          targets = torch.ones(adv_normal_images.shape[0])
          if anomaly_images.numel():
            adv_images = torch.cat((adv_images, adv_anomaly_images))
            adv_images_in = torch.cat((adv_images_in, anomaly_images))
            adv_images_out = torch.cat((adv_images_out, adv_anomaly_images))
            targets = torch.cat((torch.zeros((adv_normal_images.shape[0])), torch.ones((adv_anomaly_images.shape[0]))))
        else:
          adv_images = adv_anomaly_images
          adv_images_in = anomaly_images
          adv_images_out = adv_anomaly_images
          targets = torch.ones(adv_anomaly_images.shape[0])

        torch.cuda.empty_cache()

        return adv_images, adv_images_in, adv_images_out, targets
