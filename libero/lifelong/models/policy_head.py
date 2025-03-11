import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

class DeterministicHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2, action_squash=False):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class StochasticHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2, action_squash=False):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.action_logstd = nn.Parameter(torch.zeros(1, np.prod(output_size)))

    def forward(self, x):
        print("*****forward******")
        print(f"x shape: {x.shape}")
        print(x)
        action_mean = self.net(x)
        print(f"action_mean shape: {action_mean.shape}")
        print(action_mean)

        if action_mean.ndim == 3 and action_mean.shape[1] == 1:
            print("Squeezing time dimension")
            action_mean = action_mean.squeeze(1)

        action_logstd = self.action_logstd.expand_as(action_mean)
        # action_logstd = self.action_logstd.repeat(x.shape[0], 1) # uncomment to match batch size

        action_std = torch.exp(action_logstd)
        print("**********")
        print(action_mean.shape)
        print(action_mean)
        print(action_std.shape)
        print(action_std)
        print("**********")
        probs = D.Normal(action_mean, action_std)

        return probs

    def loss_fn(self, y_dist, target, reduction="mean"):
        log_probs = y_dist.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError
    
    # def loss_fn(self, y_dist, target, reduction="mean"):
    #     log_probs = y_dist.log_prob(target)
    #     if log_probs.ndim > 1:
    #         log_probs = log_probs.sum(dim=-1)
    #     loss = -log_probs
    #     if reduction == "mean":
    #         return loss.mean()
    #     elif reduction == "sum":
    #         return loss.sum()
    #     else:
    #         return loss


class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x):
        if x.ndim == 3:
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean"):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError
