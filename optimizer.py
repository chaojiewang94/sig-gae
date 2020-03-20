import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np


def loss_function(preds, labels, mu, logvar, emb, eps, n_nodes, norm, pos_weight, warm_up):
    """
    Computing the negative ELBO for SIGVAE:
        loss = - \E_{h(z)} \log \frac{p(x|z)p(z)}{h(z)}.

    Parameters
    ----------
    preds : torch.Tensor of shape [J, N, N],
        Reconsurcted graph probability with J samples drawn from h(z).
    labels : torch.Tensor of shape [N, N],
        the ground truth connectivity between nodes in the adjacency matrix.
    mu : torch.Tensor of shape [K+J, N, zdim],
        the gaussian mean of q(z|psi).
    logvar : torch.Tensor of shape [K+J, N, zdim],
        the gaussian logvar of q(z|psi).
    emb: torch.Tensor of shape [J, N, zdim],
        the node embeddings that generate preds.
    eps: torch.Tensor of shape [J, N, zdim],
        the random noise drawn from N(0,1) to construct emb.
    n_nodes : int,
        the number of nodes in the dataset.
    norm : float,
        normalizing constant for re-balanced dataset.
    pos_weight : torch.Tensor of shape [1],
        stands for "positive weight", used for re-balancing +/- trainning samples.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # The objective is made up of 3 components,
    # loss = rec_cost + beta * (log_posterior - log_prior), where
    # rec_cost = -mean(log p(A|Z[1]), log p(A|Z[2]), ... log p(A|Z[J])),
    # log_prior = mean(log p(Z[1]), log p(Z[2]), ..., log p(Z[J])),
    # log_posterior = mean(log post[1], log post[2], ..., log post[J]), where
    # log post[j] = 1/(K+1) {q(Z[j]\psi[j]) + [q(Z[j]|psi^[1]) + ... + q(Z[j]|psi^[k])]}.
    # In practice, the loss is computed as
    # loss = rec_lost + log_posterior_ker - log_prior_ker.

    real_min = 1e-6
    J, N, Z = emb.shape  # J * N * Z
    K = mu.shape[0] - J

    preds = torch.sigmoid(preds)  # J * N * N
    sigma = torch.exp(0.5 * logvar)  # J * N * Z
    mu_mix, mu_emb = mu[:K, :], mu[K:, :]  # K * N * Z, J * N * Z
    sigma_mix, sigma_emb = sigma[:K, :], sigma[K:, :]  # K * N * Z, J * N * Z

    loss_all = 0
    for j in range(J):

        ker = 0
        emb_j = emb[j, :, :]
        sigma_emb_j = sigma_emb[j, :, :]

        # compute posterior
        for k in range(K + 1):

            if k < K:
                mu_star = mu_mix[k, :, :]
            else:
                mu_star = mu_emb[j, :, :]

            ker += torch.exp(-0.5*torch.sum((emb_j - mu_star).pow(2) / (sigma_emb_j + real_min).pow(2), dim=1))

        log_post = torch.log(ker / torch.tensor(K+1, dtype=torch.float32) + real_min) - torch.sum(torch.log(sigma_emb_j + real_min), dim=1)  # N*1
        log_post = torch.mean(log_post, dim=0)

        # compute likelihood
        pred = preds[j, :, :]
        log_lik = norm * (pos_weight * labels * torch.log(pred + real_min) + (1 - labels) * torch.log(1 - pred + real_min))  # N * N
        log_lik = log_lik.mean()

        # compute prior
        log_prior = -0.5 * torch.sum(emb_j.pow(2), dim=1)  # N *1
        log_prior = torch.mean(log_prior, dim=0)

        loss_all_j = -(log_lik + (log_prior - log_post) * warm_up / N) / torch.tensor(J, dtype=torch.float32)  # torch.log(torch.tensor(J, dtype=torch.float32))
        loss_all += loss_all_j

    return loss_all


    # preds = torch.sigmoid(preds)
    # real_min = 1e-6
    # sigma = torch.exp(0.5 * logvar)
    # J, N, Z = emb.shape  # J * N * Z
    # K = mu.shape[0] - J
    #
    # mu_mix, mu_emb = mu[:K, :], mu[K:, :]  # K * N * Z, J * N * Z
    # sigma_mix, sigma_emb = sigma[:K, :], sigma[K:, :]  # K * N * Z, J * N * Z
    #
    # loss_all = 0
    # for j in range(J):
    #
    #     ker = 0
    #     mu_emb_j = mu_emb[j, :, :]
    #     sigma_emb_j = sigma_emb[0, :, :]
    #
    #     # compute posterior
    #     for k in range(K+1):
    #
    #         if k < K:
    #             mu_star = mu_mix[k, :, :]
    #         else:
    #             mu_star = mu_emb[j, :, :]
    #
    #         ker = ker + torch.exp(-0.5 * torch.sum((mu_emb_j - mu_star).pow(2) / (sigma_emb_j + real_min).pow(2), dim=1))
    #
    #     log_post = torch.log(ker / torch.tensor(K+1) + real_min) - torch.sum(torch.log(sigma_emb_j + real_min), dim=1)
    #     log_post = torch.mean(log_post, dim=0)
    #
    #     # compute prior
    #     log_prior = -0.5 * torch.sum(sigma_emb_j.pow(2), dim=1)
    #     log_prior = torch.mean(log_prior, dim=0)
    #
    #     # compute likelihood
    #     pred = preds[j, :, :]
    #     log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))  # N * N
    #     log_lik = log_lik.mean()
    #
    #     loss_j = log_lik
    #     loss_all = loss_all + loss_j/torch.tensor(J)
    #
    # return loss_all



    # SMALL = 1e-6
    # std = torch.exp(0.5 * logvar)
    # J, N, zdim = emb.shape
    # K = mu.shape[0] - J
    #
    # mu_mix, mu_emb = mu[:K, :], mu[K:, :]  # K * N * J, J * N * J
    # std_mix, std_emb = std[:K, :], std[K:, :]  # K * N * J, J * N * J
    #
    #
    #
    #
    # # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1, 2]).mean()
    #
    #
    # # compute log_posterior
    # Z = emb.view(J, 1, N, zdim)
    # mu_mix = mu_mix.view(1, K, N, zdim)
    # std_mix = std_mix.view(1, K, N, zdim)
    # # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # # the shape of result tensor log_post_ker_JK is [J,K]
    # log_post_ker_JK = - torch.sum(
    #     0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2,-1]
    # )
    #
    # log_post_ker_JK += - torch.sum(
    #     (std_mix + SMALL).log(), dim=[-2,-1]
    # )
    #
    # # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # # the shape of result tensor log_post_ker_J is [J, 1]
    # log_post_ker_J = - torch.sum(
    #     0.5 * eps.pow(2), dim=[-2,-1]
    # )
    # log_post_ker_J += - torch.sum(
    #     (std_emb + SMALL).log(), dim = [-2,-1]
    # )
    # log_post_ker_J = log_post_ker_J.view(-1,1)
    #
    # # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1]
    # log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)
    #
    # # apply "log-mean-exp" to the above tensor
    # log_post_ker -= np.log(K + 1.)
    # log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()

    



