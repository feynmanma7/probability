import numpy as np


def gaussian_expectation_maximum(data, K=2):
    """
    data is composed of `K` gaussian distributions.
    The likelihood function of data is,
    P(Data | \theta) = \sum_{k=1}^{K} \alpha_k * \phi(X_i|mu_k, sigma_k^2)
    s.t. \sum_k alpha_k = 1
    \phi(X_i|mu_k, sigma_k^2) = 1./sqrt(2 * pi * sigma^2) * exp{-(X_i-mu)^2/(2 * sigma^2)}


    The implicit random variables are \gamma_{j, k}, indicates that if the `j`-th record is
    generated from the `k`-th gaussian distribution.
    \gamma_{j, k} = 1, if the `j`-th record comes from `k`-th component, otherwise 0.


    The likelihood function of complete data (y_j, \gamma_j1, ..., \gamma_jK; j=1, 2, ..., N) is,
    (with i.i.d assumption)
    L(D, \gamma|\theta) = \prod_{j=1}^N \prod_{k=1}^K
        (\alpha_k * \phi(X_j | mu_k, sigma_k)) ^ \gamma_jk
    = \prod_k \prod_j {\alpha_k ^ \gamma_jk} * {\phi_k ^ \gamma_jk} # (exchange the order of product j,k)
    = \prod_k \alpha_k ^ {n_k} \prod_j \phi_k ^ \gamma_jk
    let n_k = \sum_{j=1}^N \gamma_jk.


    The log-likelihood function of complete data is,
    l(y, \gamma|\theta) = \sum_k{ n_k * log(\alpha_k) + \sum_j { \gamma_jk * log(\phi_k) } }


    E-step, compute the expectation of log-likelihood function of complete data
        with respect to the hidden variable Z (\gamma_jk), which is called Q-function.
    The Q-function is,
    Q(\theta, \theta_i) = E[log P(D, \gamma|\theta) | D, \theta^i)]
        = \sum_k { \sum_j E(\gamma_jk) * log(\alpha_k) + \sum_j E(\gamma_jk) * log(\phi_k) }
    Note: The expectation of log likelihood function of compute data w.r.t to Z(\gamma),
            only affects \gamma_jk in the Q function.

    E[\gamma_jk] = E[\gamma_jk|D, \theta]
    = P(\gamma_jk=1|D, \theta) * 1 + P(\gamma_jk=0|D, \theta) * 0 # Expectation Definition.
    = P(\gamma_jk=1|D, \theta)
    = P(\gamma_jk = 1|X_j, \theta) # i.i.d assumption.
    = \frac{ P(\gamma_jk=1, X_j|\theta) }{ P(X_j|\theta) }  # Bayes Theorem.
    = \frac{ P(\gamma_jk=1, X_j|\theta) }{ \sum_k P(\gamma_jk=1, X_j|\theta) }
    = \frac{ P(gamma_jk=1|\theta) * P(X_j|gamma_jk=1, \theta) }
        {\sum_k P(gamma_jk=1|\theta) * P(X_j|gamma_jk=1, \theta)}
    = \frac{ \alpha_k * \phi(X_j| \theta_k) }
        {\sum_k \alpha_k * \phi(X_j | \theta_k)}


    M-step, maximize the Q-function, to get \theta^{i+1}.
    \theta^{i+1} = argmax_{\theta}Q(\theta, \theta^i)
    The Q-function is concave of \theta_k(mu_k, sigma_k), compute the partial w.r.t theta,
        let it be zero.

    # mu_k = \sum_j * (E[\gamma_jk] * X_j) / \sum_j (E[\gamma_jk])

    # sigma^2_k = \sum_j * {(X_j - \mu_k)^2 * E[\gamma_jk]}
        / \sum_j E[\gamma_jk]

    # alpha_k
    s.t. \sum_k \alpha_k = 1
    Use Lagrange Multiplier, because this is a maximum problem,
    Let Lagrange(y; lambda) = y + lambda(1 - \sum_k \alpha_k),
        compute partial of Lagrange w.r.t to \alpha_k separately,
        ==>   n_k / \alpha_k - lambda = 0, k=1, 2, ..., K
        ==>   \alpha_k = n_k / lambda
        ==>   1 = \sum_k \alpha_k = \sum_k (n_k / lambda)
        ==>   lambda = \sum_k n_k, n_k = \sum_{j=1}^N \gamma_jk
        ==>   lambda = \sum_j \sum_k \gamma_jk = N

    \alpha_k = n_k / lambda = n_k / N
    """

    # Initialization
    mu = np.random.normal(170, 13, size=K)
    sigma = np.random.normal(13, 5, size=K)
    alpha = np.random.random(K)
    alpha /= np.sum(alpha)
    N = data.shape[0]
    e_gamma = np.zeros((N, K))

    epochs = 1000

    pre_Q = None
    for epoch in range(epochs):

        """
        E-step
        E[\gamma_jk] = \frac{ \alpha_k * \phi(X_j | \theta_k)}
                    {\sum_k \alpha_k * \phi(X_j | \theta_k)}
        """
        for j in range(N):
            alpha_phi = np.zeros(K)
            for k in range(K):
                alpha_phi[k] = alpha[k] * phi(data[j], mu[k], sigma[k])

            e_gamma[j, :] = alpha_phi / np.sum(alpha_phi)

        """
        M-step
        mu_k = \sum_j * (E[\gamma_jk] * X_j) / \sum_j (E[\gamma_jk])
        sigma^2_k = \sum_j * {(X_j - \mu_k)^2 * E[\gamma_jk]}
            / \sum_j E[\gamma_jk]
        \alpha_k = n_k / lambda = n_k / N
        n_k = \sum_{j=1}^N \gamma_jk
        """
        n = np.zeros(K)
        for k in range(K):
            for j in range(N):
                n[k] += e_gamma[j, k]

        for k in range(K):
            mu_k = 0
            sigma_k = 0
            for j in range(N):
                mu_k += e_gamma[j, k] * data[j]
                sigma_k += e_gamma[j, k] * (data[j] - mu[k]) ** 2

            mu[k] = mu_k / n[k]
            sigma[k] = np.sqrt(sigma_k / n[k])
            alpha[k] = n[k] / N

        Q = compute_log_likelihood(data, alpha, mu, sigma, e_gamma, n, K)
        if pre_Q is None:
            pre_Q = Q
        else:
            diff = Q - pre_Q
            pre_Q = Q
            print(epoch, diff)
            if abs(diff) < 1e-3:
                break

    return mu, sigma, alpha


def phi(X_j, mu_k, sigma_k):
    return 1. / (np.sqrt(2 * np.pi * sigma_k**2)) * np.exp(-(X_j - mu_k)**2 / (2 * sigma_k**2))


def compute_log_likelihood(data, alpha, mu, sigma, e_gamma, n, K):
    N = data.shape[0]
    l = 0
    for k in range(K):

        t = 0
        for j in range(N):
            t += e_gamma[j, k] * np.log(phi(data[j], mu[k], sigma[k]))

        l += n[k] * np.log(alpha[k]) + t

    return l


def test_expectation_maximum():
    male_mu = 180
    male_sigma = 15
    male = np.random.normal(loc=male_mu, scale=male_sigma, size=100)

    female_mu = 165
    female_sigma = 12
    female = np.random.normal(loc=female_mu, scale=female_sigma, size=90)

    data = np.concatenate([male, female], axis=0)
    np.random.shuffle(data)

    mu, sigma, alpha = gaussian_expectation_maximum(data)
    print(mu, sigma, alpha)


if __name__ == '__main__':
    test_expectation_maximum()