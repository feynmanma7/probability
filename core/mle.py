import numpy as np
np.random.seed(7)


def gaussian_mle(data):
    # Probability Density Function of data, i.i.d
    # P(D | \theta) = \prod_i p(X_i | \theta)
    # P(X_i | \theta=(mu, sigma)) = 1. / (sqrt(2*pi*sigma^2)) * exp(-(X-mu)**2/(2*sigma^2))
    # Let y_i = P(X_i | mu, sigma) for short,
    # likelihood function, with i.i.d (independently identical distributed) assumption,
    #  P(D | \theta) = L(D; mu, sigma) = \prod_i^{n} y_i
    # log-likelihood function l(D; mu, sigma) = \sum_i^{n} log(y_i)
    # l() is concave w.r.t mu,  \partial_l / \partial_mu = 0 ==> mu_{MLE} = \sum_i^{n}X_i / n
    # l() is concave w.r.t sigma, \partial_l / \partial_sigma = 0 ==>
    #       sigma^2_{MLE} = \sum_i^{n} (X_i - mu) ^ 2 / n
    #   mu is unknown, use the estimated X_bar = 1/n \sum_i^{n} X_i instead.

    n = data.shape[0]
    mu_mle = np.sum(data) / n
    sigma_mle = np.sqrt(np.sum((data - mu_mle) ** 2) / n)

    # Is the estimation biased?
    # The estimated is not biased if E(estimated) = variable, else biased.
    # X_i ~ N(mu, sigma^2), i.i.d
    # === Check mu_mle
    # E(mu_mle) = 1/n * \sum_i^{n} X_i = 1/n * n * mu = mu,
    #   thus mu_mle is not biased.
    # === Check sigma_mle
    # E{sigma_mle ^ 2} = E{ 1/n * \sum (X_i - X_bar) ^ 2 }
    # = E { 1/n * \sum ((X_i - mu) - (X_bar - mu)) ^ 2}
    # = 1/n  E {\sum (X_i - mu) ^ 2
    #   - 2 * (X_i - mu) * (X_bar - mu)
    #   + (X_bar - mu) ^ 2}
    # = 1/n  E {\sum (X_i - mu) ^ 2} - 2 * (X_bar - mu)
    #   1/n * \sum E(X_i - mu) ==> (X_bar - mu)
    #   + (X_bar - mu) ^ 2}
    # = 1/n * Var(X) - E{(X_bar - mu) ^ 2}
    # = 1/n * Var(X) - Var(X_bar)
    # = 1/n * sigma^2 - (1/n^2) * n * sigma^2
    # = (n-1)/n * sigma^2
    # ! Var(X_bar) = Var(1/n * \sum X_i) = 1/n^2 \sum Var(X_i)
    #       = 1/n^2 * n * sigma^2 = 1/n * sigma^2
    # thus sigma_mle is biased (fewer), sigma_mle^2 = (n-1) / n * sigma^2
    # calibrated_sigma = sigma_mle^2 * n / (n-1)
    calibrated_sigma = np.sqrt(np.sum((data - mu_mle) ** 2) / (n - 1))

    return mu_mle, sigma_mle, calibrated_sigma


def test_mle():
    male_mu = 180
    male_sigma = 15
    male = np.random.normal(loc=male_mu, scale=male_sigma, size=100)

    female_mu = 165
    female_sigma = 12
    female = np.random.normal(loc=female_mu, scale=female_sigma, size=90)

    male_mu_mle, male_sigma_mle, male_sigma_calibrated = gaussian_mle(male)
    print(male_mu_mle, male_sigma_mle, male_sigma_calibrated)

    female_mu_mle, female_sigma_mle, female_sigma_calibrated = gaussian_mle(female)
    print(female_mu_mle, female_sigma_mle, female_sigma_calibrated)


if __name__ == '__main__':
    test_mle()