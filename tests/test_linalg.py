import jax
from jax import numpy as jnp

import thermox


def test_linear_system():
    A = jnp.array([[3, 2], [2, 4.0]])
    b = jnp.array([1, 2.0])

    x = thermox.linalg.solve(A, b, num_samples=10000, dt=0.1, burnin=0)

    assert jnp.allclose(A @ x, b, atol=1e-1)


def test_linear_system_processed():
    A = jnp.array([[3, 2], [2, 4.0]])
    b = jnp.array([1, 2.0])

    x = thermox.linalg.solve(A, b, num_samples=10000, dt=0.1, burnin=0)
    x_processed = thermox.linalg.solve(
        thermox.utils.preprocess_drift_matrix(A),
        b,
        num_samples=10000,
        dt=0.1,
        burnin=0,
    )

    assert jnp.allclose(x_processed, x, atol=1e-5)


def test_linear_system_x0():
    A = jnp.array([[3, 2], [2, 4.0]])
    b = jnp.array([1, 2.0])
    x_true = jnp.linalg.solve(A, b)

    keys = jax.random.split(jax.random.PRNGKey(0), 200)

    def mean_err(x0):
        xs = jax.vmap(
            lambda k: thermox.linalg.solve(A, b, num_samples=10, dt=0.1, key=k, x0=x0)
        )(keys)
        return jnp.linalg.norm(xs.mean(0) - x_true)

    # started at the solution the estimate is unbiased; from zeros it is not
    assert mean_err(x_true) < 0.2 * mean_err(None)


def test_inv():
    A = jnp.array([[3, 2], [2, 4.0]])

    A_inv = thermox.linalg.inv(A, num_samples=10000, dt=0.1, burnin=0)

    assert jnp.allclose(A @ A_inv, jnp.eye(2), atol=1e-1)


def test_expnegm():
    A = jnp.array([[3, 2], [2, 4.0]])

    expnegA = thermox.linalg.expnegm(A, num_samples=10000, dt=0.1, burnin=0)

    assert jnp.allclose(expnegA, jax.scipy.linalg.expm(-A), atol=1e-1)


def test_expm():
    A = jnp.array([[-0.4, 0.1], [0.5, -0.3]])

    expA = thermox.linalg.expm(A, num_samples=100000, dt=0.1, burnin=0, alpha=1.0)

    assert jnp.allclose(expA, jax.scipy.linalg.expm(A), atol=1e-1)
