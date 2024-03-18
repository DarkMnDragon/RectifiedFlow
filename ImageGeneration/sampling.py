"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

import matplotlib.pyplot as plt

import torchvision
from tqdm import tqdm


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.
  
    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
  
    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method  # == "rectified_flow"
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'rectified_flow':
        sampling_fn = get_rectified_flow_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler,
                                                 device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


def get_rectified_flow_sampler(sde, shape, inverse_scaler, device='cuda'):
    """
    Get rectified flow sampler
  
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def euler_sampler(model, z=None):
        """
        The probability flow ODE sampler with simple Euler discretization.
    
        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                # torch.save(z0, f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_0.pt')
                z0 = torch.load(
                    '/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_0.pt')
                x = z0.detach().clone()
            else:
                x = z

            model_fn = mutils.get_model_fn(model, train=False)

            ### Uniform
            N = 2 * sde.sample_N
            dt = 1. / N
            eps = 1e-3  # default: 1e-3
            for i in range(N):
                num_t = i / N * (sde.T - eps) + eps
                print(f"Simple Euler, step = {i}, time = {num_t:.5f}, z_{i + 1}=z_{i}+f(z_{i})*dt")

                t = torch.ones(shape[0], device=device) * num_t
                pred = model_fn(x, t * 999)  ### Copy from models/utils.py 

                x = x.detach().clone() + pred * dt
                if (N - i <= 10 or i <= 10):
                    torch.save(x,
                               f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i + 1}_euler.pt')

            euler_reversal(model, x)

            x = inverse_scaler(x)
            nfe = N
            return x, nfe

    def euler_reversal(model, z1=None):
        """
        The reversed euler to recover z_0 and z_1/2N.
    
        Args:
          model: A velocity model.
          z_1: The target distribution sample (image)
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            assert z1 is not None, "z is none!"
            z = z1.detach().clone()

            model_fn = mutils.get_model_fn(model, train=False)

            ### Uniform
            N = 2 * sde.sample_N
            dt = 1. / N
            eps = 1e-3  # default: 1e-3

            t = torch.ones(shape[0], device=device) + eps

            for i in range(N, 0, -1):
                # NOTE: RL: What's the effect of the eps here ??
                num_t = i / N * (sde.T - eps) + eps
                # num_t = i / N
                print(f"Reversed Euler, step = {i}, time = {num_t:.5f}, z_{i - 1}=z_{i}-f(z_{i})*dt")
                t.fill_(num_t)
                pred = model_fn(z, t * 999)  ### Copy from models/utils.py 
                z = z - pred * dt
                if (N - i <= 10 or i <= 10):
                    torch.save(z,
                               f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i - 1}_rev_euler.pt')

            euler_reconstruction(model, z)

    def euler_reconstruction(model, z0_rev=None):
        """
        The probability flow ODE sampler with simple Euler discretization.
    
        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Reversed z0
            assert z0_rev is not None, "z0_rev is NONE!"
            x = z0_rev.detach().clone()

            model_fn = mutils.get_model_fn(model, train=False)

            ### Uniform
            N = 2 * sde.sample_N
            dt = 1. / N
            eps = 1e-3  # default: 1e-3
            for i in range(N):
                num_t = i / N * (sde.T - eps) + eps
                print(f"Euler Reconstruction, step = {i}, time = {num_t:.5f}, z_{i + 1}=z_{i}+f(z_{i})*dt")

                t = torch.ones(shape[0], device=device) * num_t
                pred = model_fn(x, t * 999)  ### Copy from models/utils.py 

                x = x.detach().clone() + pred * dt
                if (N - i <= 10 or i <= 10):
                    torch.save(x,
                               f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i + 1}_rec_euler.pt')

    def leapfrog_sampler(model, z=None):
        """The probability flow ODE sampler with leapfrog method.
    
        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # NOTE: shape = [n, c, h, w]  e.g. [64, 3, 256, 256]
                # z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                # torch.save(z0, '/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_0.pt')
                z0 = torch.load(
                    '/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_0.pt')
                z_prev = z0.detach().clone()
            else:
                z_prev = z

            model_fn = mutils.get_model_fn(model, train=False)

            ### Uniform
            dt = 1. / (2 * sde.sample_N)
            eps = 1e-3  # default: 1e-3

            t = torch.zeros(shape[0], device=device) + eps
            pred = model_fn(z_prev, t * 999)
            z = z_prev.detach().clone() + pred * dt
            torch.save(z,
                       '/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_1_leapfrog.pt')

            for i in range(1, 2 * sde.sample_N):
                # NOTE: RL: What's the effect of the eps here ??
                num_t = i / (2 * sde.sample_N) * (sde.T - eps) + eps
                print(f"Leapfrog, step = {i}, time = {num_t:.5f}, z_{i + 1}=z_{i - 1}+f(z_{i})*dt")
                t.fill_(num_t)
                pred = model_fn(z, t * 999)  ### Copy from models/utils.py 
                z_next = z_prev + pred * 2. * dt
                if (2 * sde.sample_N - i <= 10 or i <= 10):
                    torch.save(z_next,
                               f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i + 1}_leapfrog.pt')
                z_prev, z = z, z_next

            # z = torch.load('/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_1000_leapfrog.pt')
            # z_prev = torch.load('/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_999_leapfrog.pt')

            leapfrog_reversal(model, z, z_prev)

            x = inverse_scaler(z)
            nfe = 2 * sde.sample_N

            return x, nfe

    def leapfrog_reversal(model, z1=None, z_1_1_2N=None):
        """
        The reversed leapfrog to recover z_0 and z_1/2N.
    
        Args:
          model: A velocity model.
          z_1: The target distribution samples
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            assert z1 is not None and z_1_1_2N is not None, "z is none!"
            z_prev = z1.detach().clone()
            device = z1.device

            model_fn = mutils.get_model_fn(model, train=False)

            ### Uniform
            dt = 1. / sde.sample_N
            eps = 1e-3  # default: 1e-3
            t = torch.ones(shape[0], device=device)

            use_rk45_initialize = False
            use_euler_initialize = True
            use_exact_initialize = False

            if (use_euler_initialize):  # Method 1: Single Half Step Euler
                pred = model_fn(z_prev, t * 999)
                z = z_prev.detach().clone() - pred * dt / 2

            if (use_exact_initialize):  # Method 2: Exact z_{1-1/2N}
                z = z_1_1_2N.detach().clone()

                # Method 3: 下面尝试还原更精确的 z_{1-1/2N}
            if (use_rk45_initialize):
                img = z1.detach().clone()

                rtol = atol = 1e-8
                method = 'RK45'
                eps = 1e-3

                def ode_func(t, x):
                    x = from_flattened_numpy(x, img.shape).to(device).type(torch.float32)
                    vec_t = torch.ones(img.shape[0], device=x.device) * t
                    drift = model_fn(x, vec_t * 999)
                    return to_flattened_numpy(drift)

                # Initial sample
                x = img

                # NOTE: 实验添加 eps 的区别
                # 待会看一下递推的具体时间取得什么数值
                solution = integrate.solve_ivp(ode_func,
                                               (1., (2 * sde.sample_N - 1) / (2 * sde.sample_N) * (sde.T - eps) + eps),
                                               to_flattened_numpy(x),
                                               rtol=rtol, atol=atol, method=method)
                nfe = solution.nfev  # Number of Function Evaluations
                num_time_steps = len(solution.t) - 1  # Number of time steps
                print(f"Embed to First Reverse in Leapfrog, nfe = {nfe}, num_time_steps = {num_time_steps}")
                print("Time Steps:", solution.t)

                z = torch.tensor(solution.y[:, -1]).reshape(img.shape).to(device).type(torch.float32)

                torch.save(z,
                           f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{2 * sde.sample_N - 1}_rev_rk45.pt')

            for i in range(2 * sde.sample_N - 1, 0, -1):
                # NOTE: RL: What's the effect of the eps here ??
                num_t = i / (2 * sde.sample_N) * (sde.T - eps) + eps
                print(f"Reversed Leapfrog, step = {i}, time = {num_t:.5f}, z_{i - 1}=z_{i + 1}-f(z_{i})*dt")
                t.fill_(num_t)
                pred = model_fn(z, t * 999)  ### Copy from models/utils.py 
                z_next = z_prev - pred * dt
                # z = z + 0.1 * (z_next - 2 * z + z_prev)

                if (2 * sde.sample_N - i <= 10 or i <= 10):
                    torch.save(z_next,
                               f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i - 1}_rev_leapfrog.pt')

                z_prev, z = z, z_next

            leapfrog_reconstruction(model, z0=z, z_1_2N=z_prev)

    def leapfrog_reconstruction(model, z0=None, z_1_2N=None):
        """
        Use the stored z0 and z_{1/2N} to reconstruct z1
    
        Args:
          model: A velocity model
          z0: The reversed latent code
          z_1_2N: The stored first step
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            assert z0 is not None and z_1_2N is not None, "z is none!"

            z_prev = z0.detach().clone()

            model_fn = mutils.get_model_fn(model, train=False)

            dt = 1. / sde.sample_N
            eps = 1e-3  # default: 1e-3
            t = torch.zeros(shape[0], device=device) + eps
            pred = model_fn(z_prev, t * 999)

            # NOTE: choose one of initialization below
            z = z_1_2N.detach().clone()
            # z = z_prev.detach().clone() + pred * dt / 2

            torch.save(z,
                       '/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_1_rec_leapfrog.pt')

            for i in range(1, 2 * sde.sample_N):
                # NOTE: RL: What's the effect of the eps here ??
                num_t = i / (2 * sde.sample_N) * (sde.T - eps) + eps
                print(f"Leapfrog Reconstruction, step = {i}, time = {num_t:.5f}, z_{i + 1}=z_{i - 1}+f(z_{i})*dt")
                t.fill_(num_t)
                pred = model_fn(z, t * 999)  ### Copy from models/utils.py 
                z_next = z_prev + pred * dt
                if (2 * sde.sample_N - i <= 10 or i <= 10):
                    torch.save(z_next,
                               f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_{i + 1}_rec_leapfrog.pt')
                z_prev, z = z, z_next

    def rk45_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.
    
        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            rtol = atol = sde.ode_tol
            method = 'RK45'
            eps = 1e-3

            # Initial sample
            if z is None:
                # z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                z0 = torch.load(
                    '/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_0.pt')
                x = z0.detach().clone()
            else:
                x = z

            model_fn = mutils.get_model_fn(model, train=False)

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = model_fn(x, vec_t * 999)

                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            print(f"Latent to Img, nfe = {nfe}, num_time_steps = {len(solution.t) - 1}")
            print("Time Steps:", solution.t)
            torch.save(x, f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_rk45.pt')

            rk45_reversal(model_fn, x)

            x = inverse_scaler(x)

            return x, nfe

    @torch.no_grad()
    def rk45_reversal(model_fn, img):
        device = img.device

        def ode_func(t, x):
            x = from_flattened_numpy(x, img.shape).to(device).type(torch.float32)
            vec_t = torch.ones(img.shape[0], device=x.device) * t
            drift = model_fn(x, vec_t * 999)
            return to_flattened_numpy(drift)

        rtol = atol = 1e-5
        method = 'RK45'
        eps = 1e-3

        # Initial sample
        x = img.detach().clone()

        # NOTE: 实验添加 eps
        solution = integrate.solve_ivp(ode_func, (1., eps), to_flattened_numpy(x),
                                       rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev  # Number of Function Evaluations
        num_time_steps = len(solution.t) - 1  # Number of time steps
        print(f"Embed to Latent, nfe = {nfe}, num_time_steps = {num_time_steps}")
        print("Time Steps:", solution.t)

        x = torch.tensor(solution.y[:, -1]).reshape(img.shape).to(device).type(torch.float32)

        torch.save(x, f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_rev_rk45.pt')

        rk45_reconstruction(model_fn, x)

    def rk45_reconstruction(model_fn, z0_inv=None):
        with torch.no_grad():
            rtol = atol = sde.ode_tol
            method = 'RK45'
            eps = 1e-3

            assert z0_inv is not None, "z0_inv is None!"

            x = z0_inv.detach().clone()

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = model_fn(x, vec_t * 999)

                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            print(f"Reconstruction, nfe = {nfe}, num_time_steps = {len(solution.t) - 1}")
            print("Time Steps:", solution.t)
            torch.save(x,
                       f'/home/lrl/rectifiedflow/RectifiedFlow/ImageGeneration/logs/celebahq/eval/ckpt_10/z_rec_rk45.pt')

    print('Type of Sampler:', sde.use_ode_sampler)
    if sde.use_ode_sampler == 'rk45':
        return rk45_sampler
    elif sde.use_ode_sampler == 'euler':
        return euler_sampler
    elif sde.use_ode_sampler == 'leapfrog':
        return leapfrog_sampler
    else:
        assert False, 'Not Implemented!'
