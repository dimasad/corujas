import attrdict
import numpy as np
import numpy.linalg
from scipy import io, interpolate
from thesis2 import codegen, kalman, mocos, symstats, vme
from thesis2.experiments import base


class SymbolicModel(base.SymbolicModel):
    '''trakSTAR path reconstruction model.'''
    
    x = ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3', 
         'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    '''State vector.'''
    
    y = ['x_meas', 'y_meas', 'z_meas',
         'q1_meas', 'q2_meas', 'q3_meas']
    '''Measurement vector.'''
    
    p = ['x_meas_std', 'y_meas_std', 'z_meas_std',
         'q1_meas_std', 'q2_meas_std', 'q3_meas_std']
    '''Parameter vector.'''
    
    s = []
    '''Exogenous signals vector.'''
    
    c = []
    '''Constants vector.'''

    def f(self, t, x, p, s, c):
        '''Drift function.'''
        a = attrdict.AttrDict(self.unpack_arguments(t=t, x=x, p=p, s=s, c=c))
        return [
            a.vx, 
            a.vy, 
            a.vz, 
            -0.5 * (a.q1 * a.wx + a.q2 * a.wy + a.q3 * a.wz),
            -0.5 * (-a.q0 * a.wx - a.q2 * a.wz + a.q3 * a.wy),
            -0.5 * (-a.q0 * a.wy + a.q1 * a.wz - a.q3 * a.wx),
            -0.5 * (-a.q0 * a.wz - a.q1 * a.wy + a.q2 * a.wx),
            0, 
            0, 
            0,
            0,
            0,
            0,
        ]

    def meas_mean(self, t, x, p, s, c):
        '''Measurement mean.'''
        a = attrdict.AttrDict(self.unpack_arguments(t=t, x=x, p=p, s=s, c=c))
        return [a.x, a.y, a.z, a.q1, a.q2, a.q3]
    
    def meas_cov(self, t, x, p, s, c):    
        '''Measurement covariance matrix.'''
        a = attrdict.AttrDict(self.unpack_arguments(t=t, x=x, p=p, s=s, c=c))
        stds = [
            a.x_meas_std, a.y_meas_std, a.z_meas_std, 
            a.q1_meas_std, a.q2_meas_std, a.q3_meas_std
        ]
        return np.diag(stds) ** 2
    
    def meas_ll(self, y, t, x, p, s, c):
        '''Measurement log-likelihood.'''
        a = attrdict.AttrDict(
            self.unpack_arguments(t=t, x=x, y=y, p=p, s=s, c=c)
        )
        return (symstats.normal_logpdf(a.x_meas, a.x, a.x_meas_std) +
                symstats.normal_logpdf(a.y_meas, a.y, a.y_meas_std) +
                symstats.normal_logpdf(a.z_meas, a.z, a.z_meas_std) +
                symstats.normal_logpdf(a.q1_meas, a.q1, a.q1_meas_std) +
                symstats.normal_logpdf(a.q2_meas, a.q2, a.q2_meas_std) +
                symstats.normal_logpdf(a.q3_meas, a.q3, a.q3_meas_std))
    
    def prior_logpdf(self, x, p, c):
        return 0


def generated_src():
    model_generator = base.ModelGenerator(SymbolicModel(), 'GeneratedModel')
    return model_generator.generate()


def print_generated_module():
    from os import path
    module_path = path.join(path.dirname(__file__), 'generated_corujas.py')
    with open(module_path, 'w') as module_file:
        module_file.write(generated_src())


try:
    from generated_corujas import GeneratedModel
except ImportError:
    context = {'__name__': __name__}
    exec(generated_src(), context)
    GeneratedModel = context['GeneratedModel']


def unwrap_quaternion(q):
    unwrapped = np.array(q)
    
    increments = np.linalg.norm(q[1:] - q[:-1], axis=1)
    jumps = np.flatnonzero(increments > 1) + 1
    for k in jumps:
        unwrapped[k:] *= -1
    
    return unwrapped


def load_data(filepath, range_):
    data = io.loadmat(filepath)
    tmeas = data['time'].flatten()[range_]
    q = data['q'][range_]
    q_unwrapped = unwrap_quaternion(q)
    
    y_dict = dict(
        x_meas=data['x'].flatten()[range_],
        y_meas=data['y'].flatten()[range_],
        z_meas=data['z'].flatten()[range_],
        q0_meas=q_unwrapped[:, 0],
        q1_meas=q_unwrapped[:, 1],
        q2_meas=q_unwrapped[:, 2],
        q3_meas=q_unwrapped[:, 3],
    )
    return tmeas, y_dict


def spline_fit(tmeas, y_dict, smoothing_factor):
    Tknot = (tmeas[1] - tmeas[0]) * smoothing_factor
    knots = np.arange(tmeas[0] + 2 * Tknot, tmeas[-1] - 2 * Tknot, Tknot)
    splines = {}
    for yname in SymbolicModel.y + ['q0_meas']:
        splines[yname] = interpolate.LSQUnivariateSpline(
            tmeas, y_dict[yname], knots, k=5
        )
    return splines


def given_params():
    return {}


def param_guess():
    return {
        'x_meas_std': 0.2, 'y_meas_std': 0.2, 'z_meas_std': 0.2, 
        'q1_meas_std': 0.0005, 'q2_meas_std': 0.0005, 'q3_meas_std': 0.0005,
    }


def estim_problem(tmeas, y, model, col_order, meas_subdivide):
    yind = meas_subdivide * np.arange(tmeas.size)
    test = np.linspace(
        tmeas[0], tmeas[-1], (tmeas.size - 1) * meas_subdivide + 1
    )
    
    collocation = mocos.LGLCollocation(col_order)
    problem = vme.Problem(model, test, y, yind, collocation, True)
    t_fine = problem.t_fine
    return problem, t_fine


def pack_x_guess(splines, t_fine):
    q0 = splines['q0_meas'](t_fine)
    q1 = splines['q1_meas'](t_fine)
    q2 = splines['q2_meas'](t_fine)
    q3 = splines['q3_meas'](t_fine)
    q0_dot = [splines['q0_meas'].derivatives(t)[1] for t in t_fine]
    q1_dot = [splines['q1_meas'].derivatives(t)[1] for t in t_fine]
    q2_dot = [splines['q2_meas'].derivatives(t)[1] for t in t_fine]    
    q3_dot = [splines['q3_meas'].derivatives(t)[1] for t in t_fine]    
    wx = 2 * (q0 * q1_dot + q3 * q2_dot - q2 * q3_dot - q1 * q0_dot)
    wy = 2 * (-q3 * q1_dot + q0 * q2_dot + q1 * q3_dot - q2 * q0_dot)
    wz = 2 * (q2 * q1_dot - q1 * q2_dot + q0 * q3_dot - q3 * q0_dot)
    
    x_dict = dict(
        x=splines['x_meas'](t_fine),
        y=splines['y_meas'](t_fine),
        z=splines['z_meas'](t_fine),
        vx=[splines['x_meas'].derivatives(t)[1] for t in t_fine],
        vy=[splines['y_meas'].derivatives(t)[1] for t in t_fine],
        vz=[splines['z_meas'].derivatives(t)[1] for t in t_fine],
        q0=q0, q1=q1, q2=q2, q3=q3, wx=wx, wy=wy, wz=wz
    )
    return GeneratedModel.pack_x(t_fine.shape, **x_dict)


def save_data(tmeas, y_dict, t_fine, xopt, filename):
    data = y_dict.copy()
    data.update(zip(GeneratedModel.xnames, xopt))
    data.update(t_meas=tmeas, t=t_fine)
    io.matlab.mio.savemat(filename, data)


def main():
    tmeas, y_dict = load_data('datasmooth.bin', np.s_[:])

    splines = spline_fit(tmeas, y_dict, 4)
    params = {}
    params.update(given_params())
    params.update(param_guess())
    
    G = np.zeros((GeneratedModel.nx, 6))
    G[-6:] = np.eye(6) * [50, 50, 50, 5, 5, 5]
    c = GeneratedModel.pack_c(**params)
    p = GeneratedModel.pack_p(**params)
    y = GeneratedModel.pack_y(tmeas.shape, **y_dict)
    model = GeneratedModel(G, c=c, p=p)
    problem, t_fine = estim_problem(tmeas, y, model, 5, 1)
    
    x_guess = pack_x_guess(splines, t_fine)
    z0 = problem.pack_decision(x_guess, None, p)
    
    p_lb = dict(x_meas_std=0, y_meas_std=0, z_meas_std=0,
                q1_meas_std=0, q2_meas_std=0, q3_meas_std=0)
    p_fix = dict(x_meas_std=0.2, y_meas_std=0.2, z_meas_std=0.2,
                 q1_meas_std=0.0005, q2_meas_std=0.0005, q3_meas_std=0.0005)
    z_bounds = problem.pack_bounds(p_lb=p_lb, p_fix=p_fix)
    z_bounds[:, 3:7] = [y_dict['q0_meas'][0], y_dict['q1_meas'][0],
                        y_dict['q2_meas'][0],y_dict['q3_meas'][0]]
    
    nlp = problem.nlp(z_bounds)
    zopt, solinfo = nlp.solve(z0)
    xopt, dopt, popt = problem.unpack_decision(zopt)
    yopt = model.meas_mean(t_fine, xopt, popt)

    save_data(tmeas, y_dict, t_fine, xopt, 'filtered')
