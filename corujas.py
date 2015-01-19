import textwrap

import attrdict
import numpy as np
from scipy import io, interpolate
from thesis2 import codegen, kalman, mocos, symstats, vme
from thesis2.experiments import base


class SymbolicModel(base.SymbolicModel):
    '''trakSTAR path reconstruction model.'''
    
    x = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    '''State vector.'''
    
    y = ['x_meas', 'y_meas', 'z_meas']
    '''Measurement vector.'''
    
    p = []
    '''Parameter vector.'''
    
    s = []
    '''Exogenous signals vector.'''
    
    c = []
    '''Constants vector.'''

    def f(self, t, x, p, s, c):
        '''Drift function.'''
        a = attrdict.AttrDict(self.unpack_arguments(t=t, x=x, p=p, s=s, c=c))
        return [a.vx, a.vy, a.vz, 0, 0, 0]

    def meas_mean(self, t, x, p, s, c):
        '''Measurement mean.'''
        a = attrdict.AttrDict(self.unpack_arguments(t=t, x=x, p=p, s=s, c=c))
        return [a.x, a.y, a.z]
    
    def meas_cov(self, t, x, p, s, c):    
        '''Measurement covariance matrix.'''
        a = attrdict.AttrDict(self.unpack_arguments(t=t, x=x, p=p, s=s, c=c))
        return np.diag([1, 1, 1])
    
    def meas_ll(self, y, t, x, p, s, c):
        '''Measurement log-likelihood.'''
        a = attrdict.AttrDict(
            self.unpack_arguments(t=t, x=x, y=y, p=p, s=s, c=c)
        )
        return (symstats.normal_logpdf(a.x_meas, a.x, 1) +
                symstats.normal_logpdf(a.y_meas, a.y, 1) +
                symstats.normal_logpdf(a.z_meas, a.z, 1))
    
    def prior_logpdf(self, x, p, c):
        return 0


def generated_src():
    model_generator = base.ModelGenerator(SymbolicModel(), 'GeneratedModel')
    return model_generator.generate()


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


def load_data(filepath, range_):
    data = io.loadmat(filepath)
    tmeas = data['t'].flatten()[range_]
    y_dict = dict(
        x_meas=data['x'].flatten()[range_],
        y_meas=data['y'].flatten()[range_],
        z_meas=data['z'].flatten()[range_],
    )
    return tmeas, y_dict


def spline_fit(tmeas, y_dict, smoothing_factor):
    Tknot = (tmeas[1] - tmeas[0]) * smoothing_factor
    knots = np.arange(tmeas[0] + 2 * Tknot, tmeas[-1] - 2 * Tknot, Tknot)
    splines = {}
    for yname in SymbolicModel.y:
        splines[yname] = interpolate.LSQUnivariateSpline(
            tmeas, y_dict[yname], knots, k=5
        )
    return splines


def given_params():
    return {}


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
    x_dict = dict(
        x=splines['x_meas'](t_fine),
        y=splines['y_meas'](t_fine),
        z=splines['z_meas'](t_fine),
        vx=[splines['x_meas'].derivatives(t)[1] for t in t_fine],
        vy=[splines['y_meas'].derivatives(t)[1] for t in t_fine],
        vz=[splines['z_meas'].derivatives(t)[1] for t in t_fine],
    )
    return GeneratedModel.pack_x(t_fine.shape, **x_dict)


def main():
    tmeas, y_dict = load_data('20140528AC1301FREE01.bin', np.s_[158977:162202])

    splines = spline_fit(tmeas, y_dict, 4)
    params = given_params()
    
    G = np.zeros((GeneratedModel.nx, 3))
    G[-3:] = np.eye(3) * [1, 1, 1]
    c = GeneratedModel.pack_c(**params)
    p = GeneratedModel.pack_p(**params)
    y = GeneratedModel.pack_y(tmeas.shape, **y_dict)
    model = GeneratedModel(G, c=c, p=p)
    problem, t_fine = estim_problem(tmeas, y, model, 5, 1)
    
    x_guess = pack_x_guess(splines, t_fine)
    z0 = problem.pack_decision(x_guess, None, p)
    
    p_lb = dict()
    p_fix = dict()
    z_bounds = problem.pack_bounds(p_lb=p_lb, p_fix=p_fix)
    
    nlp = problem.nlp(z_bounds)
    #nlp.num_option(b'tol', 1e-6)
    zopt, solinfo = nlp.solve(z0)
    xopt, dopt, popt = problem.unpack_decision(zopt)
    yopt = model.meas_mean(t_fine, xopt, popt)

