"""Tests of high-level methods"""
import numpy as np
from .. import FastTemplateModeler, Template

import pytest


def template_function(phase,
                      c_n=[-0.181, -0.075, -0.020],
                      s_n=[-0.110, 0.000,  0.030]):
    n = 1 + np.arange(len(c_n))[:, np.newaxis]
    return (np.dot(c_n, np.cos(2 * np.pi * n * phase)) +
            np.dot(s_n, np.sin(2 * np.pi * n * phase)))

@pytest.fixture
def template():
    phase = np.linspace(0, 1, 100, endpoint=False)
    return phase, template_function(phase)


@pytest.fixture
def data(N=30, T=5, period=0.9, coeffs=(5, 10),
         yerr=0.1, rseed=42):
    rand = np.random.RandomState(rseed)
    t = np.sort(T * rand.rand(N))
    y = coeffs[0] + coeffs[1] * template_function(t / period)
    y += yerr * rand.randn(N)
    return t, y, yerr * np.ones_like(y)


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
def test_fast_templ_method_runs(nharmonics, template, data):
    t, y, yerr = data
    phase, y_phase = template

    template = Template.from_sampled(y_phase, nharmonics=nharmonics)
    template.precompute()

    model = FastTemplateModeler(template=template)
    model.fit(t, y, yerr)
    freq_template, power_template = model.autopower(samples_per_peak=1,
                                                    nyquist_factor=1)
