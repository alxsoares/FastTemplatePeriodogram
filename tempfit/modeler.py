from __future__ import print_function

from functools import wraps

import numpy as np

from .utils import weights, ModelFitParams
from .template import Template
from . import periodogram as pdg


# function wrappers for performing checks
def requires_templates(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        self._validate_templates()
        return func(self, *args, **kwargs)
    return wrap


def requires_template(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        self._validate_template()
        return func(self, *args, **kwargs)
    return wrap


def requires_data(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        self._validate_data()
        return func(self, *args, **kwargs)
    return wrap


def power_from_fit(t, y, dy, yfit):
    w = weights(dy)
    ybar = np.dot(w, y)

    chi2_0 = np.dot(w, (y-ybar)**2)
    chi2   = np.dot(w, (y-yfit)**2)

    return 1. - (chi2 / chi2_0)


class TemplateModel(object):
    """
    Template for models of the form :math:`a * M(t - tau) + c` for
    some template M.

    Parameters
    ----------
    template : Template
        Template for the model
    frequency : float, optional (default = 1.0)
        Frequency of the signal
    parameters : ModelFitParams
        Parameters for the model (a, b, c, sgn); must be a `ModelFitParams`
        instance

    Examples
    --------
    >>> params = ModelFitParams(a=1, b=1, c=0, sgn=1)
    >>> template = Template([ 1.0, 0.4, 0.2], [0.1, 0.9, 0.2])
    >>> model = TemplateModel(template, frequency=1.0, parameters=params)
    >>> t = np.linspace(0, 10, 100)
    >>> y_fit = model(t)
    """
    def __init__(self, template, frequency=1.0, parameters=None):
        self.template = template
        self.frequency = frequency
        self.parameters = parameters

    def _validate(self):
        if not isinstance(self.template, Template):
            raise TypeError("template must be a Template instance")
        if not isinstance(self.parameters, ModelFitParams):
            raise TypeError("parameters must be ModelFitParams instance (type = {0}".format(type(self.parameters)))

    def __call__(self, t):
        self._validate()

        wtau = np.arccos(self.parameters.b)
        if self.parameters.sgn == -1:
            wtau = 2 * np.pi - wtau

        tau = wtau / (2 * np.pi * self.frequency)
        phase = (self.frequency * (t - tau)) % 1.0

        return self.parameters.a * self.template(phase) + self.parameters.c


class FastTemplateModeler(object):
    """Base class for template modelers

    Fits a single template to the data. For
    fitting multiple templates, use the FastMultiTemplateModeler

    Parameters
    ----------
    template : Template
        Template to fit (must be Template instance)
    allow_negative_amplitudes : bool (optional, default=True)
        if False, then negative optimal template amplitudes
        will be replaced with zero-amplitude solutions. A False
        value prevents the modeler from fitting an inverted
        template to the data, but does not attempt to find the
        best positive amplitude solution, which would require
        substantially more computational resources.
    """
    def __init__(self, template=None, allow_negative_amplitudes=True):
        self.template = template
        self.allow_negative_amplitudes = allow_negative_amplitudes
        self.t, self.y, self.dy = None, None, None
        self.best_model = None

    def _validate_template(self):
        if self.template is None:
            raise ValueError("No template set.")
        if not isinstance(self.template, Template):
            raise ValueError("template is not a Template instance.")

        self.template.precompute()

    def _validate_data(self):
        if any([ X is None for X in [ self.t, self.y, self.dy ] ]):
            raise ValueError("One or more of t, y, dy is None; "
                             "fit(t, y, dy) must be called first.")

        if any([ self.t[i] > self.t[i+1] for i in range(len(self.t) - 1)]):
            raise ValueError("One or more observations are not consecutive.")

        if not (len(self.t) == len(self.y) and len(self.y) == len(self.dy)):
            raise ValueError("One or more of (t, y, dy) arrays are unequal lengths")

    def _validate_frequencies(self, frequencies):
        raise NotImplementedError()

    def fit(self, t, y, dy=None):
        """
        Parameters
        ----------
        t: array_like
            sequence of observation times
        y: array_like
            sequence of observations associated with times `t`
        dy: float or array_like (optional, default=None)
            error(s)/uncertaint(ies) associated with observed values `y`.
            If scalar, all observations are weighted equally, which is
            effectively the same as setting `dy=None`.

        Returns
        -------
        self : FastTemplateModeler
            Returns self
        """
        self.t = np.array(t)
        self.y = np.array(y)
        self.dy = np.array(dy)
        return self

    @requires_data
    @requires_template
    def fit_model(self, freq):
        """Fit a template model to data.

        y_model(t) = a * template(freq * (t - tau)) + c

        Parameters
        ----------
        freq : float
            Frequency at which to fit a template model

        Returns
        -------
        model : TemplateModel
            The best-fit model at this frequency
        """
        if not any([ isinstance(freq, type_) for type_ in [ float, np.floating ] ]) :
            raise ValueError('fit_model requires float argument')

        p, parameters = pdg.fit_template(self.t, self.y, self.dy,
                                         self.template.c_n, self.template.s_n,
                                         self.template.ptensors, freq,
                                         allow_negative_amplitudes=self.allow_negative_amplitudes)

        return TemplateModel(self.template, parameters=parameters,
                             frequency=freq)

    @requires_data
    def autofrequency(self, nyquist_factor=5, samples_per_peak=5,
                      minimum_frequency=None, maximum_frequency = None):
        """
        Determine a suitable frequency grid for data.

        Note that this assumes the peak width is driven by the observational
        baseline, which is generally a good assumption when the baseline is
        much larger than the oscillation period.
        If you are searching for periods longer than the baseline of your
        observations, this may not perform well.

        Even with a large baseline, be aware that the maximum frequency
        returned is based on the concept of "average Nyquist frequency", which
        may not be useful for irregularly-sampled data. The maximum frequency
        can be adjusted via the nyquist_factor argument, or through the
        maximum_frequency argument.

        Parameters
        ----------
        samples_per_peak : float (optional, default=5)
            The approximate number of desired samples across the typical peak
        nyquist_factor : float (optional, default=5)
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        minimum_frequency : float (optional)
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float (optional)
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.

        Returns
        -------
        frequency : ndarray or Quantity
            The heuristically-determined optimal frequency bin
        """
        baseline = self.t.max() - self.t.min()
        n_samples = self.t.size

        df = 1. / (baseline * samples_per_peak)

        if minimum_frequency is not None:
            nf0 = min([ 1, np.floor(minimum_frequency / df) ])
        else:
            nf0 = 1

        if maximum_frequency is not None:
            Nf = int(np.ceil(maximum_frequency / df - nf0))
        else:
            Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

        return df * (nf0 + np.arange(Nf))

    @requires_data
    @requires_template
    def autopower(self, save_best_model=True, fast=True, **kwargs):
        """
        Compute template periodogram at automatically-determined frequencies

        Parameters
        ----------
        save_best_model : optional, bool (default = True)
            Save a TemplateModel instance corresponding to the best-fit model found
        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------
        frequency, power : ndarray, ndarray
            The frequency and template periodogram power
        """
        frequency = self.autofrequency(**kwargs)
        p, bfpars = pdg.template_periodogram(self.t, self.y, self.dy, self.template.c_n,
                            self.template.s_n, frequency,
                            ptensors=self.template.ptensors, fast=fast,
                            allow_negative_amplitudes=self.allow_negative_amplitudes)

        if save_best_model:
            i = np.argmax(p)
            self._save_best_model(TemplateModel(self.template, frequency = frequency[i],
                                                       parameters = bfpars[i]))
        return frequency, p

    @requires_data
    @requires_template
    def power(self, frequency, save_best_model=True):
        """
        Compute template periodogram at a given set of frequencies; slower than
        `autopower`, but frequencies are not restricted to being evenly spaced

        Parameters
        ----------
        frequency : float or array_like
            Frequenc(ies) at which to determine template periodogram power
        save_best_model : optional, bool (default=True)
            Save best model fit, accessible via the `best_model` attribute
        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------
        power : float or ndarray
            The frequency and template periodogram power, a
        """
        # Allow inputs of any shape; we'll reshape output to match
        frequency = np.asarray(frequency)
        shape = frequency.shape
        frequency = frequency.ravel()

        def fitter(freq):
            return pdg.fit_template(self.t, self.y, self.dy,
                                    self.template.c_n, self.template.s_n,
                                    self.template.ptensors, freq,
                                    allow_negative_amplitudes=self.allow_negative_amplitudes)

        p, bfpars = zip(*map(fitter, frequency))
        p = np.array(p)

        if save_best_model:
            i = np.argmax(p)
            best_model = TemplateModel(self.template, frequency = frequency[i],
                                       parameters = bfpars[i])
            self._save_best_model(best_model)

        return p.reshape(shape)

    def _save_best_model(self, model, overwrite=False):
        if overwrite or self.best_model is None:
            self.best_model = model
        else:
            # if there is an existing best model, replace
            # with new best model only if new model improves fit
            y_fit = model(self.t)
            y_best = self.best_model(self.t)

            p_fit = power_from_fit(self.t, self.y, self.dy, y_fit)
            p_best = power_from_fit(self.t, self.y, self.dy, y_best)

            if p_fit > p_best:
                self.best_model = model


class FastMultiTemplateModeler(FastTemplateModeler):
    """
    Template modeler that fits multiple templates

    Parameters
    ----------
    templates : list of Template
        Templates to fit (must be list of Template instances)
    allow_negative_amplitudes : bool (optional, default=True)
        if False, then negative optimal template amplitudes
        will be replaced with zero-amplitude solutions. A False
        value prevents the modeler from fitting an inverted
        template to the data, but does not attempt to find the
        best positive amplitude solution, which would require
        substantially more computational resources.
    """
    def __init__(self, templates=None, allow_negative_amplitudes=True):
        self.templates = templates
        self.allow_negative_amplitudes = allow_negative_amplitudes
        self.t, self.y, self.dy = None, None, None
        self.best_model = None

    def _validate_templates(self):
        if self.templates is None:
            raise ValueError("No templates set.")
        if not hasattr(self.templates, '__iter__'):
            raise ValueError("<object>.templates must be iterable.")
        if len(self.templates) == 0:
            raise ValueError("No templates.")
        for template in self.templates:
            if not isinstance(template, Template):
                raise ValueError("One or more templates are not Template instances.")
            template.precompute()

    @requires_data
    @requires_templates
    def fit_model(self, freq):
        """Fit a template model to data.

        y_model(t) = a * template(freq * (t - tau)) + c

        Parameters
        ----------
        freq : float
            Frequency at which to fit a template model

        Returns
        -------
        model : TemplateModel
            The best-fit model at this frequency
        """
        if not any([ isinstance(freq, type_) for type_ in [ float, np.floating ] ]) :
            raise ValueError('fit_model requires float argument')

        p, parameters = zip(*[pdg.fit_template(self.t, self.y, self.dy,
                                               template.c_n, template.s_n,
                                               template.ptensors, freq,
                                               allow_negative_amplitudes=self.allow_negative_amplitudes)
                              for template in self.templates ])

        i = np.argmax(p)
        params = parameters[i]
        template = self.templates[i]

        return TemplateModel(template, parameters=params, frequency=freq)

    @requires_data
    @requires_templates
    def autopower(self, save_best_model=True, fast=True, **kwargs):
        """
        Compute template periodogram at automatically-determined frequencies

        Parameters
        ----------
        save_best_model : optional, bool (default = True)
            Save a TemplateModel instance corresponding to the best-fit model found
        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------
        frequency, power : ndarray, ndarray
            The frequency and template periodogram power
        """
        frequency = self.autofrequency(**kwargs)

        results = [pdg.template_periodogram(self.t, self.y, self.dy, template.c_n,
                                            template.s_n, frequency,
                                            ptensors=template.ptensors, fast=fast,
                                            allow_negative_amplitudes=self.allow_negative_amplitudes)
                   for template in self.templates]

        p, bfpars = zip(*results)
        if save_best_model:
            maxes = [ max(P) for P in p ]

            i = np.argmax(maxes)
            j = np.argmax(p[i])

            self._save_best_model(TemplateModel(self.templates[i],
                                                frequency=frequency[j],
                                                parameters=bfpars[i][j]))

        return frequency, np.max(p, axis=0)

    @requires_data
    def power_from_single_template(self, frequency, template, fast=False, save_best_model=True):
        """
        Compute template periodogram at a given set of frequencies; slower than
        `autopower`, but frequencies are not restricted to being evenly spaced

        Parameters
        ----------
        frequency : float or array_like
            Frequenc(ies) at which to determine template periodogram power
        save_best_model : optional, bool (default=True)
            Save best model fit, accessible via the `best_model` attribute
        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------
        power : float or ndarray
            The frequency and template periodogram power, a
        """
        # Allow inputs of any shape; we'll reshape output to match
        frequency = np.asarray(frequency)
        shape = frequency.shape
        frequency = frequency.ravel()

        p, bfpars = pdg.template_periodogram(self.t, self.y, self.dy,
                                             template.c_n, template.s_n,
                                             frequency, ptensors=template.ptensors,
                                             fast=fast, allow_negative_amplitudes=self.allow_negative_amplitudes)
        p = np.asarray(p)
        if save_best_model:
            i = np.argmax(p)
            best_model = TemplateModel(template, frequency = frequency[i],
                                       parameters = bfpars[i])
            self._save_best_model(best_model)

        return p.reshape(shape)

    @requires_data
    @requires_templates
    def power(self, frequency, save_best_model=True, fast=False):
        """
        Compute template periodogram at a given set of frequencies

        Parameters
        ----------
        frequency : float or array_like
            Frequenc(ies) at which to determine template periodogram power
        save_best_model : optional, bool (default=True)
            Save best model fit, accessible via the `best_model` attribute
        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------
        power : float or ndarray
            The frequency and template periodogram power, a
        """
        all_power = [self.power_from_single_template(frequency, template,
                                                     fast=fast,
                                                     save_best_model=save_best_model)\
                     for template in self.templates ]

        return np.max(all_power, axis=0)
