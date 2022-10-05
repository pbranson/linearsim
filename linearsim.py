import numpy as np


def jonswap(f,Tp,Hs,gamma):
    """
    Generate a JONSWAP frequency dependant variance density spectrum 
    with specified parameters

    Parameters
    ----------
    f : 1-d numpy.array
        Frequencies at which to sample to 
    Tp : float
        Peak period [s]
    Hs : float
        Significant wave height [m]
    gamma : float
        Peak enhancement factor in the range [1 .. 7]

    Returns
    -------
    1-d numpy.array
        Magnitude of the frequency spectra with units [m^2 s^-1]
    """
    B_PM = (5/4)*(1/Tp)**4
    A_PM = B_PM*(Hs/2)**2
    S_f  = A_PM*f**(-5)*np.exp(-B_PM*f**(-4))

    siga = 0.07
    sigb = 0.09

    fp = 1/Tp # peak frequency
    lind = np.where(f<=fp)
    hind = np.where(f>fp)
    Gf = np.zeros(f.shape)
    Gf[lind] = gamma**np.exp(-(f[lind]-fp)**2/(2*siga**2*fp**2))
    Gf[hind] = gamma**np.exp(-(f[hind]-fp)**2/(2*sigb**2*fp**2))
    C = 1 - 0.287*np.log(gamma)
    Sf = C*S_f*Gf

    return Sf


def argcrossdown(data):
    """
    Identify the array indexes of zero down crossings in data

    Parameters
    ----------
    data : 1-d numpy.array
        A signal, typically wave elevation timeseries in which 
        to identify zero down crossings

    Returns
    -------
    1-d numpy.array
        Magnitude of the frequency spectra with units [m^2 s^-1]
    """
    datalen = data.shape[0]
    locs = np.arange(0, datalen)
    arr = data.take(locs[:-1], axis=0, mode='clip')
    cond = np.greater_equal(arr, 0.)
    data.take(locs[1:], axis=0, mode='clip', out=arr)
    cond &= np.less(arr, 0.)
    return np.nonzero(np.concatenate([cond, np.array([False])]))[0]


def wave_height(x):
    """
    Return the crest and trough elevation of an array 
    segment containing a wave

    Parameters
    ----------
    x : 1-d numpy.array
        A signal, typically of a single wave between two 
        zero down crossings

    Returns
    -------
    crest : float 
        The maximum value in the segment
    trough : float
        The minimum value in the segment
    """
    crest = x[0]
    trough = x[0]
    for i in x[1:]:
        if i > crest:
            crest = i
        elif i < trough:
            trough = i
    return crest, trough


def wave_stats(timeseries,fs):
    """
    Calculate several time domain statistics for a given timeseries

    Parameters
    ----------
    timeseries : 1-d numpy.array
        A signal, typically of a wave elevation timeseries
    fs : float
        The sample rate of the timeseries

    Returns
    -------
    Tz : float 
        The mean zero-crossing period of the waves
    Hs : float
        Significant wave height
    Hmax : float
        The largest crest-trough wave elevation
    H13 : float
        Average of the largest 1/3 of the wave heights
    r : float
        Crest to trough correlation coefficient in range [0 .. 1].
        Larger values indicate greater groupiness.
    """

    dcs = argcrossdown(timeseries)
    Tz = np.mean(np.diff(dcs)/fs)
    Hs = 4.*np.std(timeseries)

    crests = []
    troughs = []
    for i in range(len(dcs)-1):
        crest, trough = wave_height(timeseries[dcs[i]:dcs[i+1]])
        crests.append(crest)
        troughs.append(trough)
    crests = np.array(crests)
    troughs = np.array(troughs)
    r = np.corrcoef(crests**2,troughs**2)[0,1]

    waves_heights = np.sort(crests - troughs)
    Hmax = waves_heights[-1]
    H13 = np.mean(waves_heights[-int(len(waves_heights)/3):])
    
    return Tz, Hs, Hmax, H13, r


def freqs(T,dt):
    """
    Return uniform frequency distribution for a record of 
    a given length and sample interval. 

    Parameters
    ----------
    T : float
        Total duration [s]
    dt : float
        Required sampling interval [s]

    Returns
    -------
    f : 1-d numpy.array 
        Frequencies to sample [s^-1]
    df : float
        Frequency bin width [s^-1]
    """
    N = T // dt // 2
    df = 1/(N*2*dt)
    f = np.arange(1,N+1)*df
    return f, df



def time_domain_ras(tp,hs,gamma,duration=40*60,dt=0.5,seed=None,return_ts=False,with_fft=True):
    """
    Generate a time domain realisation of a JONSWAP spectrum using the 
    Random Amplitude Scheme.

    See:
    Tucker, M. J., Challenor, P. G., & Carter, D. J. T. (1984). Numerical 
    simulation of a random sea: a common error and its effect upon wave 
    group statistics. Applied Ocean Research, 6(2), 118–122. 
    https://doi.org/10.1016/0141-1187(84)90050-6

    Merigaud, A., & Ringwood, J. V. (2018). Free-Surface Time-Series 
    Generation for Wave Energy Applications. IEEE Journal of Oceanic 
    Engineering, 43(1), 19–35. 
    https://doi.org/10.1109/JOE.2017.2691199

    Parameters
    ---------- 
    tp : float
        Peak period [s]
    hs : float
        Significant wave height [m]
    gamma : float
        Peak enhancement factor in the range [1 .. 7]
    duration: float
        Duration of record to generate with no repeated signal
    dt: float
        Sampling interval for generated timeseries
    seed: int, default = None
        Random seed, if None randonly selects a seed
    return_ts: boolean, default = False
        Return the complete timeseries (True) or the wave statistics (False)
    with_fft: boolean, default = True
        True - Use the computationally efficient inverse real Fast-Fourier 
        Transform to generate the timeseries. 
        False - Use a summation of sine and cosine components with random amplitudes
        
        Note: Both are numerically equivalent to within a reasonable precision. Option 
        provided for educational purposes.

    Returns
    -------
    If return_ts == True
    timeseries : 1-d numpy.array 
        The generated timeseries
    OR
    If return_ts == False
    Tz : float 
        The mean zero-crossing period of the waves
    Hs : float
        Significant wave height
    Hmax : float
        The largest crest-trough wave elevation
    H13 : float
        Average of the largest 1/3 of the wave heights
    r : float
        Crest to trough correlation coefficient in range [0 .. 1].
        Larger values indicate greater groupiness.
    """

    fs = 1/dt
    f, df = freqs(duration,dt)
    omega = 2*np.pi*f

    # Generate spectrum
    S = jonswap(f,tp,hs,gamma)
    # S2 = jonswap(f,5,hs,1)
    S = S

    # Random amplitude
    if seed is None:
        seed = np.random.randint(1E9)
        rs = np.random.seed(seed)
    else:
        rs = np.random.seed(seed)

    a = np.sqrt(S*df)*np.random.randn(*omega.shape)
    b = np.sqrt(S*df)*np.random.randn(*omega.shape)

    if with_fft:
        # Timeseries from inverse  - more computationally efficient
        n = len(a)
        A = np.zeros(n+1).astype('complex')
        A[1:] = (a - 1.0j*b)/2
        timeseries = np.fft.irfft(A,norm='forward')
    else:
        # Timeseries from sum of discrete components
        t = np.arange(0,duration,dt)
        timeseries = np.sum(a[:,None]*np.cos(omega[:,None]*t[None,:]) + b[:,None]*np.sin(omega[:,None]*t[None,:]),axis=0)
    
    # Time domain analysis
    if return_ts:
        return timeseries
    else:
        Tz, Hs, Hmax, H13, r = wave_stats(timeseries,fs)
        return Tz, Hs, Hmax, H13, r, seed


def time_domain_das(tp,hs,gamma,duration=40*60,dt=0.5,seed=None,return_ts=False,with_fft=True):
    """
    Generate a time domain realisation of a JONSWAP spectrum using the 
    Deterministic Amplitude Scheme (Random phase only) .

    See:
    Tucker, M. J., Challenor, P. G., & Carter, D. J. T. (1984). Numerical 
    simulation of a random sea: a common error and its effect upon wave 
    group statistics. Applied Ocean Research, 6(2), 118–122. 
    https://doi.org/10.1016/0141-1187(84)90050-6

    Merigaud, A., & Ringwood, J. V. (2018). Free-Surface Time-Series 
    Generation for Wave Energy Applications. IEEE Journal of Oceanic 
    Engineering, 43(1), 19–35. 
    https://doi.org/10.1109/JOE.2017.2691199

    Parameters
    ---------- 
    tp : float
        Peak period [s]
    hs : float
        Significant wave height [m]
    gamma : float
        Peak enhancement factor in the range [1 .. 7]
    duration: float
        Duration of record to generate with no repeated signal
    dt: float
        Sampling interval for generated timeseries
    seed: int, default = None
        Random seed, if None randonly selects a seed
    return_ts: boolean, default = False
        Return the complete timeseries (True) or the wave statistics (False)
    with_fft: boolean, default = True
        True - Use the computationally efficient inverse real Fast-Fourier 
        Transform to generate the timeseries. 
        False - Use a summation of sine and cosine components with random amplitudes
        
        Note: Both are numerically equivalent to within a reasonable precision. Option 
        provided for educational purposes.

    Returns
    -------
    If return_ts == True
    timeseries : 1-d numpy.array 
        The generated timeseries
    OR
    If return_ts == False
    Tz : float 
        The mean zero-crossing period of the waves
    Hs : float
        Significant wave height
    Hmax : float
        The largest crest-trough wave elevation
    H13 : float
        Average of the largest 1/3 of the wave heights
    r : float
        Crest to trough correlation coefficient in range [0 .. 1].
        Larger values indicate greater groupiness.
    """

    fs = 1/dt
    f, df = freqs(duration,dt)
    omega = 2*np.pi*f

    # Generate spectrum
    S = jonswap(f,tp,hs,gamma)
    
    # Random amplitude
    if seed is None:
        seed = np.random.randint(1E9)
        rs = np.random.seed(seed)
    else:
        rs = np.random.seed(seed)
    phase = 2. * np.pi * np.random.rand(*omega.shape) 

    if with_fft:
        # Timeseries from inverse  - more computationally efficient
        n = len(phase)
        A = np.zeros(n+1).astype('complex')
        A[1:] = np.sqrt(1/2*S*df) * np.exp(1.0j*phase)
        timeseries = np.fft.irfft(A,norm='forward')
    else:
        # Timeseries sum of spectral components
        t = np.arange(0,duration,dt)
        timeseries = np.sum(np.sqrt(2*S[:,None]*df)*np.cos(omega[:,None]*t[None,:] + phase[:,None]),axis=0)

    # Time domain analysis
    if return_ts:
        return timeseries
    else:
        Tz, Hs, Hmax, H13, r = wave_stats(timeseries,fs)
        return Tz, Hs, Hmax, H13, r, seed