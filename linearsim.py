import numpy as np
import scipy.stats as ss

def hs(f,Sf):
    return 4.*np.sqrt(np.sum(Sf*np.gradient(f)))

def pierson_moskowitz(f,fp,Hs):
    """
    Generate a Pierson-Moskowitz frequency dependant variance density spectrum 
    with specified parameters

    Parameters
    ----------
    f : 1-d numpy.array
        Frequencies at which to sample to 
    fp : float
        Peak frequency [1/s]
    Hs : float
        Significant wave height [m]
    gamma : float
        Peak enhancement factor in the range [1 .. 7]

    Returns
    -------
    1-d numpy.array
        Magnitude of the frequency spectra with units [m^2 s^-1]
    """
    fn = f/fp
    pm = fn**-5.*np.exp(-1.25*fn**-4.)
    scale = Hs**2/(3.2*fp)
    pm = pm*scale
    return pm

def jonswap(f,fp,Hs,gamma):
    """
    Generate a JONSWAP frequency dependant variance density spectrum 
    with specified parameters

    Parameters
    ----------
    f : 1-d numpy.array
        Frequencies at which to sample to 
    fp : float
        Peak frequency [1/s]
    Hs : float
        Significant wave height [m]
    gamma : float
        Peak enhancement factor in the range [1 .. 7]

    Returns
    -------
    1-d numpy.array
        Magnitude of the frequency spectra with units [m^2 s^-1]
    """

    fn=f/fp
    sigma = 0.07*np.ones_like(f)
    sigma[fn>=1] = 0.09
    PM = pierson_moskowitz(f,fp,Hs)
    Gf = gamma**np.exp(-0.5*((fn-1)/sigma)**2)
    C = 1 - np.exp(-5/4)*np.log(gamma) # Approximate scaling - error for increasing gamma
    Sf = C*Gf*PM
    C2 = (Hs/hs(f,Sf))**2 # Numerical adjustment based on integral
    Sf = C2*Sf
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

def spectra_stats(f,Sf,df=None):
    
    if df is None:
        df = np.gradient(f)

    m0 = np.sum(df*Sf)
    m1 = np.sum(df*f*Sf)
    m2 = np.sum(df*f**2*Sf)
    Tm01 = m0/m1
    tau = Tm01/2
    c = 1/m0*np.sum(Sf*np.cos(2.*np.pi*f*tau)*df)
    s = 1/m0*np.sum(Sf*np.sin(2.*np.pi*f*tau)*df)
    p = -1/m2*np.sum(f**2.*Sf*np.cos(2*np.pi*f*tau)*df)
    r = (c**2+s**2)**0.5
    Hm0 = 4*np.sqrt(m0)
    return Hm0, Tm01, c, s, p, r


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

    # Elevation statistics
    Hs = 4.*np.std(timeseries)
    k3 = ss.skew(timeseries)
    k4 = ss.kurtosis(timeseries)

    # Indentify waves by zero down crossings
    dcs = argcrossdown(timeseries)
    d_eta = timeseries[dcs]-timeseries[dcs+1]
    zc = dcs+timeseries[dcs]/d_eta
    Tw = np.diff(zc)/fs
    Tz = np.mean(Tw)

    # Identify individual waves
    crests = []
    troughs = []
    for i in range(len(dcs)-1):
        crest, trough = wave_height(timeseries[dcs[i]:dcs[i+1]])
        crests.append(crest)
        troughs.append(trough)
    crests = np.array(crests)
    troughs = np.array(troughs)
    r = np.corrcoef(crests**2,troughs**2)[0,1]

    # Wave heights
    wave_heights = crests - troughs
    height_inds = np.argsort(wave_heights)
    r_unbiased = np.corrcoef(crests[height_inds[:-1]]**2,troughs[height_inds[:-1]]**2)[0,1]
    sorted_heights = wave_heights[height_inds]
    Hmax = sorted_heights[-1]
    HmaxT = Tw[height_inds[-1]]
    H13 = np.mean(sorted_heights[-int(len(sorted_heights)/3):])
    H13_unbiased = np.mean(sorted_heights[-int(len(sorted_heights)/3)-1:-1])
    
    # Crest levels
    crest_inds = np.argsort(crests)
    Cmax = crests[crest_inds[-1]]
    CmaxT2 = Tw[crest_inds[-1]]/2

    return Tz, Hs, H13, H13_unbiased, Hmax, HmaxT, Cmax, CmaxT2, r, r_unbiased, k3, k4

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
    N = int(T / dt / 2)
    df = 1/(N*2*dt)
    f = np.arange(1,N+1)*df
    return f, df



def time_domain_ras(tp,hs,gamma,duration=512,dt=1/8,seed=None,fft_equiv_duration=None,return_ts=False,with_fft=True):
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
    fft_equiv_duration: float
        Use an fft resolution equivalent to fft_equiv_duration and crop result to duration.

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
    if fft_equiv_duration is None:
        f, df = freqs(duration,dt)
    else:
        f, df = freqs(fft_equiv_duration,dt)
    omega = 2*np.pi*f

    # Generate spectrum
    S = jonswap(f,1/tp,hs,gamma)

    # Random amplitude
    if seed is None:
        seed = np.random.randint(1E9)
        rs = np.random.seed(seed)
    else:
        rs = np.random.seed(seed)

    a = np.sqrt(S*df)/2*np.random.randn(*omega.shape)
    b = np.sqrt(S*df)/2*np.random.randn(*omega.shape)

    # Calculate spectral parameters of realisation
    n = len(a)
    A = np.zeros(n+1).astype('complex')
    A[1:] = (a + 1.0j*b)
    fS = np.zeros(n+1) # Add zero freqeuncy
    fS[1:] = f
    Sf = np.abs(A)**2./df*2
    Hm0, Tm01, c, s, p, r_spectra = spectra_stats(fS,Sf)

    if with_fft:
        # Timeseries from inverse  - more computationally efficient
        timeseries = np.fft.irfft(A,norm='forward')
    else:
        # Timeseries from sum of discrete components
        t = np.arange(0,duration,dt)
        timeseries = np.sum(a[:,None]*np.sin(omega[:,None]*t[None,:]) + b[:,None]*np.cos(omega[:,None]*t[None,:]),axis=0)
    
    if fft_equiv_duration is not None:
        timeseries = timeseries[:int(duration/dt)]

    # Time domain analysis
    if return_ts:
        return timeseries
    else:
        Tz, Hs, H13, H13_unbiased, Hmax, HmaxT, Cmax, CmaxT2, r_sample, r_unbiased, k3, k4 = wave_stats(timeseries,fs)
        return Tz, Tm01, Hm0, Hs, H13, H13_unbiased, Hmax, HmaxT, Cmax, CmaxT2, r_spectra, r_sample, r_unbiased, k3, k4, seed


def time_domain_das(tp,hs,gamma,duration=512,dt=1/8,seed=None,fft_equiv_duration=None,return_ts=False,with_fft=True):
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
    fft_equiv_duration: float
        Use an fft resolution equivalent to fft_equiv_duration and crop result to duration.

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
    if fft_equiv_duration is None:
        f, df = freqs(duration,dt)
    else:
        f, df = freqs(fft_equiv_duration,dt)
    omega = 2*np.pi*f

    # Generate spectrum
    S = jonswap(f,1/tp,hs,gamma)
    Hm0, Tm01, c, s, p, r_spectra = spectra_stats(f,S)
    
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

    if fft_equiv_duration is not None:
        timeseries = timeseries[:int(duration//dt)]

    # Time domain analysis
    if return_ts:
        return timeseries
    else:
        Tz, Hs, H13, H13_unbiased, Hmax, HmaxT, Cmax, CmaxT2, r_sample, r_unbiased, k3, k4 = wave_stats(timeseries,fs)
        return Tz, Tm01, Hm0, Hs, H13, H13_unbiased, Hmax, HmaxT, Cmax, CmaxT2, r_spectra, r_sample, r_unbiased, k3, k4, seed
