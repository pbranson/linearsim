import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import statsmodels.api as sm


def fit_Hs(sample,N,return_fr=False):
    bounds = [(N,N),(1E-5,1),(1E-5,1)]
    fr = ss.fit(ss.gamma,sample,bounds)
    if fr.success:
        coeffs = [p for p in fr.params]
    else:
        coeffs = [np.nan, np.nan, np.nan]
    coeffs.append(fr.nllf())
    if return_fr:
        return np.array(coeffs), fr
    else:
        return np.array(coeffs)


def fit_Tz(sample,N,plot=False,return_fr=False):
    bounds = [(0.5,1.0),(0.01,3)]
    fr = ss.fit(ss.norm,sample,bounds)
    if fr.success:
        coeffs = [p for p in fr.params]
    else:
        coeffs = [np.nan, np.nan]
    coeffs = [np.nan,] + coeffs # No shape for normal distribution
    coeffs.append(fr.nllf())
    if return_fr:
        return np.array(coeffs), fr
    else:
        return np.array(coeffs)


def fit_Hmax(sample,N,plot=False,return_fr=False):
    bounds = [(np.exp(1-np.exp(1))*N**(-1/8),np.exp(1-np.exp(1))*N**(-1/8)),(0.8,2),(0.05,0.5)] #
    fr = ss.fit(ss.genextreme,sample,bounds)
    if fr.success:
        coeffs = [p for p in fr.params]
    else:
        coeffs = [np.nan, np.nan, np.nan]
    coeffs.append(fr.nllf())
    if return_fr:
        return np.array(coeffs), fr
    else:
        return np.array(coeffs)


def fit_HmHs(sample,N,return_fr=False):
    bounds = [(np.exp(-(np.exp(1)*np.log(N))**0.5+0.5),np.exp(-(np.exp(1)*np.log(N))**0.5+0.5)),(0.2,2),(1/8,1/8)]
    fr = ss.fit(ss.genextreme,sample,bounds)
    if fr.success:
        coeffs = [p for p in fr.params]
    else:
        coeffs = [np.nan, np.nan, np.nan]
    coeffs.append(fr.nllf())
    if return_fr:
        return np.array(coeffs), fr
    else:
        return np.array(coeffs)


def fit_r(sample,N,return_fr=False):
    bounds = [(0.1,1),(0.2,0.9),(1/np.sqrt(2*N),1/np.sqrt(2*N))]
    # bounds = [(-0.5,-0.5),(-4,4),(0.0001,10)]
    fr = ss.fit(ss.genextreme,sample,bounds)
    if fr.success:
        coeffs = [p for p in fr.params]
    else:
        coeffs = [np.nan, np.nan, np.nan]
    coeffs.append(fr.nllf())
    if return_fr:
        return np.array(coeffs), fr
    else:
        return np.array(coeffs)


def summary2xarray(summary2, dim, coord):
    #Convert summary statistics tables output from statsmodel library into xarray to allow saving
    vv1=summary2.tables[0].iloc[:,[0,1]].set_index(0).T.rename(index={1:coord})
    vv2=summary2.tables[0].iloc[:,[2,3]].set_index(2).T.rename(index={3:coord})
    df = pd.concat([vv1,vv2],axis=1)
    for c in df.columns:
        # print(c)
        if c == "Date:":
            df[c] = pd.to_datetime(df[c].astype(str))
        elif c in ['Model:','Dependent Variable:']:
            df[c] = df[c].astype(str).convert_dtypes()
        else:
            df[c] = pd.to_numeric(df[c].astype(str).str.strip())

    ds_stats = xr.Dataset(df).rename({'dim_0':dim})

    # Convert the coefficients table
    ds_fits = xr.Dataset(summary2.tables[1]).rename({'dim_0':'coeff'}).expand_dims({dim:[coord,]})
    
    ds = xr.merge([ds_stats,ds_fits])
    
    # Cleanup variable names not accepted by netcdf
    for v in ds.data_vars:
        ds = ds.rename({v:"".join(i for i in v if i not in r'\/:*?"<>|[]')})

    return ds
    

def get_transform(t):
    if t == 'noop':
        return lambda x:x, ''
    elif t == 'log':
        return np.log, '\log'


def get_invtransform(t):
    if t == 'noop':
        return lambda x:x, ''
    elif t == 'log':
        return np.exp, '\exp'


def parameter_model(data,ytran='noop',xtran='log',xpow=1.,v='',p='',plot=False,labels=None):
    """
    Fit a model for the probability distribution parameters as a function of JONSWAP gamma and number of waves (N).
    
    For further details refer to publication (TBC). 

    Parameters
    ----------
    data : xarray Dataset 
        Dataset of distribution fit coefficients as a function of gamma and N.
    ytran : string either ['noop', 'log']
        The transformation to apply to the distribution coefficient
    xtran : string either ['noop', 'log']
        The transformation to apply to N
    xpow : float
        Power to raise transformed x variate, i.e. xtran(N)**xpow
    v : string
        Time-domain variable being evaluated (i.e. Hs etc)
    p : string
        Distribution parameter being modelled (i.e. loc [location], shape or scale)  
    plot : boolean
        Generate plots of model fits
    labels : Dictionary 
        Plot labels for each variable v

    Returns
    -------
    ds_regression : xarray Dataset
        A dataset with regression results returned from statsmodels.OLS
    ds_param_model : xarray Dataset 
        Dataset containing fitted coefficients for the given variable (v) and
        parameter (p). Allows estimation of p given an input gamma and N.
    """

    if plot:
        fig,axs = plt.subplot_mosaic('AA\nAA\n01',figsize=(7,8))
    else:
        fig = None

    results = []
    for gamma in data.gamma:
        this_data = data.sel(gamma=gamma)

        xt, xl = get_transform(xtran)
        yt, yl = get_transform(ytran)

        y=yt(this_data.values)
        x=xt(this_data.n.values)**xpow

        if xpow != 1.: 
            xlabel = f'$({xl}N)^{{{xpow}}}$'
        else: 
            xlabel = f'${xl}N$'
        ylabel = f'${yl}{p}$'

        X = sm.add_constant(x)
        ols = sm.OLS(y,X)
        fr = ols.fit()
        results.append(summary2xarray(fr.summary2(),dim='gamma',coord=float(gamma)))
        
        if plot:
            ax = axs['A']
            ax.scatter(x,y,15,alpha=0.5)
            y_hat=ols.predict(fr.params,X)
            ax.plot(x,y_hat,label=float(gamma),alpha=0.75)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    ds_regression = xr.merge(results).rename({'coeff':'term'})

    # Finally establish parameter dependance on gamma
    param_results =[] 
    eqn=[]
    for a,t in enumerate(ds_regression.term):
        y = ds_regression['Coef.'].sel(term=t).values
        x = ds_regression['gamma']
        X = x.to_dataframe()
        X['gamma^2'] = X['gamma']**2
        X['Intercept'] = 1
        X = X[['gamma^2','gamma','Intercept']]
        ols = sm.OLS(y,X)
        fr = ols.fit()

        param_results.append(summary2xarray(fr.summary2(),dim='term',coord=str(t.values)))

        fp = fr.params
        e = f'${fp["gamma^2"]:.1E}\gamma^2 + {fp.gamma:.1E}\gamma + {fp.Intercept:.1E}$'
        eqn.append(e)

        if plot:
            ax=axs[str(a)]
            ax.scatter(x,y,alpha=0.5)
            y_hat=ols.predict(fp,X)
            ax.plot(x,y_hat)
            ax.set_xlabel('$\gamma$')
            ax.set_ylabel(t.values)
            ax.set_title(e,fontsize=10)

    eqn = f'{ylabel} = x1.{xlabel} + const'
    
    if plot:
        lax=axs['A'].legend(loc='upper left', bbox_to_anchor=(1.0, 0.95))
        lax.set_title('$\gamma$')
        fig.axes[0].set_title(f'Variable : {labels[v]}, Parameter: ${p}$\n{eqn}')
        fig.tight_layout()
        
    ds_param_model = xr.merge(param_results)

    # Append input transformations to allow recreation    
    for input,value in dict(ytran=ytran,xtran=xtran,xpow=xpow,equation=eqn).items():
        ds_param_model[input]=value
        ds_regression[input]=value

    ds_regression = ds_regression.expand_dims({'variable':[v],'parameter':[p]})
    ds_param_model = ds_param_model.expand_dims({'variable':[v],'parameter':[p]})

    return ds_regression, ds_param_model, fig

def get_distribution(ds_model,v='Hs',gamma=1.0,N=50):
    """
    Get a parametric probability distribution for a given variable, gamma and number of waves 

    Parameters
    ----------
    ds_model : xarray Dataset 
        Dataset of univariate parametric model coefficients.
    v : string
        Time-domain variable, one of ['Hs','Tz','Hmax','HmHs','r_sample']
    gamma : float in range [1..8]
        JONSWAP peak enhancement factor
    N : float in range [10..500]
        The expected number of waves in the sample.

    Returns
    -------
    dist_inst : scipy.stats.rv_continuous 
        A frozen scipy.stats.rv_continuous with distribution parameters 
        predicted by the empirical model coefficients of ds_model.
    """

    ds_var = ds_model.sel(variable=v)
    dist_name = str(ds_var.distribution.values)
    dist = getattr(ss, dist_name)

    params = {}
    for p in ds_var.parameter:
        ds_param = ds_var.sel(parameter=p)

        # Fixed parameter definition - no fit
        cs = ds_param['Coef.'].sel(term=['x1','const'],coeff=['gamma^2','gamma','Intercept']).values
        if np.isnan(cs[0,0]):
            slope=1.
            intercept=0.
        else: # Parameter determined by fit
            slope = float(cs[0,0]*gamma**2 + cs[0,1]*gamma + cs[0,2])
            intercept = float(cs[1,0]*gamma**2 + cs[1,1]*gamma + cs[1,2])
        
        ytr, _ = get_invtransform(str(ds_param.ytran.values))
        xtr, _ = get_transform(str(ds_param.xtran.values))
        xpow = float(ds_param.xpow.values)

        y = slope*xtr(N)**xpow+intercept
        param_value = ytr(y)
        params[str(p.values)] = param_value

    #Instantiate the distribution with model parameters
    if dist_name == "norm":
        dist_inst = dist(*[params['loc'],params['scale']])
    else:
        dist_inst = dist(*[params['shape'],params['loc'],params['scale']])
    return dist_inst