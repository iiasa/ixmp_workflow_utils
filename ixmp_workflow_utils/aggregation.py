import logging

import pandas as pd
from pyam.utils import isstr, KNOWN_FUNCS

log = logging.getLogger(__name__)


def _get_method_func(method):
    """Translate a string to a known method"""
    if not isstr(method):
        return method

    if method in KNOWN_FUNCS:
        return KNOWN_FUNCS[method]


def wavg(df):
    d = df['value']
    w = df['value_weight']
    return (d * w).sum() / w.sum()


def aggregate_model(model_df, region_aggregation, variable_config):
    log.info('# of variables(data): {}'.format(len(model_df.variables())))
    log.info('# of variables(info): {}'.format(len(variable_config.keys())))

    # Convert variable_config from yaml into a DataFrame.
    vc = pd.DataFrame.from_dict(variable_config, orient='index',
                                columns=['unit', 'required', 'method',
                                         'weight'])
    vc.reset_index(drop=False, inplace=True)
    vc.columns = ['variable', 'unit', 'required', 'method', 'weight']
    # default method
    vc.loc[vc.method.isnull(), 'method'] = 'sum'  # change all empty into sum
    vc.loc[vc.method == 'w.avg', 'method'] = 'wavg'  # match function name
    methods = vc.method.unique()
    # Placeholder to add checks that parsed methods are accepted
    # - however don't want this output for the user, but for instance manager.
    #    acc_methods = ['max','min','sum','wavg']
    #    rej_methods = [x for x in methods if x not in acc_methods]
    #    if len(rej_methods):
    #        log.warning('Only {} aggregation methods accepted.
    #  These methods were rejected: {}'.format(acc_methods, rej_methods))
    log.info('methods: {}'.format(methods))

    for region, subregions in region_aggregation.items():
        log.info('Aggregating variables for region {}, subregions {}'
                 .format(region, subregions))
        subregs_df = model_df.filter(region=subregions)
        if len(subregs_df):
            #
            # remove all variables which do have tgt region data in model_df
            tgt_region_df = model_df.filter(region=region)
            tgt_variables = tgt_region_df['variable'].unique()
            if len(tgt_variables):
                keep_vars = set(subregs_df.variables()) - set(tgt_variables)
                subregs_df = subregs_df.filter(variable=keep_vars)
        if len(subregs_df):
            for method in methods:
                # Make list of variables for method,
                # but keep only those in subregs_df
                varlist = vc.loc[vc.method == method, ['variable', 'weight']]
                varlist = varlist[varlist['variable'].isin(
                    list(model_df.variables()))]
                log.info('{} variables use {} aggregation method'.format(
                    len(varlist), method))
                dft = (subregs_df.filter(variable=varlist['variable'].tolist())
                       .as_pandas())
                if method != 'wavg':
                    _method = _get_method_func(method)
                    dfgrouped = (dft.groupby(by=['model', 'scenario',
                                                 'variable', 'year', 'unit'])
                                 .apply(_method))
                    if len(dfgrouped):
                        dfgrouped = dfgrouped['value'].reset_index()
                        dfgrouped['region'] = region
                        model_df.append(dfgrouped, inplace=True)
                else:  # weighted averages
                    weights = subregs_df.filter(
                        variable=varlist.weight.tolist()).as_pandas()
                    if len(weights):
                        weights = weights.rename({'variable': 'var_weight',
                                                  'value': 'value_weight'},
                                                 axis=1)
                        weights = pd.merge(weights, varlist,
                                           left_on='var_weight',
                                           right_on='weight')
                        weights = weights[['model', 'scenario', 'region',
                                           'unit', 'year', 'var_weight',
                                           'variable', 'value_weight', ]]
                        cols = ['model', 'scenario', 'variable', 'region',
                                'year']
                        dfweights = pd.merge(dft, weights, left_on=cols,
                                             right_on=cols,
                                             suffixes=('_var', '_weight'))

                        dfgrouped = dfweights.groupby(by=['model', 'scenario',
                                                          'variable', 'year',
                                                          'unit_var']
                                                      ).apply(eval(method)
                                                              ).reset_index()
                        dfgrouped.rename({'unit_var': 'unit', 0: 'value'},
                                         inplace=True, axis=1)
                        dfgrouped['region'] = region
                        # append - if not empty
                        if not dfgrouped.empty:
                            model_df.append(dfgrouped, inplace=True)
