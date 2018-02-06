import numpy as np
import pandas as pd
from sklearn import decomposition
import multiprocessing
import tabulate
import itertools
from IPython.display import display, HTML


def color2igraphColor(color):
    """Convert a color-list to an igraph color string with format 'rgb(#, #, #)'.

    Parameters
    ----------
    color : listlike
        List with color values.

    Returns
    -------
    TYPE
        igraph compatible color string.
    """
    colorStr = 'rgb(' + ', '.join(map(str, [128, 64, 255])) + ')'
    return colorStr


def objDir2Df(obj):
    """Given an object, return a DataFrame with all non-dunder parameters along with
    theur type. If an exception are raised during determining the type, the except is
    returned in place of the type. Object

    Parameters
    ----------
    obj : Object
        Object to get parameter info off.

    Returns
    -------
    DataFrame
        DataFrame with parameters and parameter types.
    """
    lst = list()
    gen = (el for el in dir(obj) if not (el.startswith('__') and el.endswith('__')))
    for el in gen:
        try:
            tup = [el, eval('type(pp.%s])' % el)]
        except Exception as e:
            tup = [el, e]
        try:
            tup.append(eval('pp.%s.shape' % el))
        except Exception as e:
            try:
                tup.append(eval('len(pp.%s)') % el)
            except Exception as e:
                tup.append(None)
        lst.append(tup)
    df = pd.DataFrame(lst, columns=['parameter', 'partype', 'shape'])
    return df


def nanEqual(a, b):
    """Returs elementvise true for equal elements in inputs a and b, treading
       NaN's as equal.
       Inputs must both have the 'shape'-attribute and be of equal shape.

    Parameters
    ----------
    a : array
        First array
    b : array
        Second array

    Returns
    -------
    boolean array

    Raises
    ------
    ValueError
    The inputs aren't of equal shape.
    """
    if not (hasattr(a, 'shape') or hasattr(b, 'shape')):
        raise ValueError("Inputs a nad b must both have the shape-attribute (like numpy arrays)")
    if a.shape != b.shape:
        raise ValueError("Inputs shapes must be identical. a.shape = {} and b.shape is {}"
                         .format(a.shape, b.shape))
    return np.all( (a == b) | (np.isnan(a) & np.isnan(b)) )  # noqa


def timedelta2unit(timedelta, unit):
    """Convert a timedelta or iterable with Timedeltas to
       a numeric value matching a given unit.

    Parameters
    ----------
    timedelta : pd.Timedelta
        Timedelta to convert. Can be an itetable of Timdeltas.
    unit : str
        Unit to convert timedeltas to. Valid arguments are {'s', 'h', 'd', 'y'}.

    Returns
    -------
    np.double
        Numpy array of timedelta values matching given unit.

    Raises
    ------
    ValueError
    If input(s) lack(s) the 'total_seconds' method.
    """

    def _inner(time, unit):
        """Convert a pd.Timedelta ('time') to a numeric value matching 'unit'self.

        Parameters
        ----------
        time : pd.Timedelta
            Timedelta to convert.
        unit : str
            String indicating unit, same as outer function.

        Returns
        -------
        double
            time converted to a double matching given 'unit'

        Raises
        ------
        ValueError
            If input `time` is missing the attribute total_seconds.
        """
        if not hasattr(time, 'total_seconds'):
            raise ValueError(('The input (or the elements of the iterable)' +
                              'must have the attribute "total_seconds".'))
        ts = time.total_seconds()
        if unit == 's':
            return ts
        if unit == 'h':
            return ts / 3600
        if unit == 'd':
            return ts / (24 * 3600)
        if unit == 'y':
            return ts / (365 * 24 * 3600)

    if hasattr(timedelta, '__iter__'):
        return np.fromiter((_inner(td, unit.lower()) for td in timedelta), np.double)
    return _inner(timedelta, unit.lower())


def standardizeData(data, getStdMean=False):
    """
    Normalize the data by substracting the mean from each feature,
    and dividing by the standard deviation.

    Parameters
    ----------
    data : array
        Data to be standardized columnwise.
    getStdMean : bool, optional
        return the standard deviation and mean for the data before the stnadardization,
        along with the data iteself.

    Returns
    -------
    array
        The snandardized data, and depending on the flag getStdMean, also the std and
        mean of the data.

    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # set values of where there's no data to 1.0, since we're dividing with the std
    std[mean == 0] = 1.0
    normData = (data - mean)/std
    if getStdMean:
        return (normData, std, mean)
    return normData


def pcaFit(toPca, performStandardization=True, **kwargs):
    """Standardize data, create a PCA object and fit the data.

    Parameters
    ----------
    toPca : np.array
        Data to perform PCA analysis on.
    performStandardization : bool, optional
        Standardize data if True, leave as be othewise.
    **kwargs : dict, optional
        Additional keyword arguments to PCA.

    Returns
    -------
    PCA
        Fitted PCA instance.
    """
    if performStandardization:
        toPca, std, mean = standardizeData(toPca, getStdMean=True)
    else:
        mean, std = np.NaN, np.NaN
    pca = decomposition.PCA(**kwargs)
    pca.fit(toPca)
    pca.norm_std = std
    pca.norm_mean = mean
    return pca


def icaFit(toIca, performStandardization=True, **kwargs):
    """Standardize data, create a ICA object and fit the data.

    Parameters
    ----------
    toIca : np.array
        Data to perform ICA analysis on.
    performStandardization : bool, optional
        Standardize data if True, leave as be othewise.
    random_state : np.random.RandomState, optional
        Random state used for the analyses, default seed is int(time.time() * 1e6)
    **kwargs : dict, optional
        Additional keyword arguments to ICA.

    Returns
    -------
    ICA
        Fitted ICA instance.
    """
    if performStandardization:
        toIca, std, mean = standardizeData(toIca, getStdMean=True)
    else:
        mean, std = np.NaN, np.NaN
    ica = decomposition.FastICA(**kwargs)
    ica.fit(toIca)
    ica.norm_std = std
    ica.norm_mean = mean
    return ica


def mapAsync(func, funcargLst, n=None):
    """Given a function and an iterable with tuple-packed arguments, evalueate the
    the function in parallel using multiple processors.

    Parameters
    ----------
    func : function
        Function to evalueate.
    funcargLst : list
        List with tuples containing arguemnts for func.
    n : int, optional
        Number of processors to use. Default is 16 or number of processors - 1
        (if there isn't 16 processors avaiable).

    Returns
    -------
    list
        List with result.
    """
    if n is None:
        n = 16 if 16 < multiprocessing.cpu_count() else multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(n) as pool:
        call = pool.starmap_async(func, funcargLst)  # parse args correctly
        res = call.get()
        pool.close()
        pool.join()
        return res


def lstDct2dct(lst):
    """Merge a list of dicts into a single dict.
    Raises warning if there's overlapping key values.

    Parameters
    ----------
    lst : list
        List with dictionaries.

    Returns
    -------
    dict
        Merged dictionary.
    """
    retDct = dict()
    for dct in lst:
        keyIntersection = set(dct.keys()).intersection(set(retDct.keys()))
        if keyIntersection:
            Warning('Keys are being overwritten: {}'.format(keyIntersection))
        retDct.update(dct)
    return retDct


def randomSample(itr, n):
    """Draw n samples from iterable, where each element can only be drawn once.

    Parameters
    ----------
    itr : iterable
        Iterable to draw from.
    n : int
        number of elements to draw.

    Returns
    -------
    list
        n elements from itr.

    Raises
    ------
    ValueError
    If the requested number of draws isn't smaller then the length if itr.
    """
    if len(itr) <= n:
        raise ValueError("'itr' must be longer that n")
    idx = np.arange(len(itr))
    np.random.shuffle(idx)
    ret = [itr[i] for i in idx[:n]]
    return ret


def getFirstDayInTimeseries(ts):
    """Given a Pandas time series, get the Timestamp for the start of the day for the
    earliest entry in the time series.

    Parameters
    ----------
    ts : Series[Tiemstamp]
        Pandas Series with Timestamps (datetime might also work).

    Returns
    -------
    Timestamp
        Pandas Timestamp matching the data for the first entry in ts.
    """
    t0 = ts.min()
    t0d = t0.date()
    return pd.Timestamp(t0d)


def swapMatrixRows(m, r0, r1, inplace=False):
    """Given a matrix m, swap rows with indices r0 and r1.

    Parameters
    ----------
    m : np.ndarray
        2D ndarray.
    r0 : int
        First row to swap.
    r1 : int
        Second row to swap.
    inplace : bool, optional
        If True, make the swapping inplace, otherwise return a swapped copy of the matrix.

    Returns
    -------
    None or ndarray
        None if inplace is True, the swapped matrix otherwise.
    """
    if inplace:
        m[[r0, r1]] = m[[r1, r0]]
        return None
    # Don't modify matrix inplace
    mt = m.copy()
    mt[[r0, r1]] = mt[[r1, r0]]
    return mt


def swapMatrixCols(m, c0, c1, inplace=False):
    """Given a matrix m, swap columns with indices c0 and c1.

    Parameters
    ----------
    m : np.ndarray
        2D ndarray.
    c0 : int
        First column to swap.
    c1 : int
        Second column to swap.
    inplace : bool, optional
        If True, make the swapping inplace, otherwise return a swapped copy of the matrix.

    Returns
    -------
    None or ndarray
        None if inplace is True, the swapped matrix otherwise.
    """
    if inplace:
        col0 = m[:, c0].copy()
        m[:, c0] = m[:, c1]
        m[:, c1] = col0
        return None
    # Don't modify the matrix inplace
    mt = m.copy()
    mt[:, c0] = m[:, c1]  # copy from original, unmodified matrix
    mt[:, c1] = m[:, c0]  # copy from original, unmodified matrix
    return mt


def stackColumns(m):
    """Stacks culumns of a matrix.

    Parameters
    ----------
    m : np.ndarray
        2D ndarray

    Returns
    -------
    np.ndarray
    """
    assert m.ndim == 2, f"Input should be a 2D ndarray, but it have dimmension {m.ndim}."
    return m.T.flatten()


def gridsearch2dataframe(clf, score='mean_test_score'):
    """Turn a grid search result into a DataFrame with irrelevant data discarded.
    The result is readdy to be passed to the plotting.heatmapFromGridsearchDf function.

    Parameters
    ----------
    clf : sklearn.model_selection.GridSearchCV
        The result of the grid search.
    score : str, optional
        Name of the score-column to keep.

    Returns
    -------
    DataFrame
    """
    df = pd.DataFrame(clf.cv_results_)
    # Remove unnecssary columns
    keepcols = {k for k in df.columns if k.startswith('param_')}
    toDrop = set(df.columns) - keepcols - {score}
    df = df.drop(toDrop, axis=1)
    return df


def inNotebook():
    """
    Returns True if called fron a Notebook, False otherwise.
    """
    try:
        return len(get_ipython().config) > 0
    except NameError:
        return False


def questionSummary(df, qstr, samplesize=0):
    """Show a summary of a question.

    Parameters
    ----------
    df : DataFrame
        DataFrame with questions
    qstr : str
        String to be used in the df.filter(like=qstr) to filter out question columns.
    samplesize : int, optional
        Number of samples to display from df.

    Raises
    ------
    ValueError
        If qstr matches more than one question.
    """
    dfs = df.filter(regex=qstr + '__')
    basename = dfs.columns[0].split('__')[0]
    if dfs.shape[1] != 5:
        column_names = ''.join(sorted({'\n• ' + el.split('__')[0] for el in dfs.columns}))
        raise ValueError("The query string matches more than one answer, specify to match one of these:" + column_names)
    dfs_question = dfs[basename + '__question'][0]
    dfs_answer_type = dfs[basename + '__answer_type'][0]
    dfs_answer_vc = pd.DataFrame(dfs[basename + '__answer'].value_counts())
    dfs_response_vc = pd.DataFrame(dfs[basename + '__response'].value_counts())
    sort_idx = np.argsort(dfs_answer_vc.index)
    dfs_answer_vc = dfs_answer_vc.iloc[sort_idx].reset_index().rename(columns={'index': 'answer_index',
                                                                               basename + '__answer': 'count'})
    dfs_response_vc = dfs_response_vc.iloc[sort_idx].reset_index().rename(columns={'index': 'response_index'})
    dfs_response_vc = dfs_response_vc.drop(basename + '__response', axis=1)
    dfs_resp_ans = dfs_response_vc.join(dfs_answer_vc)

    if inNotebook():
        display(HTML('<h3><i>Question:</i>  ' + dfs_question + '</h3>'))
        display(HTML('<i>Question str:</i>    <tt>' + qstr + '</tt>'))
        display(HTML('<i>Answer type:</i>    <tt>' + dfs_answer_type + '</tt>'))
        display(dfs_resp_ans)
        if samplesize > 0:
            dfs_print = dfs.sample(samplesize).drop([basename + '__' + el for el in ('answer_type', 'question', 'condition')], axis=1)
            display(dfs_print)
    else:
        print('Answer Question:  ' + dfs_question)
        print('Answer type:  ' + dfs_answer_type, end='\n\n')
        print(tabulate.tabulate(dfs_resp_ans, dfs_resp_ans.columns, tablefmt='pqsl'), end='\n\n')
        print(tabulate.tabulate(dfs_print.sample(samplesize), dfs_print.columns, tablefmt='pqsl'), end='\n\n')


class QuestionCompleter():

    """Use to autocomplete questions (colums) from a DataFrame.
    It autocompletes base questions as well as individual columns.
    """

    def __init__(self, _df, which='both'):
        """Init method

        Parameters
        ----------
        _df : DataFrame
            DataFrame whose columns to complete from.
        which : str, optional
            What to complete, valid values are'both', 'questions' or 'columns'

        Raises
        ------
        ValueError
            If invalid value are given as the which-argument.
        """
        if which not in "both questions columns".split():
            raise ValueError("which-argument must be 'both', 'columns' or 'questions', but was %s" % which)
        self._names = dict()
        if (which == 'questions' or which == 'both'):
            self._names.update({el[0]: el[0] for el in _df.columns.str.split('__')})
        if (which == 'columns' or which == 'both'):
            self._names.update({el: el for el in _df.columns})
        self._keys = self._names.keys()

    def __getattr__(self, attr):
        return self._names[attr]

    def __dir__(self):
        return self._keys


class QuestionFilterer():
    """Filter out all 5 columns for given questions, with tab-completion."""

    def __init__(self, _df):
        """Init method

        Parameters
        ----------
        _df : DataFrame
            DataFrame to filter columns from.
        """
        super(QuestionFilterer, self).__init__()
        self._df = _df
        self._questions = {col.split('__')[0] for col in self._df.columns}
        self._regex_char_set = set('$^\\+-*[](),.')

    def __dir__(self):
        return self._questions

    def __getattr__(self, key):
        if '__' in key:
            return self._df.filter(regex=key + '$')
        else:
            return self._df.filter(like=key)

    def __getitem__(self, key):
        return self._df.filter(regex=key)


def questionResponse(df, qstr):
    """Get value_counts for a question response.
    It called on the __response-column, but sorted by the __answer-column.

    Parameters
    ----------
    df : DataFrame
        DataFrame with relevant columns.
    qstr : str
        Question string.

    Returns
    -------
    pd.Series
        Pandas Series with value_counts.
    """
    return df[qstr + '__response'].value_counts().iloc[np.argsort(df[qstr + '__answer'].value_counts().index)]


def sortWeekdays(itr):
    itr2 = [el.lower().strip() for el in itr]
    days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
    return sorted(itr2, key=lambda day: days[day])


def getColsRowsWithNull(df):
    null_columns = df.columns[df.isnull().any()]
    df_nullcols = df[null_columns]
    nullrows_mask = df_nullcols.isnull().any(axis=1)
    return df_nullcols[nullrows_mask]


def chunkifyNonOverlappingPairs(itr_a, itr_b, chunksize):
    """Given two iterables, compute the tuple of their inner product, where overlapping elements
    have been removed. The tuple-pairs are grouped into tuples of size chunksize.

    Parameters
    ----------
    itr_a : iterable
        Iterable to use for constructing one side of the pairs.
    itr_b : iterable
        Iterable to use for constructing the other side of the pairs.
    chunksize : int
        Size of returned chunks.

    Yields
    ------
    tuepe(tuples)
        Tuple with tuples of pairs.
    """
    cnt = 0
    lst = list()
    while True:
        for ua, ub in itertools.product(itr_a, itr_b):  # loop over inner-product pairs
            if ua != ub:  # sort out pair with identical keys
                lst.append((ua, ub))
                cnt += 1
            if cnt % chunksize == 0 and cnt > 0 and len(lst):  # yield when appropiate
                yield tuple(lst)
                lst = list()
        break
    if len(lst):
        yield tuple(lst)  # when the for-iterator is exhausted, yield the last bit (non-full lst)
