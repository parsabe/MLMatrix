# Numpy Most important functions


<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Arguments</th>
      <th>Explanation of Important Arguments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>numpy.array</td>
      <td>
        <ul>
          <li>object</li>
          <li>dtype=None</li>
          <li>copy=True</li>
          <li>order='K'</li>
          <li>subok=False</li>
          <li>ndmin=0</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>object:</strong> Any array-like input.</li>
          <li><strong>dtype:</strong> Desired data type of the array (e.g., np.float32).</li>
          <li><strong>copy:</strong> Whether to copy the data (True) or not (False).</li>
          <li><strong>ndmin:</strong> Minimum number of dimensions required.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.zeros</td>
      <td>
        <ul>
          <li>shape</li>
          <li>dtype=float</li>
          <li>order='C'</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>shape:</strong> Shape of the new array (e.g., (3, 4)).</li>
          <li><strong>dtype:</strong> Desired data type (e.g., int).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.ones</td>
      <td>
        <ul>
          <li>shape</li>
          <li>dtype=float</li>
          <li>order='C'</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>shape:</strong> Shape of the new array (e.g., (3, 4)).</li>
          <li><strong>dtype:</strong> Desired data type (e.g., int).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.linspace</td>
      <td>
        <ul>
          <li>start</li>
          <li>stop</li>
          <li>num=50</li>
          <li>endpoint=True</li>
          <li>retstep=False</li>
          <li>dtype=None</li>
          <li>axis=0</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>start:</strong> Starting value of the sequence.</li>
          <li><strong>stop:</strong> End value of the sequence.</li>
          <li><strong>num:</strong> Number of evenly spaced samples to generate.</li>
          <li><strong>retstep:</strong> If True, return the step size.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.random.rand</td>
      <td>
        <ul>
          <li>*args</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>*args:</strong> Dimensions of the output array (e.g., (2, 3)).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.random.randn</td>
      <td>
        <ul>
          <li>*args</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>*args:</strong> Dimensions of the output array, sampled from a normal distribution.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.dot</td>
      <td>
        <ul>
          <li>a</li>
          <li>b</li>
          <li>out=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>a, b:</strong> Input arrays to compute dot product.</li>
          <li><strong>out:</strong> Output array (optional).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.mean</td>
      <td>
        <ul>
          <li>a</li>
          <li>axis=None</li>
          <li>dtype=None</li>
          <li>out=None</li>
          <li>keepdims=False</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>a:</strong> Input array.</li>
          <li><strong>axis:</strong> Axis along which the means are computed (default: entire array).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.median</td>
      <td>
        <ul>
          <li>a</li>
          <li>axis=None</li>
          <li>out=None</li>
          <li>overwrite_input=False</li>
          <li>keepdims=False</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>a:</strong> Input array.</li>
          <li><strong>axis:</strong> Axis along which the medians are computed (default: entire array).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.std</td>
      <td>
        <ul>
          <li>a</li>
          <li>axis=None</li>
          <li>dtype=None</li>
          <li>out=None</li>
          <li>ddof=0</li>
          <li>keepdims=False</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>a:</strong> Input array.</li>
          <li><strong>ddof:</strong> Delta degrees of freedom for sample standard deviation.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>numpy.cov</td>
      <td>
        <ul>
          <li>m</li>
          <li>y=None</li>
          <li>rowvar=True</li>
          <li>bias=False</li>
          <li>ddof=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>m:</strong> Input matrix.</li>
          <li><strong>rowvar:</strong> If True, rows are variables, columns are observations.</li>
          <li><strong>ddof:</strong> Degrees of freedom.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


```python
import numpy as np

# 1. Create an array
arr = np.array([1, 2, 3, 4], dtype=float)

# 2. Zeros array
zeros_arr = np.zeros((3, 3))

# 3. Ones array
ones_arr = np.ones((2, 2))

# 4. Arange example
arange_arr = np.arange(0, 10, 2)

# 5. Linspace example
linspace_arr = np.linspace(0, 1, 5)

# 6. Reshape array
reshaped_arr = arr.reshape(2, 2)

# 7. Random rand array
rand_arr = np.random.rand(2, 3)

# 8. Random randn array (normal distribution)
randn_arr = np.random.randn(2, 3)

# 9. Random integers array
randint_arr = np.random.randint(0, 10, (2, 3))

# 10. Compute mean
mean_value = np.mean(rand_arr)

# 11. Compute median
median_value = np.median(rand_arr)

# 12. Compute standard deviation
std_value = np.std(rand_arr)

# 13. Sum array
sum_value = np.sum(arr)

# 14. Dot product
dot_product = np.dot(np.array([1, 2]), np.array([3, 4]))

# 15. Matrix inversion
matrix = np.array([[1, 2], [3, 4]])
inverse_matrix = np.linalg.inv(matrix)

# 16. Matrix determinant
determinant = np.linalg.det(matrix)

# 17. Concatenate arrays
concat_arr = np.concatenate([arr, np.array([5, 6])])

# 18. Vertical stack
vstack_arr = np.vstack((arr, arr))

# 19. Horizontal stack
hstack_arr = np.hstack((arr, arr))

# 20. Covariance matrix
cov_matrix = np.cov(np.array([1, 2, 3]), np.array([4, 5, 6]))

```

# Pandas 

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Arguments</th>
      <th>Explanation of Important Arguments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>pandas.DataFrame</td>
      <td>
        <ul>
          <li>data=None</li>
          <li>index=None</li>
          <li>columns=None</li>
          <li>dtype=None</li>
          <li>copy=False</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>data:</strong> 2D array-like data structure (e.g., dict, ndarray).</li>
          <li><strong>index:</strong> Row labels.</li>
          <li><strong>columns:</strong> Column labels.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.read_csv</td>
      <td>
        <ul>
          <li>filepath_or_buffer</li>
          <li>sep=','</li>
          <li>header='infer'</li>
          <li>names=None</li>
          <li>index_col=None</li>
          <li>usecols=None</li>
          <li>dtype=None</li>
          <li>nrows=None</li>
          <li>skiprows=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>filepath_or_buffer:</strong> Path or URL to the file.</li>
          <li><strong>sep:</strong> Delimiter to use (default: ',').</li>
          <li><strong>usecols:</strong> List of columns to parse.</li>
          <li><strong>nrows:</strong> Number of rows to read (optional).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.head</td>
      <td>
        <ul>
          <li>n=5</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>n:</strong> Number of rows to return from the top (default: 5).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.describe</td>
      <td>
        <ul>
          <li>percentiles=None</li>
          <li>include=None</li>
          <li>exclude=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>percentiles:</strong> List of percentiles to include in the output (optional).</li>
          <li><strong>include/exclude:</strong> Specify data types to include/exclude in summary statistics.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.groupby</td>
      <td>
        <ul>
          <li>by</li>
          <li>axis=0</li>
          <li>level=None</li>
          <li>as_index=True</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>by:</strong> Column or list of columns to group by.</li>
          <li><strong>as_index:</strong> Whether to return the group labels as an index (default: True).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.merge</td>
      <td>
        <ul>
          <li>left</li>
          <li>right</li>
          <li>how='inner'</li>
          <li>on=None</li>
          <li>left_on=None</li>
          <li>right_on=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>left, right:</strong> DataFrames to merge.</li>
          <li><strong>how:</strong> Type of merge to be performed ('left', 'right', 'outer', 'inner').</li>
          <li><strong>on:</strong> Columns or indexes to join on.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.pivot_table</td>
      <td>
        <ul>
          <li>data</li>
          <li>values=None</li>
          <li>index=None</li>
          <li>columns=None</li>
          <li>aggfunc='mean'</li>
          <li>fill_value=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>data:</strong> DataFrame to pivot.</li>
          <li><strong>values:</strong> Column(s) to aggregate.</li>
          <li><strong>index:</strong> Rows to group by.</li>
          <li><strong>aggfunc:</strong> Aggregation function (default: 'mean').</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.concat</td>
      <td>
        <ul>
          <li>objs</li>
          <li>axis=0</li>
          <li>join='outer'</li>
          <li>ignore_index=False</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>objs:</strong> List or dict of DataFrames/Series to concatenate.</li>
          <li><strong>axis:</strong> Axis to concatenate along (default: 0).</li>
          <li><strong>join:</strong> Join method ('inner' or 'outer').</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.drop</td>
      <td>
        <ul>
          <li>labels</li>
          <li>axis=0</li>
          <li>index=None</li>
          <li>columns=None</li>
          <li>inplace=False</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>labels:</strong> Rows/columns to drop.</li>
          <li><strong>axis:</strong> Whether to drop rows (0) or columns (1).</li>
          <li><strong>inplace:</strong> If True, modify the DataFrame in place.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>pandas.apply</td>
      <td>
        <ul>
          <li>func</li>
          <li>axis=0</li>
          <li>raw=False</li>
          <li>result_type=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>func:</strong> Function to apply to DataFrame/Series.</li>
          <li><strong>axis:</strong> Apply along rows (1) or columns (0).</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

```python
import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# Read a CSV file
df_csv = pd.read_csv('data.csv')
print("CSV DataFrame:\n", df_csv)

# Display the first 5 rows
print("First 5 rows:\n", df.head())

# Get a statistical summary
print("Describe:\n", df.describe())

# Group by a column
df_grouped = df.groupby('A').sum()
print("Grouped DataFrame:\n", df_grouped)

# Merge two DataFrames
df2 = pd.DataFrame({'A': [1, 2], 'C': [7, 8]})
df_merged = pd.merge(df, df2, on='A', how='inner')
print("Merged DataFrame:\n", df_merged)

# Pivot table
df_pivot = pd.pivot_table(df, values='B', index='A', aggfunc='mean')
print("Pivot Table:\n", df_pivot)

# Concatenate DataFrames
df_concat = pd.concat([df, df2], axis=1)
print("Concatenated DataFrame:\n", df_concat)

# Drop a column
df_dropped = df.drop('B', axis=1)
print("DataFrame after dropping column:\n", df_dropped)

# Apply a function to a DataFrame
df_applied = df.apply(lambda x: x * 2)
print("Applied function:\n", df_applied)

```
# Scipy



<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Arguments</th>
      <th>Explanation of Important Arguments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>scipy.integrate.quad</td>
      <td>
        <ul>
          <li>func</li>
          <li>a</li>
          <li>b</li>
          <li>args=()</li>
          <li>epsabs=1.49e-08</li>
          <li>epsrel=1.49e-08</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>func:</strong> Function to integrate.</li>
          <li><strong>a, b:</strong> Limits of integration (start and end points).</li>
          <li><strong>args:</strong> Extra arguments to pass to the function.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.optimize.minimize</td>
      <td>
        <ul>
          <li>fun</li>
          <li>x0</li>
          <li>args=()</li>
          <li>method=None</li>
          <li>jac=None</li>
          <li>hess=None</li>
          <li>options=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>fun:</strong> The objective function to be minimized.</li>
          <li><strong>x0:</strong> Initial guess for the minimization.</li>
          <li><strong>method:</strong> Optimization method (e.g., 'BFGS', 'CG').</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.linalg.inv</td>
      <td>
        <ul>
          <li>a</li>
          <li>overwrite_a=False</li>
          <li>check_finite=True</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>a:</strong> Matrix to invert.</li>
          <li><strong>overwrite_a:</strong> If True, allows overwriting the input matrix (may improve performance).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.fft.fft</td>
      <td>
        <ul>
          <li>x</li>
          <li>n=None</li>
          <li>axis=-1</li>
          <li>norm=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x:</strong> Input array for Fourier transform.</li>
          <li><strong>n:</strong> Length of the FFT. If None, the length of the input is used.</li>
          <li><strong>axis:</strong> Axis over which to compute the FFT.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.spatial.distance.euclidean</td>
      <td>
        <ul>
          <li>u</li>
          <li>v</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>u, v:</strong> Input vectors to compute the Euclidean distance between them.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.stats.norm.pdf</td>
      <td>
        <ul>
          <li>x</li>
          <li>loc=0</li>
          <li>scale=1</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x:</strong> Points at which to evaluate the PDF.</li>
          <li><strong>loc:</strong> Mean ("center") of the distribution (default: 0).</li>
          <li><strong>scale:</strong> Standard deviation (default: 1).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.stats.ttest_ind</td>
      <td>
        <ul>
          <li>a</li>
          <li>b</li>
          <li>axis=0</li>
          <li>equal_var=True</li>
          <li>nan_policy='propagate'</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>a, b:</strong> Sample data to compare.</li>
          <li><strong>equal_var:</strong> If True, perform a standard independent 2-sample test assuming equal variance.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.cluster.hierarchy.linkage</td>
      <td>
        <ul>
          <li>y</li>
          <li>method='single'</li>
          <li>metric='euclidean'</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>y:</strong> Input data (or distance matrix).</li>
          <li><strong>method:</strong> Linkage method to compute ('single', 'complete', 'average', etc.).</li>
          <li><strong>metric:</strong> Distance metric to use.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.interpolate.interp1d</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>kind='linear'</li>
          <li>axis=-1</li>
          <li>fill_value=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Input data points for interpolation.</li>
          <li><strong>kind:</strong> Specifies the kind of interpolation ('linear', 'nearest', etc.).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>scipy.signal.find_peaks</td>
      <td>
        <ul>
          <li>x</li>
          <li>height=None</li>
          <li>threshold=None</li>
          <li>distance=None</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x:</strong> Input data in which to find peaks.</li>
          <li><strong>height:</strong> Required height of peaks (optional).</li>
          <li><strong>distance:</strong> Minimum distance between peaks (optional).</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


```python
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.linalg as linalg
import scipy.fft as fft
import scipy.spatial.distance as distance
import scipy.stats as stats
import scipy.cluster.hierarchy as hierarchy
import scipy.interpolate as interpolate
import scipy.signal as signal

# 1. Integrate a function (e.g., f(x) = x^2) over [0, 1]
result, error = integrate.quad(lambda x: x**2, 0, 1)
print("Integration result:", result)

# 2. Minimize a function (e.g., f(x) = (x-3)^2)
result = optimize.minimize(lambda x: (x - 3)**2, x0=0)
print("Minimization result:", result.x)

# 3. Inverse of a matrix
matrix = np.array([[1, 2], [3, 4]])
inverse_matrix = linalg.inv(matrix)
print("Inverse matrix:\n", inverse_matrix)

# 4. Fast Fourier Transform
x = np.array([1, 2, 1, 0, 1, 2])
fft_result = fft.fft(x)
print("FFT result:", fft_result)

# 5. Euclidean distance between two vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
euclidean_dist = distance.euclidean(u, v)
print("Euclidean distance:", euclidean_dist)

# 6. Probability Density Function of a normal distribution
x = np.linspace(-3, 3, 100)
pdf_values = stats.norm.pdf(x, loc=0, scale=1)
print("PDF values (normal distribution):", pdf_values)

# 7. T-test for two independent samples
a = [1, 2, 3, 4, 5]
b = [5, 6, 7, 8, 9]
t_stat, p_value = stats.ttest_ind(a, b)
print("T-test statistic:", t_stat, "P-value:", p_value)

# 8. Linkage matrix for hierarchical clustering
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
linkage_matrix = hierarchy.linkage(data, method='single')
print("Linkage matrix:\n", linkage_matrix)

# 9. 1D interpolation
x = np.linspace(0, 10, num=10)
y = np.sin(x)
interp_func = interpolate.interp1d(x, y, kind='linear')
y_interp = interp_func(5.5)
print("Interpolated value at 5.5:", y_interp)

# 10. Find peaks in a signal
signal_data = np.array([0, 1, 0, 2, 1, 0, 1, 3, 0])
peaks, _ = signal.find_peaks(signal_data, height=1)
print("Peaks at indices:", peaks)


```
# Matplotlib

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Arguments</th>
      <th>Explanation of Important Arguments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>matplotlib.pyplot.plot</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>color=None</li>
          <li>linestyle='-'</li>
          <li>marker=None</li>
          <li>label=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data points for the plot.</li>
          <li><strong>color:</strong> Line color.</li>
          <li><strong>linestyle:</strong> Line style (e.g., '-', '--').</li>
          <li><strong>marker:</strong> Marker type (e.g., 'o', '^').</li>
          <li><strong>label:</strong> Label for the legend.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.scatter</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>c=None</li>
          <li>marker='o'</li>
          <li>cmap=None</li>
          <li>s=20</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data points for the scatter plot.</li>
          <li><strong>c:</strong> Color of points.</li>
          <li><strong>marker:</strong> Marker style (e.g., 'o', 'x').</li>
          <li><strong>cmap:</strong> Colormap for mapping color values.</li>
          <li><strong>s:</strong> Size of the points.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.hist</td>
      <td>
        <ul>
          <li>x</li>
          <li>bins=10</li>
          <li>range=None</li>
          <li>density=False</li>
          <li>color=None</li>
          <li>edgecolor='black'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x:</strong> Data to plot.</li>
          <li><strong>bins:</strong> Number of bins or bin edges.</li>
          <li><strong>range:</strong> Lower and upper range of the bins.</li>
          <li><strong>density:</strong> If True, normalize the histogram.</li>
          <li><strong>color:</strong> Color of the bars.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.bar</td>
      <td>
        <ul>
          <li>x</li>
          <li>height</li>
          <li>width=0.8</li>
          <li>bottom=None</li>
          <li>color=None</li>
          <li>edgecolor=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x:</strong> The x coordinates of the bars.</li>
          <li><strong>height:</strong> The heights of the bars.</li>
          <li><strong>width:</strong> The width of the bars.</li>
          <li><strong>bottom:</strong> Bottom of the bars.</li>
          <li><strong>color:</strong> Color of the bars.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.boxplot</td>
      <td>
        <ul>
          <li>x</li>
          <li>notch=False</li>
          <li>vert=True</li>
          <li>patch_artist=False</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x:</strong> Data to be plotted.</li>
          <li><strong>notch:</strong> If True, create a notch for the box plot.</li>
          <li><strong>vert:</strong> If True, make vertical boxes.</li>
          <li><strong>patch_artist:</strong> If True, fill the boxes with color.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.imshow</td>
      <td>
        <ul>
          <li>arr</li>
          <li>cmap=None</li>
          <li>interpolation='nearest'</li>
          <li>aspect='equal'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>arr:</strong> Array to be displayed as an image.</li>
          <li><strong>cmap:</strong> Colormap for the image.</li>
          <li><strong>interpolation:</strong> Interpolation method (e.g., 'nearest', 'bilinear').</li>
          <li><strong>aspect:</strong> Aspect ratio of the image.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.subplot</td>
      <td>
        <ul>
          <li>nrows</li>
          <li>ncols</li>
          <li>index</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>nrows:</strong> Number of rows in the grid.</li>
          <li><strong>ncols:</strong> Number of columns in the grid.</li>
          <li><strong>index:</strong> Position of the subplot in the grid.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.title</td>
      <td>
        <ul>
          <li>label</li>
          <li>loc='center'</li>
          <li>pad=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>label:</strong> Title of the plot.</li>
          <li><strong>loc:</strong> Location of the title ('left', 'center', 'right').</li>
          <li><strong>pad:</strong> Padding between the title and plot.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.legend</td>
      <td>
        <ul>
          <li>handles=None</li>
          <li>labels=None</li>
          <li>loc='best'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>handles:</strong> List of artist handles (e.g., Line2D objects).</li>
          <li><strong>labels:</strong> Labels for the legend.</li>
          <li><strong>loc:</strong> Location of the legend (e.g., 'upper right', 'best').</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>matplotlib.pyplot.savefig</td>
      <td>
        <ul>
          <li>filename</li>
          <li>dpi=None</li>
          <li>bbox_inches='tight'</li>
          <li>format=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>filename:</strong> Filename or path to save the figure.</li>
          <li><strong>dpi:</strong> Dots per inch (resolution) of the output file.</li>
          <li><strong>bbox_inches:</strong> Bounding box in inches (e.g., 'tight' to fit the plot tightly).</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

```python

import matplotlib.pyplot as plt
import numpy as np

# 1. Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, color='blue', linestyle='--', marker='o', label='Sine wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.show()

# 2. Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, c='red', marker='x', s=100, cmap='viridis')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.colorbar(label='Color scale')
plt.show()

# 3. Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, color='green', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# 4. Bar plot
categories = ['A', 'B', 'C']
values = [3, 7, 5]
plt.bar(categories, values, color='purple', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()

# 5. Box plot
data = [np.random.normal(size=100) for _ in range(3)]
plt.boxplot(data, notch=True, vert=True, patch_artist=True)
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Box Plot')
plt.show()

# 6. Image display
matrix = np.random.rand(10, 10)
plt.imshow(matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap')
plt.show()

# 7. Subplots
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Plot 1')
axs[0, 1].scatter(x, y)
axs[0, 1].set_title('Scatter 1')
axs[1, 0].hist(data, bins=30)
axs[1, 0].set_title('Histogram 1')
axs[1, 1].bar(categories, values)
axs[1, 1].set_title('Bar Plot 1')
plt.tight_layout()
plt.show()

# 8. Title
plt.plot(x, y)
plt.title('Simple Plot', loc='left', pad=20)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# 9. Legend
plt.plot(x, y, label='Sine wave')
plt.legend(loc='upper right')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot with Legend')
plt.show()

# 10. Save figure
plt.plot(x, y)
plt.title('Saved Plot')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

```
# Seaborn

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Arguments</th>
      <th>Explanation of Important Arguments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>seaborn.scatterplot</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>hue=None</li>
          <li>style=None</li>
          <li>size=None</li>
          <li>palette=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>style:</strong> Variable for marker style.</li>
          <li><strong>size:</strong> Variable for marker size.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.lineplot</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>hue=None</li>
          <li>style=None</li>
          <li>markers=False</li>
          <li>palette=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>style:</strong> Variable for line style.</li>
          <li><strong>markers:</strong> If True, add markers to the line plot.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.histplot</td>
      <td>
        <ul>
          <li>data</li>
          <li>x=None</li>
          <li>y=None</li>
          <li>hue=None</li>
          <li>stat='count'</li>
          <li>bins=None</li>
          <li>palette=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>data:</strong> Data to plot.</li>
          <li><strong>x, y:</strong> Variables for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>stat:</strong> Statistic to compute (e.g., 'count', 'density').</li>
          <li><strong>bins:</strong> Number of bins or bin edges.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.barplot</td>
      <td>
        <ul>
          <li>x=None</li>
          <li>y=None</li>
          <li>hue=None</li>
          <li>palette=None</li>
          <li>estimator=mean</li>
          <li>ci=95</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
          <li><strong>estimator:</strong> Statistical function to estimate (e.g., mean, median).</li>
          <li><strong>ci:</strong> Confidence interval for the estimate.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.boxplot</td>
      <td>
        <ul>
          <li>x=None</li>
          <li>y=None</li>
          <li>hue=None</li>
          <li>palette=None</li>
          <li>notch=False</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
          <li><strong>notch:</strong> If True, add notches to the boxplot.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.heatmap</td>
      <td>
        <ul>
          <li>data</li>
          <li>vmin=None</li>
          <li>vmax=None</li>
          <li>cmap='viridis'</li>
          <li>center=None</li>
          <li>annot=False</li>
          <li>fmt='.2f'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>data:</strong> Data to display.</li>
          <li><strong>vmin, vmax:</strong> Minimum and maximum values for color scaling.</li>
          <li><strong>cmap:</strong> Colormap to use.</li>
          <li><strong>center:</strong> Center value for color scaling.</li>
          <li><strong>annot:</strong> If True, annotate cells with values.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.pairplot</td>
      <td>
        <ul>
          <li>data</li>
          <li>hue=None</li>
          <li>palette=None</li>
          <li>markers='o'</li>
          <li>kind='scatter'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>data:</strong> Data to plot.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
          <li><strong>markers:</strong> Marker style.</li>
          <li><strong>kind:</strong> Type of plot to use ('scatter', 'kde').</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.jointplot</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>kind='scatter'</li>
          <li>hue=None</li>
          <li>palette=None</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>kind:</strong> Type of plot to use ('scatter', 'kde').</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.lmplot</td>
      <td>
        <ul>
          <li>x</li>
          <li>y</li>
          <li>hue=None</li>
          <li>col=None</li>
          <li>row=None</li>
          <li>fit_reg=True</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>col:</strong> Column for faceting.</li>
          <li><strong>row:</strong> Row for faceting.</li>
          <li><strong>fit_reg:</strong> If True, fit a regression line.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.violinplot</td>
      <td>
        <ul>
          <li>x=None</li>
          <li>y=None</li>
          <li>hue=None</li>
          <li>palette=None</li>
          <li>split=False</li>
          <li>inner='box'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>x, y:</strong> Data for the x and y axes.</li>
          <li><strong>hue:</strong> Variable for color encoding.</li>
          <li><strong>palette:</strong> Color palette to use.</li>
          <li><strong>split:</strong> If True, split the violins for hue variable.</li>
          <li><strong>inner:</strong> Type of plot to display inside the violin (e.g., 'box', 'quartile').</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>seaborn.heatmap</td>
      <td>
        <ul>
          <li>data</li>
          <li>vmin=None</li>
          <li>vmax=None</li>
          <li>cmap='viridis'</li>
          <li>center=None</li>
          <li>annot=False</li>
          <li>fmt='.2f'</li>
          <li>**kwargs</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><strong>data:</strong> Data to display.</li>
          <li><strong>vmin, vmax:</strong> Minimum and maximum values for color scaling.</li>
          <li><strong>cmap:</strong> Colormap to use.</li>
          <li><strong>center:</strong> Center value for color scaling.</li>
          <li><strong>annot:</strong> If True, annotate cells with values.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


```python

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate example data
np.random.seed(0)
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B'], size=100)
})

# 1. Scatter plot
sns.scatterplot(x='x', y='y', hue='category', style='category', data=data, palette='viridis')
plt.title('Scatter Plot')
plt.show()

# 2. Line plot
sns.lineplot(x='x', y='y', hue='category', data=data, markers=True, palette='coolwarm')
plt.title('Line Plot')
plt.show()

# 3. Histogram
sns.histplot(data['x'], bins=20, color='purple')
plt.title('Histogram')
plt.show()

# 4. Bar plot
sns.barplot(x='category', y='x', data=data, palette='pastel')
plt.title('Bar Plot')
plt.show()

# 5. Box plot
sns.boxplot(x='category', y='x', data=data, palette='Set2')
plt.title('Box Plot')
plt.show()

# 6. Heatmap
matrix = np.random.rand(10, 10)
sns.heatmap(matrix, cmap='coolwarm', annot=True)
plt.title('Heatmap')
plt.show()

# 7. Pair plot
sns.pairplot(data, hue='category')
plt.title('Pair Plot')
plt.show()

# 8. Joint plot
sns.jointplot(x='x', y='y', data=data, kind='scatter', hue='category')
plt.title('Joint Plot')
plt.show()

# 9. LM plot
sns.lmplot(x='x', y='y', data=data, hue='category', fit_reg=True)
plt.title('LM Plot')
plt.show()

# 10. Violin plot
sns.violinplot(x='category', y='x', data=data, palette='muted')
plt.title('Violin Plot')
plt.show()


```
# Statistics in AI and Data Science

<table>
  <thead>
    <tr>
      <th>Statistic</th>
      <th>Explanation</th>
      <th>Formula</th>
      <th>Python Function</th>
      <th>Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mean</td>
      <td>Average of all data points.</td>
      <td>\(\mu = \frac{1}{N} \sum_{i=1}^{N} x_i\)</td>
      <td>numpy.mean()</td>
      <td>
        <ul>
          <li>a (array-like)</li>
          <li>axis (int, optional)</li>
          <li>dtype (data-type, optional)</li>
          <li>keepdims (bool, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Median</td>
      <td>Middle value when data points are sorted.</td>
      <td>\( \text{Median} \)</td>
      <td>numpy.median()</td>
      <td>
        <ul>
          <li>a (array-like)</li>
          <li>axis (int, optional)</li>
          <li>overwrite_input (bool, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Mode</td>
      <td>Most frequently occurring value(s).</td>
      <td>\(\text{Mode} \)</td>
      <td>scipy.stats.mode()</td>
      <td>
        <ul>
          <li>a (array-like)</li>
          <li>axis (int, optional)</li>
          <li>keepdims (bool, optional)</li>
          <li>nan_policy ('propagate', 'raise', 'omit')</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Standard Deviation</td>
      <td>Measure of the amount of variation or dispersion of a set of values.</td>
      <td>\(\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}\)</td>
      <td>numpy.std()</td>
      <td>
        <ul>
          <li>a (array-like)</li>
          <li>axis (int, optional)</li>
          <li>dtype (data-type, optional)</li>
          <li>ddof (int, optional)</li>
          <li>keepdims (bool, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Variance</td>
      <td>Measure of how far a set of numbers are spread out from their average value.</td>
      <td>\(\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2\)</td>
      <td>numpy.var()</td>
      <td>
        <ul>
          <li>a (array-like)</li>
          <li>axis (int, optional)</li>
          <li>dtype (data-type, optional)</li>
          <li>ddof (int, optional)</li>
          <li>keepdims (bool, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Correlation Coefficient</td>
      <td>Measure of the strength and direction of a linear relationship between two variables.</td>
      <td>\( \rho_{xy} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \)</td>
      <td>numpy.corrcoef()</td>
      <td>
        <ul>
          <li>x (array-like)</li>
          <li>y (array-like, optional)</li>
          <li>rowvar (bool, optional)</li>
          <li>bias (bool, optional)</li>
          <li>ddof (int, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Linear Regression</td>
      <td>Predictive modeling technique that estimates the relationship between a dependent variable and one or more independent variables.</td>
      <td>\( y = \beta_0 + \beta_1 x \)</td>
      <td>sklearn.linear_model.LinearRegression()</td>
      <td>
        <ul>
          <li>fit_intercept (bool, optional)</li>
          <li>normalize (bool, optional)</li>
          <li>copy_X (bool, optional)</li>
          <li>n_jobs (int, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Principal Component Analysis (PCA)</td>
      <td>Dimensionality reduction technique that transforms data into a set of orthogonal components.</td>
      <td>\( X_{new} = X \cdot W \)</td>
      <td>sklearn.decomposition.PCA()</td>
      <td>
        <ul>
          <li>n_components (int, float, None, or str, optional)</li>
          <li>whiten (bool, optional)</li>
          <li>svd_solver (str, optional)</li>
          <li>tol (float, optional)</li>
          <li>random_state (int, RandomState instance, or None, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Chi-Square Test</td>
      <td>Statistical test used to determine if there is a significant association between categorical variables.</td>
      <td>\(\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}\)</td>
      <td>scipy.stats.chi2_contingency()</td>
      <td>
        <ul>
          <li>observed (array-like)</li>
          <li>correction (bool, optional)</li>
          <li>lambda_ (function, optional)</li>
          <li>weights (array-like, optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>ANOVA (Analysis of Variance)</td>
      <td>Statistical method used to compare means among multiple groups.</td>
      <td>F = \(\frac{\text{Between-group variance}}{\text{Within-group variance}}\)</td>
      <td>scipy.stats.f_oneway()</td>
      <td>
        <ul>
          <li>args (array-like)</li>
          <li>**kwargs</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

```python

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Example data
data = np.random.randn(100)
data2 = np.random.randn(100)
df = pd.DataFrame({'x': data, 'y': data2, 'category': np.random.choice(['A', 'B'], size=100)})

# 1. Mean
mean_value = np.mean(data)
print(f"Mean: {mean_value}")

# 2. Median
median_value = np.median(data)
print(f"Median: {median_value}")

# 3. Mode
mode_value = stats.mode(data)
print(f"Mode: {mode_value.mode[0]}")

# 4. Standard Deviation
std_dev = np.std(data)
print(f"Standard Deviation: {std_dev}")

# 5. Variance
variance = np.var(data)
print(f"Variance: {variance}")

# 6. Correlation Coefficient
correlation = np.corrcoef(data, data2)[0, 1]
print(f"Correlation Coefficient: {correlation}")

# 7. Linear Regression
model = LinearRegression().fit(df[['x']], df['y'])
print(f"Linear Regression Coefficients: {model.coef_}, Intercept: {model.intercept_}")

# 8. PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[['x', 'y']])
print(f"PCA Components:\n{pca.components_}")

# 9. Chi-Square Test
contingency_table = pd.crosstab(df['category'], np.random.choice(['X', 'Y'], size=100))
chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2_stat}, p-value: {p_val}")

# 10. ANOVA
anova_result = stats.f_oneway(data, data2)
print(f"ANOVA F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")


```

# Machine learning Algorithms

## Supervised Learning

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Explanation</th>
      <th>Formula</th>
      <th>Python Function</th>
      <th>Key Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Regression</td>
      <td>Predicts continuous values using a linear combination of input features.</td>
      <td>\( y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \)</td>
      <td>sklearn.linear_model.LinearRegression()</td>
      <td>
        <ul>
          <li>fit_intercept</li>
          <li>normalize</li>
          <li>copy_X</li>
          <li>n_jobs</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Predicts the probability of a binary outcome using a logistic function.</td>
      <td>\( P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}} \)</td>
      <td>sklearn.linear_model.LogisticRegression()</td>
      <td>
        <ul>
          <li>penalty</li>
          <li>C</li>
          <li>solver</li>
          <li>max_iter</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Splits data into subsets based on the value of input features.</td>
      <td>\( \text{Gini Impurity or Entropy for split criterion} \)</td>
      <td>sklearn.tree.DecisionTreeClassifier()</td>
      <td>
        <ul>
          <li>criterion</li>
          <li>splitter</li>
          <li>max_depth</li>
          <li>min_samples_split</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Ensemble of decision trees that improves performance by averaging predictions.</td>
      <td>\( \text{Average of predictions from multiple decision trees} \)</td>
      <td>sklearn.ensemble.RandomForestClassifier()</td>
      <td>
        <ul>
          <li>n_estimators</li>
          <li>criterion</li>
          <li>max_depth</li>
          <li>min_samples_split</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>K-Nearest Neighbors (KNN)</td>
      <td>Classifies data points based on the labels of the k-nearest data points.</td>
      <td>\( \text{Distance Metric: Euclidean or Manhattan} \)</td>
      <td>sklearn.neighbors.KNeighborsClassifier()</td>
      <td>
        <ul>
          <li>n_neighbors</li>
          <li>weights</li>
          <li>algorithm</li>
          <li>leaf_size</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Support Vector Machines (SVM)</td>
      <td>Finds the hyperplane that best separates classes by maximizing the margin.</td>
      <td>\( y = w^T x + b \)</td>
      <td>sklearn.svm.SVC()</td>
      <td>
        <ul>
          <li>C</li>
          <li>kernel</li>
          <li>gamma</li>
          <li>degree</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Naive Bayes</td>
      <td>Classifies data based on applying Bayes' Theorem with strong independence assumptions.</td>
      <td>\( P(y|x) = \frac{P(x|y)P(y)}{P(x)} \)</td>
      <td>sklearn.naive_bayes.GaussianNB()</td>
      <td>
        <ul>
          <li>priors</li>
          <li>var_smoothing</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Gradient Boosting</td>
      <td>Builds an ensemble of weak models (usually decision trees) to improve performance.</td>
      <td>\( F(x) = F_{m-1}(x) + h_m(x) \)</td>
      <td>sklearn.ensemble.GradientBoostingClassifier()</td>
      <td>
        <ul>
          <li>learning_rate</li>
          <li>n_estimators</li>
          <li>max_depth</li>
          <li>subsample</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Artificial Neural Networks (ANN)</td>
      <td>Uses layers of neurons to learn complex patterns in data.</td>
      <td>\( y = f(W^T x + b) \)</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>layers</li>
          <li>optimizer</li>
          <li>loss</li>
          <li>metrics</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Linear Discriminant Analysis (LDA)</td>
      <td>Finds a linear combination of features that best separates classes.</td>
      <td>\( y = w^T x + b \)</td>
      <td>sklearn.discriminant_analysis.LinearDiscriminantAnalysis()</td>
      <td>
        <ul>
          <li>solver</li>
          <li>shrinkage</li>
          <li>priors</li>
          <li>n_components</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

```python 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample dataset
X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"Linear Regression Score: {lin_reg.score(X_test, y_test)}")

# 2. Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f"Logistic Regression Score: {log_reg.score(X_test, y_test)}")

# 3. Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print(f"Decision Tree Score: {tree.score(X_test, y_test)}")

# 4. Random Forest
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
print(f"Random Forest Score: {forest.score(X_test, y_test)}")

# 5. K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(f"KNN Score: {knn.score(X_test, y_test)}")

# 6. Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
print(f"SVM Score: {svm.score(X_test, y_test)}")

# 7. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print(f"Naive Bayes Score: {nb.score(X_test, y_test)}")

# 8. Gradient Boosting
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
print(f"Gradient Boosting Score: {gbc.score(X_test, y_test)}")

# 9. Neural Network (Sequential Model with Keras)
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Neural Network Accuracy: {accuracy}")

# 10. Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(f"LDA Score: {lda.score(X_test, y_test)}")


```


## unsupervised learning

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Explanation</th>
      <th>Formula</th>
      <th>Python Function</th>
      <th>Key Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>K-Means Clustering</td>
      <td>Partitions data into K clusters by minimizing within-cluster variance.</td>
      <td>\( \sum_{i=1}^{K} \sum_{x_j \in C_i} \|x_j - \mu_i\|^2 \)</td>
      <td>sklearn.cluster.KMeans()</td>
      <td>
        <ul>
          <li>n_clusters</li>
          <li>init</li>
          <li>n_init</li>
          <li>max_iter</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Hierarchical Clustering</td>
      <td>Builds a tree of clusters by iteratively merging or splitting clusters.</td>
      <td>\( D(i,j) = \text{Linkage}(C_i, C_j) \)</td>
      <td>scipy.cluster.hierarchy.linkage()</td>
      <td>
        <ul>
          <li>method ('single', 'complete', 'average', etc.)</li>
          <li>metric</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>DBSCAN</td>
      <td>Density-based clustering that identifies clusters of high density.</td>
      <td>N/A</td>
      <td>sklearn.cluster.DBSCAN()</td>
      <td>
        <ul>
          <li>eps</li>
          <li>min_samples</li>
          <li>metric</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Principal Component Analysis (PCA)</td>
      <td>Reduces the dimensionality of data by transforming it into a set of orthogonal components.</td>
      <td>\( X_{\text{new}} = X \cdot W \)</td>
      <td>sklearn.decomposition.PCA()</td>
      <td>
        <ul>
          <li>n_components</li>
          <li>whiten</li>
          <li>svd_solver</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>t-SNE (t-Distributed Stochastic Neighbor Embedding)</td>
      <td>Reduces the dimensionality of data while preserving local structure.</td>
      <td>N/A</td>
      <td>sklearn.manifold.TSNE()</td>
      <td>
        <ul>
          <li>n_components</li>
          <li>perplexity</li>
          <li>learning_rate</li>
          <li>n_iter</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Gaussian Mixture Models (GMM)</td>
      <td>Assumes data is generated from a mixture of several Gaussian distributions.</td>
      <td>\( P(x) = \sum_{i=1}^{K} \pi_i \mathcal{N}(x | \mu_i, \Sigma_i) \)</td>
      <td>sklearn.mixture.GaussianMixture()</td>
      <td>
        <ul>
          <li>n_components</li>
          <li>covariance_type</li>
          <li>tol</li>
          <li>max_iter</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Self-Organizing Maps (SOM)</td>
      <td>A type of neural network used for clustering and dimensionality reduction.</td>
      <td>N/A</td>
      <td>minisom.MiniSom()</td>
      <td>
        <ul>
          <li>x (grid width)</li>
          <li>y (grid height)</li>
          <li>input_len</li>
          <li>sigma</li>
          <li>learning_rate</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Independent Component Analysis (ICA)</td>
      <td>Decomposes data into independent components.</td>
      <td>\( X = A \cdot S \)</td>
      <td>sklearn.decomposition.FastICA()</td>
      <td>
        <ul>
          <li>n_components</li>
          <li>algorithm</li>
          <li>whiten</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Autoencoders</td>
      <td>A neural network architecture used for unsupervised learning, typically for dimensionality reduction.</td>
      <td>N/A</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>layers</li>
          <li>optimizer</li>
          <li>loss</li>
          <li>metrics</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from minisom import MiniSom

# Sample dataset
X = np.random.rand(100, 4)

# 1. K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(f"K-Means Labels: {kmeans.labels_}")

# 2. Hierarchical Clustering
Z = linkage(X, method='ward')
dendrogram(Z)
plt.show()

# 3. DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X)
print(f"DBSCAN Labels: {dbscan.labels_}")

# 4. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA Components:\n{pca.components_}")

# 5. t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
print(f"t-SNE result shape: {X_tsne.shape}")

# 6. Gaussian Mixture Models (GMM)
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
print(f"GMM Labels: {gmm.predict(X)}")

# 7. Self-Organizing Map (SOM)
som = MiniSom(x=7, y=7, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)
print("SOM Training Complete")

# 8. Independent Component Analysis (ICA)
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X)
print(f"ICA Components:\n{X_ica}")

# 9. Autoencoder (Neural Network)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Autoencoder model
autoencoder = Sequential([
    Dense(8, input_shape=(X.shape[1],), activation='relu'),
    Dense(2, activation='relu'),  # Latent space
    Dense(8, activation='relu'),
    Dense(X.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=10, batch_size=10, verbose=0)
encoded_data = autoencoder.predict(X)
print(f"Autoencoder Encoded Data Shape: {encoded_data.shape}")


```
# Deep learning

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Explanation</th>
      <th>Formula</th>
      <th>Python Function</th>
      <th>Key Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Artificial Neural Networks (ANN)</td>
      <td>A feed-forward neural network with one or more hidden layers used for classification or regression.</td>
      <td>\( y = f(W^T x + b) \)</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>layers</li>
          <li>activation</li>
          <li>optimizer</li>
          <li>loss</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Convolutional Neural Networks (CNN)</td>
      <td>A type of deep neural network used primarily for image processing tasks, like classification and object detection.</td>
      <td>\( y = \text{Conv2D}(X) + \text{Pooling Layers} \)</td>
      <td>tensorflow.keras.layers.Conv2D()</td>
      <td>
        <ul>
          <li>filters</li>
          <li>kernel_size</li>
          <li>activation</li>
          <li>strides</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Recurrent Neural Networks (RNN)</td>
      <td>A class of neural networks designed for sequence modeling tasks such as time-series prediction and NLP.</td>
      <td>\( h_t = f(W \cdot h_{t-1} + U \cdot x_t) \)</td>
      <td>tensorflow.keras.layers.SimpleRNN()</td>
      <td>
        <ul>
          <li>units</li>
          <li>activation</li>
          <li>return_sequences</li>
          <li>dropout</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Long Short-Term Memory (LSTM)</td>
      <td>A type of RNN designed to handle long-term dependencies by using gates to regulate information flow.</td>
      <td>\( f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \)</td>
      <td>tensorflow.keras.layers.LSTM()</td>
      <td>
        <ul>
          <li>units</li>
          <li>activation</li>
          <li>recurrent_activation</li>
          <li>return_sequences</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Gated Recurrent Units (GRU)</td>
      <td>A simplified version of LSTM, with fewer gates and parameters, used for sequence prediction.</td>
      <td>\( r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \)</td>
      <td>tensorflow.keras.layers.GRU()</td>
      <td>
        <ul>
          <li>units</li>
          <li>activation</li>
          <li>recurrent_activation</li>
          <li>return_sequences</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Autoencoders</td>
      <td>An unsupervised learning algorithm where the model tries to reconstruct the input from a compressed representation.</td>
      <td>N/A</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>encoder layers</li>
          <li>decoder layers</li>
          <li>loss</li>
          <li>optimizer</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Generative Adversarial Networks (GAN)</td>
      <td>A class of models used for generative tasks, consisting of a generator and a discriminator that are trained simultaneously.</td>
      <td>N/A</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>generator model</li>
          <li>discriminator model</li>
          <li>loss</li>
          <li>optimizer</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Deep Belief Networks (DBN)</td>
      <td>A type of generative deep learning model consisting of multiple layers of latent variables.</td>
      <td>N/A</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>layers</li>
          <li>learning rate</li>
          <li>activation function</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Transformer Networks</td>
      <td>A deep learning model used for sequence-to-sequence tasks, typically in natural language processing.</td>
      <td>N/A</td>
      <td>tensorflow.keras.layers.MultiHeadAttention()</td>
      <td>
        <ul>
          <li>num_heads</li>
          <li>key_dim</li>
          <li>dropout</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

```python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras.layers import MultiHeadAttention

# Sample Data
X_train = tf.random.normal((100, 28, 28, 1))  # For CNN
y_train = tf.random.uniform((100,), maxval=10, dtype=tf.int32)

# 1. Artificial Neural Networks (ANN)
ann = Sequential([
    Dense(64, activation='relu', input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=5)

# 2. Convolutional Neural Networks (CNN)
cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Flatten(),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=5)

# 3. Recurrent Neural Networks (RNN)
rnn = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn.fit(X_train, y_train, epochs=5)

# 4. Long Short-Term Memory (LSTM)
lstm = Sequential([
    LSTM(64, activation='tanh', input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm.fit(X_train, y_train, epochs=5)

# 5. Gated Recurrent Units (GRU)
gru = Sequential([
    GRU(64, activation='tanh', input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
gru.fit(X_train, y_train, epochs=5)

# 6. Autoencoders
autoencoder = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(28*28, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
X_flat = tf.reshape(X_train, (-1, 28*28))
autoencoder.fit(X_flat, X_flat, epochs=5)

# 7. Generative Adversarial Networks (GAN)
# Generator
generator = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(28*28, activation='sigmoid')
])

# Discriminator
discriminator = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(1, activation='sigmoid')
])

# GAN training loop can be complex; this is the general setup for generator and discriminator.

# 8. Transformer Networks
input_layer = Input(shape=(10, 64))  # Example input of shape (batch, seq_len, feature_dim)
transformer = MultiHeadAttention(num_heads=4, key_dim=64)(input_layer, input_layer)
print(f"Transformer output shape: {transformer.shape}")

```
# NLP

<table>
  <thead>
    <tr>
      <th>Technique</th>
      <th>Explanation</th>
      <th>Formula</th>
      <th>Python Function</th>
      <th>Key Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tokenization</td>
      <td>Splits text into smaller units like words or sentences.</td>
      <td>N/A</td>
      <td>nltk.word_tokenize(), spacy.tokenizer()</td>
      <td>
        <ul>
          <li>text</li>
          <li>language (optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Stop Words Removal</td>
      <td>Removes common words that are usually not useful for analysis.</td>
      <td>N/A</td>
      <td>nltk.corpus.stopwords.words()</td>
      <td>
        <ul>
          <li>language</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Stemming</td>
      <td>Reduces words to their root form.</td>
      <td>N/A</td>
      <td>nltk.stem.PorterStemmer()</td>
      <td>
        <ul>
          <li>word</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Lemmatization</td>
      <td>Reduces words to their base or dictionary form.</td>
      <td>N/A</td>
      <td>nltk.stem.WordNetLemmatizer()</td>
      <td>
        <ul>
          <li>word</li>
          <li>pos (optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Part-of-Speech (POS) Tagging</td>
      <td>Assigns parts of speech to each word in a sentence.</td>
      <td>N/A</td>
      <td>nltk.pos_tag(), spacy.pos_tag()</td>
      <td>
        <ul>
          <li>text</li>
          <li>language (optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Named Entity Recognition (NER)</td>
      <td>Identifies named entities like people, organizations, and locations in text.</td>
      <td>N/A</td>
      <td>spacy.ner(), nltk.chunk.ne_chunk()</td>
      <td>
        <ul>
          <li>text</li>
          <li>language (optional)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Bag of Words (BoW)</td>
      <td>Represents text data by counting word occurrences in a document.</td>
      <td>\( \text{vector} = [\text{count}(w_1), \text{count}(w_2), \ldots] \)</td>
      <td>sklearn.feature_extraction.text.CountVectorizer()</td>
      <td>
        <ul>
          <li>ngram_range</li>
          <li>stop_words</li>
          <li>max_features</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>TF-IDF</td>
      <td>Term Frequency-Inverse Document Frequency, measures word importance.</td>
      <td>\( \text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w) \)</td>
      <td>sklearn.feature_extraction.text.TfidfVectorizer()</td>
      <td>
        <ul>
          <li>ngram_range</li>
          <li>stop_words</li>
          <li>max_features</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Word Embeddings</td>
      <td>Maps words to dense vectors in a continuous vector space.</td>
      <td>N/A</td>
      <td>gensim.models.Word2Vec()</td>
      <td>
        <ul>
          <li>sentences</li>
          <li>vector_size</li>
          <li>window</li>
          <li>min_count</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Transformers</td>
      <td>Models that use attention mechanisms to handle sequences.</td>
      <td>N/A</td>
      <td>transformers.BertModel()</td>
      <td>
        <ul>
          <li>pretrained_model_name_or_path</li>
          <li>config</li>
          <li>tokenizer</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Sequence-to-Sequence Models (Seq2Seq)</td>
      <td>Models for tasks like translation where input and output are sequences.</td>
      <td>N/A</td>
      <td>tensorflow.keras.Sequential()</td>
      <td>
        <ul>
          <li>encoder</li>
          <li>decoder</li>
          <li>sequence_length</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>



```python

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

# Sample text
text = "Natural Language Processing (NLP) is a field of artificial intelligence."

# 1. Tokenization
tokens = word_tokenize(text)
print(f"Tokens: {tokens}")

# 2. Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print(f"Filtered Words: {filtered_words}")

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]
print(f"Stemmed Words: {stemmed_words}")

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
print(f"Lemmatized Words: {lemmatized_words}")

# 5. Part-of-Speech (POS) Tagging
pos_tags = pos_tag(tokens)
print(f"POS Tags: {pos_tags}")

# 6. Named Entity Recognition (NER)
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# 7. Bag of Words (BoW)
corpus = [text]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(f"BoW Matrix:\n{X.toarray()}")

# 8. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
print(f"TF-IDF Matrix:\n{X_tfidf.toarray()}")

# 9. Word Embeddings
sentences = [["natural", "language", "processing", "is", "fun"],
             ["deep", "learning", "with", "transformers"]]
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=0)
vector = model.wv['language']
print(f"Word2Vec Vector for 'language': {vector}")

# 10. Transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(text, return_tensors='pt')
model = BertModel.from_pretrained('bert-base-uncased')
outputs = model(**inputs)
print(f"Transformer Outputs: {outputs.last_hidden_state}")


```


