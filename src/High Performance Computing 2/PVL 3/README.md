<p>
  <strong>Experiment:</strong> Strong Scaling Analysis (Fixed Problem Size N=10000)<br>
  <strong>Objective:</strong> Measure the reduction in solution time as the number of processing cores increases from 1 to 16.
</p>

<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>Cores (p)</th>
      <th>Solver Time (s)</th>
      <th>Speedup S(p) = T1 / Tp</th>
      <th>Efficiency E(p) = S(p) / p</th>
      <th>Status vs Reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>257.02</td>
      <td>1.00x</td>
      <td>100.0%</td>
      <td>Faster (Ref: 1700s)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>130.86</td>
      <td>1.96x</td>
      <td>98.2%</td>
      <td>Faster (Ref: 858s)</td>
    </tr>
    <tr>
      <td>4</td>
      <td>65.53</td>
      <td>3.92x</td>
      <td>98.0%</td>
      <td>Faster (Ref: 450s)</td>
    </tr>
    <tr>
      <td>8</td>
      <td>33.21</td>
      <td>7.74x</td>
      <td>96.7%</td>
      <td>Faster (Ref: 269s)</td>
    </tr>
    <tr>
      <td>16</td>
      <td>18.15</td>
      <td>14.16x</td>
      <td>88.5%</td>
      <td>Faster (Ref: 163s)</td>
    </tr>
  </tbody>
</table>

