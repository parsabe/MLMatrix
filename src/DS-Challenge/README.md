<h1>SWaT and WADI Dataset Overview</h1>

<h2>1. SWaT (Secure Water Treatment) Dataset</h2>

<ul>
  <li><strong>Origin & Purpose:</strong>
    Developed by the iTrust Centre (Singapore University of Technology & Design) to study cyber–physical security in industrial control systems.
    The testbed replicates a water treatment plant with six stages (P1–P6), controlled via PLCs, SCADA, and HMIs.
  </li>
  <li><strong>Collection Period:</strong>
    Approximately 11 days: 7 days of normal operation, followed by 4 days with 36 distinct cyber-attacks.
  </li>
  <li><strong>Data Details:</strong>
    <ul>
      <li>~51 variables: 25 sensors and 26 actuators/pumps, sampled at 1 Hz.</li>
      <li>Includes process data (sensor readings, actuator statuses) and network traffic logs.</li>
      <li>Records are labeled as under attack or normal.</li>
    </ul>
  </li>
  <li><strong>Attack Types:</strong>
    Sensor spoofing, actuator manipulation, replay attacks, and network-based injection.
  </li>
  <li><strong>Research & Benchmarks:</strong>
    <ul>
      <li>Used widely in ML anomaly detection studies (SVM, CNNs, LSTM, GRU, VAE, attention models).</li>
      <li>F1-scores in published work often exceed 80–95%.</li>
      <li>Used by Kaspersky MLAD, which detected 23 of 34 attacks and 7 unknown anomalies.</li>
    </ul>
  </li>
</ul>

<h2>2. WADI (Water Distribution) Dataset</h2>

<ul>
  <li><strong>Context:</strong>
    Companion dataset to SWaT, simulating a water distribution system.
  </li>
  <li><strong>Data Size:</strong>
    Contains approximately 123 sensor and actuator variables collected over multiple days.
  </li>
  <li><strong>Usage:</strong>
    Frequently used together with SWaT in intrusion detection system research.
  </li>
</ul>

<h2>3. SWaT in Academic Research</h2>

<ul>
  <li><strong>Performance Benchmarks:</strong>
    <ul>
      <li>LSTM models achieve around 98% attack detection accuracy.</li>
      <li>Sequence-to-sequence models detect 29 of 36 attacks effectively.</li>
      <li>CNNs are faster and lighter than RNNs for some detection tasks.</li>
      <li>Latest techniques (e.g., VAE + attention) improve cross-stage detection.</li>
    </ul>
  </li>
  <li><strong>Challenges:</strong>
    <ul>
      <li>Complex multi-stage attack detection is still difficult.</li>
      <li>Precision–recall balance is challenging.</li>
      <li>Hybrid process + network feature models show promise.</li>
    </ul>
  </li>
</ul>

<h2>4. Dataset Comparison Table</h2>

<table border="1">
  <tr>
    <th>Feature</th>
    <th>SWaT</th>
    <th>WADI</th>
  </tr>
  <tr>
    <td>Duration</td>
    <td>~11 days (7 normal, 4 attack)</td>
    <td>Multi-day timeline</td>
  </tr>
  <tr>
    <td>Data Type</td>
    <td>~51 process vars + network logs</td>
    <td>~120+ sensor/actuator vars</td>
  </tr>
  <tr>
    <td>Sampling Frequency</td>
    <td>1 Hz</td>
    <td>1 Hz</td>
  </tr>
  <tr>
    <td>Attack Count</td>
    <td>36 distinct attacks</td>
    <td>Multiple attack scenarios</td>
  </tr>
  <tr>
    <td>Common ML Methods</td>
    <td>SVM, RF, CNN, RNN, VAE, attention</td>
    <td>Same</td>
  </tr>
  <tr>
    <td>F1 Scores</td>
    <td>80–98%, advanced methods >95%</td>
    <td>Similar</td>
  </tr>
</table>

