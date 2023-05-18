# MILP approach for solving the Unit Commitment Problem

## Introduction
Stuff

## Example

|Unit|P_min (MW)|P_max (MW)|Min up (hr)|Min down (hr)|No-load cost (\$)|Marginal cost (\$/MWh)|Startup cost (\$)|Initial status|
|---|---| ---| ---|   ---|   ---|   ---|  ---|   ---|
|A   | 150| 250| 3|   3|   0|   10|  1000|    ON|
|B   | 50| 100| 2|   1|   0|   12|  600| OFF|
|C   |10  |50  |1   |1   |0   |20  |100 |OFF|

Hourly demand profile is shown below

|1|2|3|
|---|---|---|
|150|300|200|

Callout the simplifications and assumptions in the above example

- marginal costs are often a function of time
- startup costs are often modeled by an xponeential function of time. which causes additional complications. Various approaches have been proposed
- no cool-down cost
- No-load costs are often non-zero because >>>>

## A simple economic dispatch model with LP

Simplifying assumptions:
1. All units are running and available at all times. I.e. we don't have the choice to turn a unit ON/OFF.
2. Startup costs are ignored.
3. Minimum up- and down-time constraints do not apply.
4. Initial states of the units are ignored.

We are interested in finding the optimal power output of each unit in each time period that minimizes the total marginal cost.

Parameters:

$c_j$ := Marginal cost of running unit $j$ (\$/MWh)

$D_t$ := Total demand in time slot $t$ (MW)

$P_{j, min}$ := Minimum recommended power output for unit $j$ (MW)

$P_{j, max}$ := Maximum recommended power output for unit $j$ (MW)

$J$ := Indexed set of all generators (a.k.a. units)

$T$ := Indexed set of all time slots/periods (a.k.a. planning horizon)

Variables:

$p_{j,t}$ := Power output of unit $j$ in time slot $t$ (MW)

Since we are only considering the marginal costs of runnint the units, the objective is pretty straightforward

\begin{equation*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t}
\end{equation*}

There are two sets of constraints we need to consider. The power output of each generating unit must be within the recommended output range. And the total power output of all generating units must satisfy the demand in each time period. We can write these constraints as follows:

\begin{gather*}
P_{j, min} \le p_{j, t} \le P_{j, max}\ \text{(Output Range)}\\
\sum_{j \in J} p_{j, t} \ge D_t\ \text{(Demand)}\\
\end{gather*}

We get the following model

\begin{gather*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t}\\\tag{LP}
     \text{s.t.}\quad P_{j, min} \le p_{j, t} \le P_{j, max}\\
     \sum_{j \in J} p_{j, t} \ge D_t
\end{gather*}

Let us solve (LP) using `cvxpy`.


```python
import cvxpy as cp
import cvxopt as cv
import numpy as np
import pandas as pd

from solver.utils import get_result_summary, prettify

# parameters
cA = 10; cB = 12; cC = 20;

p_min_A = 150
p_min_B = 50
p_min_C = 10

p_max_A = 250
p_max_B = 100
p_max_C = 50

D1 = 150; D2 = 300; D3 = 200
```


```python
#variables
pA1 = cp.Variable(1, nonneg=True, name='pA1')
pA2 = cp.Variable(1, nonneg=True, name='pA2')
pA3 = cp.Variable(1, nonneg=True, name='pA3')

pB1 = cp.Variable(1, nonneg=True, name='pB1')
pB2 = cp.Variable(1, nonneg=True, name='pB2')
pB3 = cp.Variable(1, nonneg=True, name='pB3')

pC1 = cp.Variable(1, nonneg=True, name='pC1')
pC2 = cp.Variable(1, nonneg=True, name='pC2')
pC3 = cp.Variable(1, nonneg=True, name='pC3')
```


```python
obj_LP = cp.Minimize(
    cA * (pA1 + pA2 + pA3) +
    cB * (pB1 + pB2 + pB3) +
    cC * (pC1 + pC2 + pC3)
)
```


```python
cons_LP = [

    # output range
    pA1 >= p_min_A,
    pA1 <= p_max_A,
    pA2 >= p_min_A,
    pA2 <= p_max_A,
    pA3 >= p_min_A,
    pA3 <= p_max_A,
    
    pB1 >= p_min_B,
    pB1 <= p_max_B,
    pB2 >= p_min_B,
    pB2 <= p_max_B,
    pB3 >= p_min_B,
    pB3 <= p_max_B,
    
    pC1 >= p_min_C,
    pC1 <= p_max_C,
    pC2 >= p_min_C,
    pC2 <= p_max_C,
    pC3 >= p_min_C,
    pC3 <= p_max_C,

    # demand constraint
    pA1 + pB1 + pC1 >= D1,
    pA2 + pB2 + pC2 >= D2,
    pA3 + pB3 + pC3 >= D3,
]
```


```python
LP = cp.Problem(obj_LP, cons_LP)
```


```python
LP.solve();
```


```python
summary_LP = get_result_summary(LP)
```


```python
summary_LP['status']
```




    'optimal'




```python
summary_LP['optimal_value']
```




    7799.999998684703




```python
prettify(summary_LP['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pA1</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pA2</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pA3</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pB1</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pB2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pB3</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pC1</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pC2</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pC3</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



The minimum marginal cost of running the three units over the horizon is <mark> $7799.99 </mark>. Also note that unit A runs at minimum capacity during periods 1 and 3. During the peak period 2, the output of unit A is increased to meet the increased demand while units B and C continue to run at minimum capacity throughout the horizon. This makes sense because units B and C have a higher marginal cost than unit A. Next, we consider a model which allows us to choose which units are ON during each slot.

## A first MIP formulation

Here, we get rid of the first assumption:
1. ~All units are running and available at all times. I.e. we don't have the choice to turn a unit ON/OFF.~
2. Startup costs are ignored.
3. Minimum up- and down-time constraints do not apply.
4. Initial states of the units are ignored.

We now have the choice to turn one or more units ON/OFF during any of the time slots. We will use a set of binary variables to model this choice.

\begin{equation*}
u_{j, t} :=
\begin{cases}
   1 &\text{if unit $j$ is ON in slot $t$}\\
   0 &\text{otherwise }
\end{cases}
\end{equation*}

Our objective as well as the demand constraint remain unchanged. However, the output range constraint needs to be modified in order to incorporate the new $u_{j, t}$ variables.

\begin{align*}
P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\ \text{(Output Range)}\\
\end{align*}

Note that in the above formulation, if the unit $j$ is OFF in period $t$, the power output $p_{j, t}$ is forced to be zero, so that we don't have any power output contribution from an OFF unit. Below is the new MIP model



\begin{gather*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t}\\ \tag{MIP-1}
     \text{s.t.}\quad P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\\ 
     \sum_{j \in J} p_{j, t} \ge D_t\\
     u_{j, t} \in \{0, 1\} \ \forall\ j, t
\end{gather*}

Let's solve (MIP-1) with `cxvpy`


```python
# ON/OFF variables
uA1 = cp.Variable(1, boolean=True, name='uA1')
uA2 = cp.Variable(1, boolean=True, name='uA2')
uA3 = cp.Variable(1, boolean=True, name='uA3')

uB1 = cp.Variable(1, boolean=True, name='uB1')
uB2 = cp.Variable(1, boolean=True, name='uB2')
uB3 = cp.Variable(1, boolean=True, name='uB3')

uC1 = cp.Variable(1, boolean=True, name='uC1')
uC2 = cp.Variable(1, boolean=True, name='uC2')
uC3 = cp.Variable(1, boolean=True, name='uC3')
```


```python
# Objective remains unchanged
obj_MIP1 = LP.objective
```


```python
# New constraints
cons_MIP1 = [
    # output range
    pA1 >= p_min_A * uA1,
    pA1 <= p_max_A * uA1,
    pA2 >= p_min_A * uA2,
    pA2 <= p_max_A * uA2,
    pA3 >= p_min_A * uA3,
    pA3 <= p_max_A * uA3,
    
    pB1 >= p_min_B * uB1,
    pB1 <= p_max_B * uB1,
    pB2 >= p_min_B * uB2,
    pB2 <= p_max_B * uB2,
    pB3 >= p_min_B * uB3,
    pB3 <= p_max_B * uB3,
    
    pC1 >= p_min_C * uC1,
    pC1 <= p_max_C * uC1,
    pC2 >= p_min_C * uC2,
    pC2 <= p_max_C * uC2,
    pC3 >= p_min_C * uC3,
    pC3 <= p_max_C * uC3,
] + LP.constraints[1:] # demand constraint remains unchanged
```


```python
MIP1 = cp.Problem(obj_MIP1, cons_MIP1)
```


```python
MIP1.solve();
```


```python
summary_MIP1 = get_result_summary(MIP1)
```


```python
summary_MIP1['status']
```




    'optimal'




```python
summary_MIP1['optimal_value']
```




    7700.0




```python
prettify(summary_MIP1['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pA1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pA2</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pA3</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pB1</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pB2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pB3</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pC1</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pC2</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pC3</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>uA1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>uA2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>uA3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>uB1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>uB2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>uB3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>uC1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>uC2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>uC3</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



The optimal cost with the basic MIP model turns out to be <mark>$7700</mark>. Note that this is <mark>\$100 cheaper</mark> than the result of the (LP) model. Indeed, being able to choose which units to commit during each time slot has saved us some money! Note that the model chose to ????????????. This makes sense because unit A is the cheapest to run. For the high demand periods, all three generators were needed, but they were all running at their respective minimum output.

## An MIP model with startup costs

Here, we get rid of the second assumption:
1. ~All units are running and available at all times. I.e. we don't have the choice to turn a unit ON/OFF.~
2. ~Startup costs are ignored.~
3. ~Initial states of the units are ignored.~
4. Minimum up- and down-time constraints do not apply.

We now consider the startup costs of the units. A unit incurs a startup cost in a time period only if it was started up in that period. We need a binary variable to indicate if a unit was turnd ON in a given period. If so, the unit will incur the startup cost in that period in addition to the marginal cost. (For simplicity, we have assumed the cool-down costs to be $0$.)

We introduce a new parameter to denote the startup costs of the units

\begin{align*}
    c_j^u := \text{Startup cost of unit } j\ (\$)
\end{align*}

and a new binary variable

\begin{equation*}
\alpha_{j, t} :=
\begin{cases}
   1 &\text{if unit $j$ was started in period $t$}\\
   0 &\text{otherwise }
\end{cases}
\end{equation*}

The new objective is given by:

\begin{align*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t} + \alpha_{j, t} c_j^u\\
\end{align*}

The demand constraint as well as the output range constraints remain the same as for (MIP-1). However, we need a new constraint in order to ensure that $\alpha_{j, t} = 1$ if and only if unit $j$ was started up in period $t$. This constraint can be modelled by the below function

\begin{align*}
    \alpha_{j, t} = \lfloor \frac{u_{j, t} - u_{j, t-1} + 1}{2} \rfloor
\end{align*}

The above non-linear function can be expressed in terms of linear constraints as follows:

\begin{align*}
    \alpha_{j, t} &\le \frac{u_{j, t} - u_{j, t-1} + 1}{2},\ \ \ \  \alpha_{j, t} + 1 \ge \frac{u_{j, t} - u_{j, t-1} + 1}{2} + .25\qquad \text{(Startup)}
\end{align*}

Our new model can be written as:

\begin{gather*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t} + \alpha_{j, t} c_j^u\\
     \text{s.t.}\quad P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\\ 
     \sum_{j \in J} p_{j, t} \ge D_t\\ \tag{MIP-2}
     \alpha_{j, t} \le \frac{u_{j, t} - u_{j, t-1} + 1}{2}\\
     \alpha_{j, t} + 1 \ge \frac{u_{j, t} - u_{j, t-1} + 1}{2} + .25\\
     u_{j, t} \in \{0, 1\} \ \forall\ j, t\\
     \alpha_{j, t} \in \{0, 1\} \ \forall\ j, t
\end{gather*}

Lets solve (MIP-2) with `cvxpy`


```python
# parameters
## startup costs
cA_up = 1000
cB_up = 600
cC_up = 100

## initial states
uA0 = 1 # ON
uB0 = 0 # OFF
uC0 = 0 # OFF
```


```python
# startup variables
alpha_A1 = cp.Variable(1, boolean=True, name='alpha_A1')
alpha_A2 = cp.Variable(1, boolean=True, name='alpha_A2')
alpha_A3 = cp.Variable(1, boolean=True, name='alpha_A3')


alpha_B1 = cp.Variable(1, boolean=True, name='alpha_B1')
alpha_B2 = cp.Variable(1, boolean=True, name='alpha_B2')
alpha_B3 = cp.Variable(1, boolean=True, name='alpha_B3')


alpha_C1 = cp.Variable(1, boolean=True, name='alpha_C1')
alpha_C2 = cp.Variable(1, boolean=True, name='alpha_C2')
alpha_C3 = cp.Variable(1, boolean=True, name='alpha_C3')
```


```python
# objective
obj_MIP2 = cp.Minimize(
    cA * (pA1 + pA2 + pA3) + cA_up * (alpha_A1 + alpha_A2 + alpha_A3) +
    cB * (pB1 + pB2 + pB3) + cB_up * (alpha_B1 + alpha_B2 + alpha_B3) +
    cC * (pC1 + pC2 + pC3) + cC_up * (alpha_C1 + alpha_C2 + alpha_C3)
)
```


```python
# constraints
cons_MIP2 = MIP1.constraints + [
    alpha_A1 <= (uA1 - uA0 + 1)/2,
    alpha_A1 >= (uA1 - uA0 + 1)/2 - .75,
    alpha_A2 <= (uA2 - uA1 + 1)/2,
    alpha_A2 >= (uA2 - uA1 + 1)/2 - .75,
    alpha_A3 <= (uA3 - uA2 + 1)/2,
    alpha_A3 >= (uA3 - uA2 + 1)/2 - .75,
    
    alpha_B1 <= (uB1 - uB0 + 1)/2,
    alpha_B1 >= (uB1 - uB0 + 1)/2 - .75,
    alpha_B2 <= (uB2 - uB1 + 1)/2,
    alpha_B2 >= (uB2 - uB1 + 1)/2 - .75,
    alpha_B3 <= (uB3 - uB2 + 1)/2,
    alpha_B3 >= (uB3 - uB2 + 1)/2 - .75,
    
    alpha_C1 <= (uC1 - uC0 + 1)/2,
    alpha_C1 >= (uC1 - uC0 + 1)/2 - .75,
    alpha_C2 <= (uC2 - uC1 + 1)/2,
    alpha_C2 >= (uC2 - uC1 + 1)/2 - .75,
    alpha_C3 <= (uC3 - uC2 + 1)/2,
    alpha_C3 >= (uC3 - uC2 + 1)/2 - .75,
]
```


```python
MIP2 = cp.Problem(obj_MIP2, cons_MIP2)
```


```python
MIP2.solve();
```


```python
summary = get_result_summary(MIP2)
```


```python

```


```python
prettify(summary['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_A1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_A2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_A3</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_B1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha_B2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_B3</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_C1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha_C2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_C3</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pA1</th>
      <td>150.0</td>
    </tr>
    <tr>
      <th>pA2</th>
      <td>240.0</td>
    </tr>
    <tr>
      <th>pA3</th>
      <td>150.0</td>
    </tr>
    <tr>
      <th>pB1</th>
      <td>50.0</td>
    </tr>
    <tr>
      <th>pB2</th>
      <td>50.0</td>
    </tr>
    <tr>
      <th>pB3</th>
      <td>50.0</td>
    </tr>
    <tr>
      <th>pC1</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>pC2</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>pC3</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>uA1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uA2</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uA3</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uB1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uB2</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uB3</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uC1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uC2</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uC3</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
d = {'a': 1, 'b': 2}
```


```python
for k,v in d.items():
    print(k, v)
```

    a [1]
    b [2]



```python
pd.DataFrame(d)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
t = [('a', 1), ('b', 2)]
```


```python
pd.DataFrame(t)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python
summary
```




    {'status': 'optimal',
     'optimal_value': 8500.0,
     'optimal_solution': {'pA1': 150.0,
      'pA2': 240.0,
      'pA3': 150.0,
      'alpha_A1': 0.0,
      'alpha_A2': 0.0,
      'alpha_A3': 0.0,
      'pB1': 50.0,
      'pB2': 50.0,
      'pB3': 50.0,
      'alpha_B1': 1.0,
      'alpha_B2': 0.0,
      'alpha_B3': 0.0,
      'pC1': 10.0,
      'pC2': 10.0,
      'pC3': 10.0,
      'alpha_C1': 1.0,
      'alpha_C2': 0.0,
      'alpha_C3': 0.0,
      'uA1': 1.0,
      'uA2': 1.0,
      'uA3': 1.0,
      'uB1': 1.0,
      'uB2': 1.0,
      'uB3': 1.0,
      'uC1': 1.0,
      'uC2': 1.0,
      'uC3': 1.0}}




```python
summary_MIP2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>optimal_value</th>
      <th>optimal_solution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_A1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_A2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_A3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_B1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha_B2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_B3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_C1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha_C2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>alpha_C3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pA1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>pA2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>pA3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>pB1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>pB2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>pB3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>pC1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>pC2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>pC3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>uA1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uA2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uA3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uB1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uB2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uB3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uC1</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uC2</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>uC3</th>
      <td>optimal</td>
      <td>8500.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_result_summary(MIP1)
```




    {'status': 'optimal',
     'optimal_value': 7700.0,
     'optimal_solution': {'pA1': 150.0,
      'pA2': 240.0,
      'pA3': 150.0,
      'pB1': 50.0,
      'pB2': 50.0,
      'pB3': 50.0,
      'pC1': 10.0,
      'pC2': 10.0,
      'pC3': 10.0,
      'uA1': 1.0,
      'uA2': 1.0,
      'uA3': 1.0,
      'uB1': 1.0,
      'uB2': 1.0,
      'uB3': 1.0,
      'uC1': 1.0,
      'uC2': 1.0,
      'uC3': 1.0}}




```python
150 * cA
```




    1500



# References

[1] https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/298917/UnitCommitment.pdf?sequence=3


```python

```
