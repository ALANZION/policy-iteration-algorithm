# POLICY ITERATION ALGORITHM

## AIM
The aim of this experiment is to implement the Policy Iteration Algorithm in Reinforcement Learning to determine the optimal policy and corresponding value function for a given environment. Policy Iteration combines iterative policy evaluation and policy improvement steps to achieve convergence towards an optimal policy.

## PROBLEM STATEMENT
In Reinforcement Learning, the agent interacts with an environment modeled as a Markov Decision Process (MDP).
The challenge is to find an optimal policy that maximizes the long-term cumulative reward.
Policy Iteration addresses this by:

Evaluating the value of a given policy (Policy Evaluation).
Improving the policy based on the evaluated value function (Policy Improvement).
Repeating these steps until the policy converges to the optimal policy.
## POLICY ITERATION ALGORITHM
The steps involved in the Policy Iteration Algorithm are:

Initialization

Initialize an arbitrary policy π and value function V(s).
Policy Evaluation

For the current policy π, compute the value function V(s) for all states until convergence.
Policy Improvement

Update the policy by choosing actions that maximize the expected return using the current value function.
Check for Convergence

If the policy does not change (π′ = π), then the policy is optimal and the algorithm terminates.
Otherwise, repeat steps 2 and 3.

## POLICY IMPROVEMENT FUNCTION
### Name :ALAN ZION H
### Register Number : 212223240004
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state,reward, done in P[s][a]:
          Q[s][a]+= prob*(reward+gamma*V[next_state]*(not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi

```
## POLICY ITERATION FUNCTION
### Name :ALAN ZION H
### Register Number : 212223240004
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```
## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="813" height="731" alt="image" src="https://github.com/user-attachments/assets/95c83419-0ef1-456f-8da3-f39d671c468f" />
<img width="560" height="182" alt="image" src="https://github.com/user-attachments/assets/2ccbcf36-c716-405f-82c5-5056757b7583" />

### 2. Policy, Value function and success rate for the Improved Policy
<img width="783" height="286" alt="image" src="https://github.com/user-attachments/assets/43ef3c46-7392-4f07-81c1-126b2ba64ee5" />

<img width="892" height="161" alt="image" src="https://github.com/user-attachments/assets/730b3799-201a-48ae-99f9-e665ce6f748e" />

<img width="650" height="308" alt="image" src="https://github.com/user-attachments/assets/8f040a0e-8b8a-48b3-b4ce-cbc6ee51b572" />

### 3. Policy, Value function and success rate after policy iteration
<img width="690" height="231" alt="image" src="https://github.com/user-attachments/assets/bbf90a97-1202-4b60-851e-d53d31809d5e" />

<img width="810" height="311" alt="image" src="https://github.com/user-attachments/assets/85878d90-e1a4-45bf-976f-bcf0bcaaa36f" />

<img width="900" height="132" alt="image" src="https://github.com/user-attachments/assets/e13ab2e5-ab7d-4482-a04b-8a46ed3f0ba1" />

<img width="729" height="254" alt="image" src="https://github.com/user-attachments/assets/2d2bdb1b-5931-4480-9b65-14fe6e2d3bd5" />



## RESULT:


