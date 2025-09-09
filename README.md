# POLICY ITERATION ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## POLICY ITERATION ALGORITHM
Include the steps involved in policy iteration algorithm
</br>
</br>

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

### 2. Policy, Value function and success rate for the Improved Policy
<img width="546" height="168" alt="image" src="https://github.com/user-attachments/assets/ce5a1594-97f5-4b77-ab1f-4b266b083116" />

### 3. Policy, Value function and success rate after policy iteration
<img width="587" height="197" alt="image" src="https://github.com/user-attachments/assets/65fffeae-6ceb-4082-81d0-86ed312fbef6" />

<img width="704" height="43" alt="image" src="https://github.com/user-attachments/assets/69aa4b26-f49e-4321-b1c3-47eca001efa9" />

<img width="640" height="169" alt="image" src="https://github.com/user-attachments/assets/d71bd737-fc36-4e3b-830b-e64a32281391" />


## RESULT:


