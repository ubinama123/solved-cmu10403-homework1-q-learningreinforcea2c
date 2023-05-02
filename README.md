Download Link: https://assignmentchef.com/product/solved-cmu10403-homework1-q-learning_reinforce_a2c
<br>
<h1>Problem 1: DQN</h1>

In this problem you will implement Q-learning, using tabular and learned representations for the Q-function. This question will be graded out of 47 points, but you can earn up to 67 points by completing the extra credit problem (1.3c).

<h2>Problem 1.1: Relations among Q &amp; V &amp; C</h2>

The objective of this question is to understand different Bellman Optimality Equations, their strengths and limitations. Consider the Bellman Optimality Equation for the Value function,

If we continue expanding the value function <em>V </em>(<em>s</em><sub>2</sub>) using its own Bellman Optimality Equation, then we obtain a repeating structure:

There are a few more ways in which we can group this repeating sequence. First, we can capture the sequence starting at <em>R</em>(<em>s,a</em>) and ending at max, and observe that it too has a repeating substructure property:

We’ll call this repeating expression the state-value function <em>Q</em>(<em>s,a</em>) and use it to rewrite the Bellman Optimality equation as:

<em>Q</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub>) = <em>R</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub>) + <em>γ </em><sup>X</sup><em>T</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,s</em><sub>2</sub>)max<em>Q</em>(<em>s</em><sub>2</sub><em>,a</em><sub>2</sub>)<em>.</em>

<em>a</em>2

<em>s</em>2

Next, we can capture another pattern by grouping the expression beginning at <em>γ </em>and ending at <em>R</em>(<em>s,a</em>):

We’ll call this repeating expression the continuation function <em>C</em>(<em>s,a</em>), which can be written in terms of the value function:

<em>C</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub>) = <em>γ </em><sup>X</sup><em>T</em>(<em>s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,s</em><sub>2</sub>)<em>V </em>(<em>s</em><sub>2</sub>)<em>.</em>

<em>s</em>2

<ol>

 <li>(3 pts) Derive the recurrence relation (Bellman Optimality Equation) for <em>C</em>(<em>s,a</em>).</li>

 <li>(4 pts) Fill the following table to express the three functions in terms of each other.</li>

</ol>

<table width="559">

 <tbody>

  <tr>

   <td width="59"></td>

   <td width="224">V(s)</td>

   <td width="154">Q(s,a)</td>

   <td width="122">C(s,a)</td>

  </tr>

  <tr>

   <td width="59">V(s)</td>

   <td width="224">V(s) = V(s)</td>

   <td width="154">V(s) = max<em><sub>a </sub>Q</em>(<em>s,a</em>)</td>

   <td width="122">(<em>a</em>)</td>

  </tr>

  <tr>

   <td width="59">Q(s,a)</td>

   <td width="224">(<em>b</em>)</td>

   <td width="154">Q(s,a) = Q(s,a)</td>

   <td width="122">(<em>c</em>)</td>

  </tr>

  <tr>

   <td width="59">C(s,a)</td>

   <td width="224"><em>C</em>(<em>s,a</em>) = <em>γ </em><sup>P</sup><em><sub>s</sub></em>0 <em>T</em>(<em>s,a,s</em><sup>0</sup>)<em>V </em>(<em>s</em><sup>0</sup>)</td>

   <td width="154">(<em>d</em>)</td>

   <td width="122">C(s,a) = C(s,a)</td>

  </tr>

 </tbody>

</table>

Use the relation between the functions and your understanding of MDPs to answer the following True/False questions. Please include a 1-2 sentence explanation for each. Consider the scenario when we want to compute the optimal action without the knowledge of transition function <em>T</em>(<em>s,a,s</em><sup>0</sup>).

<ol start="3">

 <li>(2 pts) Can you derive the optimal policy given only <em>Q</em>(<em>s,a</em>)?</li>

 <li>(2 pts) Can you derive the optimal policy given only <em>V </em>(<em>s</em>) and <em>R</em>(<em>s,a</em>)?</li>

 <li>(2 pts) Can you derive the optimal policy given only <em>C</em>(<em>s,a</em>) and <em>R</em>(<em>s,a</em>)?</li>

</ol>

<h2>Problem 1.2: Temporal Difference &amp; Monte Carlo</h2>

Answer the true/false questions below, providing one or two sentences for <strong>explanation</strong>.

<ol>

 <li>(2 pts) TD methods can’t learn in an online manner since they require full trajectories.</li>

 <li>(2 pts) MC can be applied even with non-terminating episodes.</li>

</ol>

<h2>Problem 1.3: DQN Implementation</h2>

You will implement DQN and use it to solve two problems in OpenAI Gym: Cartpole-v0 and MountainCar-v0. While there are many (fantastic) implementations of DQN on Github, the goal of this question is for you to implement DQN from scratch <em>without </em>looking up code online.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> Please write your code in the DQN implementation.py. You are free to change/delete the template code if you want.

<strong>Code Submission</strong>: Your code should be reasonably well-commented in key places of your implementation. Make sure your code also has a README file.

<strong>How to measure if I ”solved” the environment? </strong>You should achieve the reward of 200 (Cartpole-v0) and around -110 or higher (MountainCar-v0) in consecutive 50 trials. <em>i.e. </em>evaluate your policy on 50 episodes.

<strong>Runtime Estimation</strong>: To help you better manage your schedule, we provide you with a reference runtime of DQN on a MacBook Pro 2018. For Cartpole-v0, it takes 5 minutes to first reach a reward of 200 and 68 minutes to finish 5000 episodes. For MountainCar-v0, it takes 40 ∼ 50 minutes to reach a reward around -110 and 200 minutes to finish 10000 episodes.

<ul>

 <li><strong> </strong>Implement a deep Q-network with experience replay. While the original DQN paper [5] uses a convolutional architecture, a neural network with 3 fully-connected layers should suffice for the low-dimensional environments that we are working with. For the deep Q-network, look at the QNetwork and DQN Agent classes in the code. You will have to implement the following:

  <ul>

   <li>Create an instance of the Q Network class.</li>

   <li>Create a function that constructs a greedy policy and an exploration policy (greedy) from the Q values predicted by the Q Network.</li>

   <li>Create a function to train the Q Network, by interacting with the environment.</li>

   <li>Create a function to test the Q Network’s performance on the environment.</li>

  </ul></li>

</ul>

For the replay buffer, you should use the experimental setup of [5] to the extent possible. Starting from the Replay Memory class, implement the following functions:

<ul>

 <li>Append a new transition from the memory.</li>

 <li>Sample a batch of transitions from the memory to train your network.</li>

 <li>Collect an initial number of transitions using a random policy.</li>

 <li>Modify your training function of your network to learn from experience sampled <em>from the memory</em>, rather than learning online from the agent.</li>

</ul>

Train your network on both the CartPole-v0 environment and the MountainCar-v0 environment (separately) until convergence, <em>i.e. </em>train a different network for each environment. We recommend that you periodically checkpoint your network to ensure no work is lost if your program crashes. Answer following questions in your report: (a) (17 pts) Describe your implementation, including the optimizer, the neural network architecture and any hyperparameters you used.

<ul>

 <li>For each environment, plot the average cumulative test reward throughout training.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> You are required to plot at least 2000 more episodes after you solve CartPole-v0, and at least 1000 more episodes after you solve MountainCar-v0. To do this, every 100 episodes, evaluate the current policy for 20 episodes and average the total reward achieved. Note that in this case we are interested in total reward without discounting or truncation.</li>

 <li> For each environment, plot the TD error throughout training. Does the TD error decrease when the reward increases? Suggest a reason why this may or may not be the case.</li>

 <li> We want you to generate a <em>video capture </em>of an episode played by your trained Q-network at different points of the training process (0<em>/</em>3, 1<em>/</em>3, 2<em>/</em>3, and 3<em>/</em>3 through the training process) of both environments. We provide you with a helper function to create the required video captures in test video().</li>

</ul>

<ul>

 <li>Implement any of the modifications below. Describe what you implemented, and run some experiments to determine if the modifications yield a better RL algorithm. You may implement multiple of the modifications, but you will not receive more than 20 points of extra credit.

  <ul>

   <li> Double DQN, as described in [9].</li>

   <li> Dueling DQN, as described in [10].</li>

   <li>Residual DQN, as described in [1].</li>

  </ul></li>

</ul>

<h1>Guidelines on References</h1>

We recommend you to read all the papers mentioned in the references. There is a significant overlap between different papers, so in reality you should only need certain sections to implement what we ask of you. We provide pointers for relevant sections for this assignment for your convenience.

The work in [4] contains the description of the experimental setup. Algorithm 1 describes the main algorithm. Section 3 (paragraph 3) describes the replay memory. Section 4 explains preprocessing (paragraph 1) and the model architecture (paragraph 3). Section 5 describes experimental details, including reward truncation, the optimization algorithm, the exploration schedule, and other hyperparameters). The methods section in [5], may clarify a few details so it may be worth to read selectively if questions remain after reading [4].

<h1>Guidelines on Hyperparameters</h1>

In this assignment you will implement improvements to the simple update Q-learning formula that make learning more stable and the trained model more performant. We briefly comment on the meaning of each hyperparameter and some reasonable values for them.

<ul>

 <li>Discount factor <em>γ</em>: 1<em>.</em>0 for MountainCar, and 0<em>.</em>99 for CartPole.</li>

 <li>Learning rate <em>α</em>: 0<em>.</em>001 for Cartpole and 0<em>.</em>0001 for Mountaincar.</li>

 <li>Exploration probability-greedy: While training, we suggest you start from a high epsilon value, and anneal this epsilon to a small value (0<em>.</em>05 or 0<em>.</em>1) during training. We have found decaying epsilon linearly from 0<em>.</em>5 to 0<em>.</em>05 over 100000 iterations works well. During test time, you may use a greedy policy, or an epsilon greedy policy with small epsilon (0<em>.</em>05).</li>

 <li>Number of training episodes: For MountainCar-v0, you should see improvements within 2000 (or even 1000) episodes. For CartPole-v0, you should see improvements starting around 2000 episodes.</li>

</ul>

Look at the average reward achieved in the last few episodes to test if performance has plateaued; it is usually a good idea to consider reducing the learning rate or the exploration probability if performance plateaus.

<ul>

 <li>Replay buffer size: 50000; this hyperparameter is used only for experience replay. It determines how many of the last transitions experienced you will keep in the replay buffer before you start rewriting this experience with more recent transitions.</li>

 <li>Batch size: 32; typically, rather doing the update as in (2), we use a small batch of sampled experiences from the replay buffer; this provides better hardware utilization. In addition to the hyperparameters:</li>

 <li>Optimizer: You may want to use Adam as the optimizer. Think of Adam like a fancier SGD with momentum, it will automatically adjust the learning rate based on the statistics of the gradients its observing.</li>

 <li>Loss function: you can use Mean Squared Error.</li>

</ul>

The implementations of the methods in this homework have multiple hyperparameters. These hyperparameters (and others) are part of the experimental setup described in [4, 5]. For the most part, we strongly suggest you to follow the experimental setup described in each of the papers. [4, 5] was published first; your choice of hyperparameters and the experimental setup should follow closely their setup. We recommend you to read all these papers. We have given pointers for the most relevant portions for you to read in a previous section.

<h1>REINFORCE/A2C Installation instructions (Linux)</h1>

In the next part, you will implement 2 policy gradient algorithms and evaluate them on the OpenAI Gym LunarLander-v2 environment. This environment is considered solved if the agent can achieve an average score of at least 200. We’ve provided Python packages that you may need in requirements.txt. To install these packages using pip and virtualenv, run the following commands: apt-get install swig virtualenv env source env/bin/activate pip install -U -r requirements.txt

If your installation is successful, then you should be able to run the provided template code:

python reinforce.py python a2c.py

Note: You will need to install swig and box2d in order to install gym[box2d], which contains the LunarLander-v2 environment. You can install box2d by running pip install git+https://github.com/pybox2d/pybox2d

If you simply do pip install box2d, you can sometimes get an error because the pip package for box2d depends on an older version of swig.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> For additional installation instructions, see <a href="https://github.com/openai/gym">https://github.com/openai/gym</a><a href="https://github.com/openai/gym">.</a>

<h1>Problem 2: REINFORCE</h1>

In this section, you will implement episodic REINFORCE [11], a policy-gradient learning algorithm. Please write your code in reinforce.py; the template code provided inside is there to give you an idea on how you can structure your code, but is not mandatory to use.

Policy gradient methods directly optimize the policy <em>π</em>(<em>A </em>| <em>S,θ</em>), which is parameterized by <em>θ</em>. The REINFORCE algorithm proceeds as follows. We generate an episode by following policy <em>π</em>. After each episode ends, for each time step <em>t </em>during that episode, we update the policy parameters <em>θ </em>with the REINFORCE update. This update is proportional to the product of the return <em>G<sub>t </sub></em>experienced from time step <em>t </em>until the end of the episode and the gradient of log<em>π</em>(<em>A<sub>t </sub></em>| <em>S<sub>t</sub>,θ</em>). See Algorithm 1 for details.

<strong>Algorithm 1 </strong>REINFORCE

1: <strong>procedure </strong>REINFORCE

2:          <em>Start with policy model π<sub>θ </sub></em>3:  <strong>repeat:</strong>

4:                                   <em>Generate an episode S</em><sub>0</sub><em>,A</em><sub>0</sub><em>,r</em><sub>0</sub><em>,…,S<sub>T</sub></em><sub>−1</sub><em>,A<sub>T</sub></em><sub>−1</sub><em>,r<sub>T</sub></em><sub>−1 </sub><em>following π<sub>θ</sub></em>(·)

5:                          <strong>for </strong><em>t from T </em>− 1 <em>to </em>0:

<em>T</em>

6:

7:

8:                          <em>Optimize π<sub>θ </sub>using </em>∇<em>L</em>(<em>θ</em>)

9: <strong>end procedure</strong>

For the policy model <em>π</em>(<em>A </em>| <em>S,θ</em>), we recommend starting with a model that has:

<ul>

 <li>three fully connected layers with 16 units each, each followed by ReLU activations</li>

 <li>another fully connected layer with 4 units (the number of actions)</li>

 <li>a softmax activation (so the output is a proper distribution)</li>

</ul>

Initialize bias for each layer to zero. We recommend using a variance scaling kernel initializer

that draws samples from a uniform distribution over [−<em>α,α</em>] for <em>α </em>= <sup>p</sup>(3 ∗ scale<em>/n</em>) where scale = 1<em>.</em>0 and <em>n </em>is the average of the input and output units. HINT: Read the Keras documentation.

You can use the model.summary() and model.get config() calls to inspect the model architecture.

You can choose which optimizer and hyperparameters to use, so long as they work for learning on LunarLander-v2. We recommend using Adam as the optimizer. It will automatically adjust the learning rate based on the statistics of the gradients it’s observing. You can think of it like a fancier SGD with momentum. Keras provides a version of Adam <a href="https://keras.io/optimizers/">https:</a>

<a href="https://keras.io/optimizers/">//keras.io/optimizers/</a><a href="https://keras.io/optimizers/">.</a>

Train your implementation on the LunarLander-v2 environment until convergence<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>. Be sure to keep training your policy for at least 1000 more episodes after it reaches 200 reward so that you are sure it consistently achieves 200 reward and so that this convergence is reflected in your graphs. Then, answer the following questions.

<ol>

 <li>Describe your implementation, including the optimizer and any hyperparameters you used (learning rate, <em>γ</em>, etc.). Your description should be detailed enough that someone could reproduce your results.</li>

 <li>Plot the learning curve: Every <em>k </em>episodes, freeze the current cloned policy and run 100 test episodes, recording the mean and standard deviation of the cumulative reward. Plot the mean cumulative reward on the y-axis with the standard deviation as error-bars against the number of training episodes on the x-axis. Write a paragraph or two describing your graph(s) and the learning behavior you observed. Be sure to address the following questions:

  <ul>

   <li>What trends did you see in training?</li>

   <li>How does the final policy perform?</li>

   <li>The REINFORCE algorithm may be unstable. If you observe such instability in your implementation, what could be the reason?</li>

  </ul></li>

</ol>

Hint: You can use matplotlib’s plt.errorbar() function. <a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html">https://matplotlib.org/ </a><a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html">api/_as_gen/matplotlib.pyplot.errorbar.html</a>

<h1>Problem 3: Advantage-Actor Critic</h1>

In this section, you will implement N-step Advantage Actor Critic (A2C) [2]. Please write your code in a2c.py; the template code provided inside is there to give you an idea on how you can structure your code, but is not mandatory to use.

<strong>Algorithm 2 </strong>N-step Advantage Actor-Critic

<table width="624">

 <tbody>

  <tr>

   <td colspan="2" width="624">1: <strong>procedure </strong>N-Step Advantage Actor-Critic2:               <em>Start with policy model π<sub>θ </sub>and value model V<sub>ω</sub></em></td>

  </tr>

  <tr>

   <td width="50">3:</td>

   <td width="574"><strong>repeat:</strong></td>

  </tr>

  <tr>

   <td width="50">4:</td>

   <td width="574"><em>Generate an episode S</em><sub>0</sub><em>,A</em><sub>0</sub><em>,r</em><sub>0</sub><em>,…,S<sub>T</sub></em><sub>−1</sub><em>,A<sub>T</sub></em><sub>−1</sub><em>,r<sub>T</sub></em><sub>−1 </sub><em>following π<sub>θ</sub></em>(·)</td>

  </tr>

  <tr>

   <td width="50">5:</td>

   <td width="574"><strong>for </strong><em>t from T </em>− 1 <em>to </em>0:</td>

  </tr>

  <tr>

   <td width="50">6:</td>

   <td width="574"><em>V<sub>end </sub></em>= 0 if (<em>t </em>+ <em>N </em>≥ <em>T</em>) <em>else V<sub>ω</sub></em>(<em>s<sub>t</sub></em><sub>+<em>N</em></sub>)</td>

  </tr>

  <tr>

   <td width="50">7:</td>

   <td width="574"><em> else </em>0)</td>

  </tr>

  <tr>

   <td width="50">8:9: 10:</td>

   <td width="574"><em>Optimize π<sub>θ </sub>using </em>∇<em>L</em>(<em>θ</em>)</td>

  </tr>

  <tr>

   <td colspan="2" width="624">11:                       <em>Optimize V<sub>ω </sub>using </em>∇<em>L</em>(<em>ω</em>)12: <strong>end procedure</strong></td>

  </tr>

 </tbody>

</table>

N-step A2C provides a balance between bootstraping using the value function and using the full Monte-Carlo return, using an N-step trace as the learning signal. See Algorithm 2 for details. N-step A2C includes both REINFORCE with baseline (<em>N </em>= ∞) and the 1-step A2C covered in lecture (<em>N </em>= 1) as special cases and is therefore a more general algorithm.

The critic updates the state-value parameters <em>ω</em>, and the actor updates the policy parameters <em>θ </em>in the direction suggested by the N-step trace.

Start off with the same policy architecture described in Problem 1 for both the actor and the critic. Play around with the network architecture of the critic’s state-value approximator to find one that works for LunarLander-v2. Once again, you can choose which optimizer and hyperparameters to use, so long as they work for learning on LunarLander-v2.

Answer the following questions:

<ol>

 <li><strong>[</strong>10 pts] Describe your implementation, including the optimizer, the critic’s network architecture, and any hyperparameters you used (learning rate, <em>γ</em>, etc.).</li>

 <li><strong>[</strong>20 pts] Train your implementation on the LunarLander-v2 environment several times with N varying as [1, 20, 50, 100] (it’s alright if the N=1 case is hard to get working). Plot the learning curves for each setting of N in the same fashion as Problem 1. You may find that your plots with error bars will be too busy to plot all values of N on the same graph. If this is the case, make a different plot for each value of N. Once again, write a paragraph or two describing your graph(s) and the learning behavior you observed. Be sure to address the following questions:

  <ul>

   <li>What trends did you observe in training?</li>

   <li>How does the final policy perform?</li>

   <li>If you found A2C to be unstable or otherwise difficult to train, why might this be the case? What about the algorithm formulation could cause training instability, and what improvements might be made to improve it?</li>

  </ul></li>

 <li><strong>[</strong>10 pts] Discuss how the performance of your implementation of A2C compares with REINFORCE and how A2C’s performance varies with N. Which algorithm and N setting learns faster, and why do you think this is the case?</li>

</ol>

<h1>Extra credit</h1>

A major bottleneck in training policy gradient algorithms is that only one episode (or batch of states) is generated at a time. However, once the policy has been updated once, the training data is no longer drawn from the current policy distribution, becoming “invalid” in a sense. A similar challenge occurs when parallelizing training, since once a parameter update is performed by one worker, the policy distribution changes and invalidates the data gathered and gradients computed by the other workers. Mnih <em>et al. </em>argue that the exploration noise from asynchronous policy updates can be beneficial to learning [3].

First, let’s introduce a more complex environment. Many deep reinforcement learning papers (at least in the past few years) have used Atari games as performance benchmarks due to their greater complexity. Apply your implementation of A2C to any of the OpenAI gym Breakout environments. We recommend either Breakout-v0 or BreakoutNoFrameskip-v4 environments. You will need to use a larger, more complex policy network than the one you used in Problem 1 and 2, as well as some tricks like learning rate decay. Think carefully about your hyperparameters, particularly <em>N</em>. You should be able to reach at least 200 average reward after 10-20 hours of training on AWS; note that running these large networks on a laptop may take up to two weeks.

Then, implement multi-threaded synchronous Advantage Actor-Critic by gathering episode rollouts in parallel and performing a single gradient update. What speedup can you achieve? How might you measure this? Then, implement Asynchronous Advantage Actor-Critic (A3C) with multiple threads, using your multi-threaded synchronous Advantage Actor-Critic as a starting point. Do you see a learning speedup or increased stability compared to a synchronous implementation?

Up to 15 points extra credit will be awarded total, contingent on implementation, results, and analysis. Describe how you implemented the task and provide metrics and graphs showing improvement as well as explanations as to why that might be the case. You may also wish to include links to videos of your trained policies. If nothing else, it is entertaining and rewarding to see an agent you trained play Breakout at a superhuman level.

<h1>Guidelines on implementation</h1>

This homework requires a significant implementation effort. It is hard to read through the papers once and know immediately what you will need to be implement. We suggest you to think about the different components (e.g., model definition, model updater, model runner, …) that you will need to implement for each of the different methods that we ask you about, and then read through the papers having these components in mind. By this we mean that you should try to divide and implement small components with well-defined functionalities rather than try to implement everything at once. Much of the code and experimental setup is shared between the different methods so identifying well-defined reusable components will save you trouble.

Some hyperparameter and implementation tips and tricks:

<ul>

 <li>For efficiency, you should try to vectorize your code as much as possible and use <strong>as few loops as you can </strong>in your code. In particular, in lines 5 and 6 of Algorithm 1 (REINFORCE) and lines 5 to 7 of Algorithm 2 (A2C) you should not use two nested loops. How can you formulate a single loop to calculate the cumulative discounted rewards? Hint: Think backwards!</li>

 <li>Moreover, it is likely that it will take between 10K and 50K episodes for your model to converge, though you should see improvements within 5K episodes (about 30 minutes to one hour). On a NVIDIA GeForce GTX 1080 Ti GPU, it takes about five hours to run 50K training episodes with our REINFORCE implementation.</li>

 <li>For A2C, downscale the rewards by a factor of 1e-2 (i.e., divide by 100) when training (but not when plotting the learning curve) This will help with the optimization since the initial weights of the critic are far away from being able to predict a large range such as [−200<em>,</em>200]. You are welcome to try downscaling the rewards of REINFORCE as well.</li>

 <li>Normalizing the returns <em>G<sub>t </sub></em>over each episode by subtracting the mean and dividing by the standard deviation may improve the performance of REINFORCE.</li>

 <li>Likewise, batch normalization between layers can improve stability and convergence rate of both REINFORCE and A2C. Keras has a built-in batch normalization layer <a href="https://keras.io/layers/normalization/">https://keras.io/layers/normalization/</a><a href="https://keras.io/layers/normalization/">.</a></li>

 <li>Feel free to experiment with different policy architectures. Increasing the number of hidden units in earlier layers may improve performance.</li>

 <li>We recommend using a discount factor of <em>γ </em>= 0<em>.</em></li>

 <li>Try out different learning rates. A good place to start is in the range [1e-5<em>,</em>1e-3]. Also, you may find that varying the actor and critic learning rates for A2C can help performance. There is no reason that the actor and critic must have the same learning rate.</li>

 <li>Policy gradient algorithms can be fairly noisy. You may have to run your code for several tens of thousand training episodes to see a consistent improvement for REINFORCE and A2C.</li>

 <li>Instead of training one episode at a time, you can try generating a fixed number of steps in the environment, possibly encompassing several episodes, and training on such a batch instead.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> After this assignment, we highly recommend that you look at DQN implementations on Github to see how others have structured their code.

<a href="#_ftnref2" name="_ftn2">[2]</a> You can use the Monitor wrapper to generate both the performance curves and the video captures.

<a href="#_ftnref3" name="_ftn3">[3]</a> <a href="https://github.com/openai/gym/issues/100">https://github.com/openai/gym/issues/100</a>

<a href="#_ftnref4" name="_ftn4">[4]</a> LunarLander-v2 is considered solved if your implementation can attain an average score of at least 200.