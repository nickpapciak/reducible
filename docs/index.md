---
layout: default
---

<h1 align="center">Same Network, Half the Size</h1>

## what is `reducible`❓
Reducible compresses the parameters of a neural network using a lossy compression technique called rank-k approximation. 


## our inspiration 🤩
Our idea came from when the team was thinking about how awesome JPEG is. It is able to take your images, throw out a ton of the data, and yet reconstruct the image in a way that makes it seem nearly identical to the human eye. For this hackathon, we thought about how we could use this idea to compress not just images, but compress neural networks. The sheer size of many deep learning models can be impractical for everyday use on devices with limited size and memory. However, being able to bring nearly accurate models but with half the space can be game changing for applications in mobile and edge computing, IoT devices, and the design of future models in general.


## how we built it 🛠️

We created a custom TensorFlow layer that acts similarly to a Dense layer but instead stores a rank-k approximation of the matrix.

## challenges 😰

Tensorflow had issues running on our Macs, so we had to use online compute power. We also had to make a custom file format for loading our compressed model because TensorFlow would not serialize our objects correctly. 

## results 💪

Our network has ~66% fewer parameters while only suffering from a ~0.45% decrease in accuracy!

## what we learned 🤓

We learned a great deal about the math behind SVD (singular value decomposition), as well as how to work with the TensorFlow backend API and make custom Layers. We also learned about object serialization and common space optimizations.

## what's next 🌎
Our current optimization algorithm assumes that a higher rank leads to a better approximation, which from our experimentation is mostly true, but isn't necessarily always true. This ends up being an integer programming problem, which is NP hard, but we want to explore possible efficient algorithms.

We also want to optimize our custom filetype with non-lossy serialization. Currently, our model has a predicted reduction of ~66%, while having an actual reduction of ~16%. This is because, although our compressed matrices have ~66% fewer parameters, we do not have the same serialization optimizations that TensorFlow includes that allows them to bring the filesize down so dramatically despite being unoptimized. 



<br>
<br>

# Deep In The Weeds: How does Reducible Work?
## what are we doing exactly?
For every Dense layer in a TensorFlow network, we replace it with a custom layer where the weights are approximated by a rank-$$k$$ matrix. 

### what is a low-rank approximation?
An $$m \times n$$ matrix can be decomposed into the product of two rank-k matrices. One has size $$m \times k$$, while the other has size $$k \times n$$. This converts a matrix which requires $$m\cdot n$$ space to a series of matrices which requires $$k\cdot(m+n)$$ space.


### why low-rank matrix approximations? 
1. Compression: a low-rank approximation provides a lossy compressed version of the matrix.
2. De-noising: if matrix $$A$$ is a noisy version of some original datapoints with 'good' dataset which is approximately low-rank, then conducting an Low-rank matrix approximation can potentially remove noise
3. Matrix completion: Low-rank approximations offfer a first-cut approach to the matrix completion problem 

### how can we compute a rank-$$k$$ approximation? 

Thankfully, the Eckart–Young–Mirsky theorem gives us an answer. The low-rank solution is given by the truncated SVD, or singular value decomposition, where we decompose a matrix $$A$$ into $${A = U \Sigma V^T}^{**}$$. The SVD (specifically PCA) provides a way to represent a matrix in a more compact form by ordering the data in order of the 'most significant' components. Then, we ... 

1. express $$A$$ in terms of its components ordered by their contribution to the model
2. keep only the $$k$$ most high contributing components. 

If you truncate the matrices $$U$$, $$\Sigma$$, and $$V$$ by keeping only the first $$k$$ singular values (columns of $$U$$, rows of $$V^T$$, and diagonal entries of $$\Sigma$$), you get an approximation of the matrix. 

Every matrix A has an SVD, and it is unique. The columns of $$U$$ and $$V$$ form orthonormal bases for the domain and codomain of $$A$$ and the singular values in $$\Sigma$$ represent the scaling factors along the coordinate axes.

** Where $$U$$ is a $$m\times n$$ orthogonal matrix, $$V$$ is a $$n\times n$$ orthogonal matrix, and $\Sigma$ is a $$m\times n$$ diagonal amtrix with non-negative entries

## challenges?
- Most neural networks are full-rank, so a rank-$$k$$ approximation is a balancing act. If we make our matrices too small, we lose too much information. Furthermore, we haven't implemented a learning algorithm so we cannot fine tune our layers. 
- What $$k$$ do we use for each layer? This turns into a mixed-integer nonlinear optimization minimization problem (MINLP) over the values of $$k$$ and the model accuracy, which is computationally costly to compute.


## credits



Sidebar image by catalyststuff <a href="https://www.freepik.com/free-vector/cute-cat-with-laptop-cartoon-vector-icon-illustration-animal-technology-icon-concept-isolated-premium-vector-flat-cartoon-style_18537593.htm#query=cute%20computer&position=7&from_view=search&track=ais&uuid=ad651732-f38b-4266-ae25-0602f214e1b0">on Freepik</a> 

Tab image by catalyststuff<a href="https://www.freepik.com/free-vector/cute-cat-hole-cartoon-vector-icon-illustration-animal-nature-icon-concept-isolated-premium-vector-flat-cartoon-style_23006709.htm#query=cat&position=18&from_view=author&uuid=e0bb35be-cd2a-4fb5-a1e7-f6e97ce0638b"> on Freepik</a>