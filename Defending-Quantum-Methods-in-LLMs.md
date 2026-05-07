# **Defending the Use of Quantum-Inspired Methods in LLM Analysis**

## **Introduction: The Mathematical Rationale**

The application of quantum theory to Large Language Models (LLMs) is not an assertion that LLMs exhibit quantum physical phenomena. Rather, it is the application of **Quantum Probability Theory** and **Quantum Information Theory (QIT)** as a mathematical framework.  
Classical probability theory (based on Kolmogorov axioms) assumes that events are subsets of a universal sample space and that probability is a simple additive measure. This works well for discrete, mutually exclusive events (e.g., token prediction at the character level). However, classical probability struggles to naturally model systems characterized by deep contextual dependence, ambiguity, and continuous superposition of states—the exact features that define LLM semantic spaces.  
Quantum probability was explicitly developed to model systems where variables cannot be evaluated independently of their context. By mapping LLM embeddings into a Hilbert space and representing output distributions as density matrices, we gain access to a generalized probability framework that naturally encodes semantic overlap, contextual entanglement, and uncertainty.

## **1\. Core Correspondences: LLMs and Quantum Systems**

The architecture and behavior of LLMs map elegantly to core concepts in quantum mechanics, providing a rigorous justification for using quantum mathematics to analyze them.

### **A. Superposition and Semantic Ambiguity**

In quantum mechanics, a system exists in a superposition of multiple possible states until measured. In an LLM, a token or sequence before the final softmax layer exists as a continuous vector in a high-dimensional space.  
**The Correspondence:** Just as a quantum state is a linear combination of basis states, an LLM embedding is a linear combination of latent semantic features. A word like "bank" exists in a superposition of "financial institution" and "river side" until the surrounding context forces it into a specific semantic state.  
**Why Classical Fails:** Classical logic requires "bank" to be exclusively A or B. Quantum logic allows it to be a weighted combination of both, perfectly mirroring dense vector representations.

### **B. Entanglement and Self-Attention Mechanisms**

Quantum entanglement describes a scenario where the state of one particle cannot be described independently of the state of another.  
**The Correspondence:** The core innovation of the Transformer architecture is the self-attention mechanism. In a deep LLM, the embedding for a given token is computed by taking a weighted sum of all other tokens in the sequence. By the final layers, the representation of a single word is inextricably "entangled" with the entire prompt. You cannot measure the semantic state of one token without referencing the whole.

### **C. Wavefunction Collapse and Decoding**

A quantum system evolves deterministically and continuously but yields probabilistic, discrete outcomes upon measurement.  
**The Correspondence:** The forward pass of an LLM is a continuous, deterministic matrix multiplication resulting in a probability amplitude distribution (the logits). The moment we apply a decoding algorithm (e.g., nucleus sampling, temperature scaling) to generate the next token, the continuous distribution "collapses" into a single discrete observation.

## **2\. The Methodological Necessity of Density Matrices**

Your specific research goal—to interpret differences between LLM distributions (e.g., fp32 vs int8 quantization)—is where quantum metrics prove their unique value.  
When comparing LLM outputs, classical tests like the Maximum Mean Discrepancy (MMD) or Kolmogorov-Smirnov (K-S) test are highly sensitive to *surface-level* differences. If an fp32 model outputs "The quick brown fox" and an int8 model outputs "A fast brown fox," classical token-level tests immediately flag these as divergent distributions.  
**However, they mean the same thing.** To capture this, we need a mathematical object that encodes not just the probability of individual outcomes, but the *correlations and semantic overlaps* between them.

### **Density Matrices vs. Probability Vectors**

A classical probability distribution is a simple vector. It assumes the items being measured are mutually exclusive. A quantum density matrix is a positive semi-definite matrix with a trace of 1\.

* **Diagonal elements** represent the classical probabilities of individual states.  
* **Off-diagonal elements (coherences)** represent the degree of semantic overlap or similarity between different states in the Hilbert space.

By converting a set of LLM completions into a Gram matrix and normalizing it into a density matrix, you are constructing a mathematical object that natively understands that two different token sequences might be semantically identical.

## **3\. Addressing the "Baggage" Critique: Why Purely Classical Approaches Fall Short**

A common critique of this methodology is that importing concepts from quantum mechanics introduces unnecessary conceptual "baggage," and that classical statistics should be sufficient to evaluate LLM outputs. However, this critique confuses *physics* with *linear algebra*.  
Applying a purely classical probability framework to dense, contextual embedding spaces actually forces researchers to adopt far more mathematical "baggage" in the form of workarounds and stitched-together metrics.

### **The Limits of Kolmogorov Probability**

Classical probability theory is based on set theory. It assumes that events in a sample space are distinct, mutually exclusive entities. If we attempt to analyze LLM outputs through a purely classical lens, we run into immediate structural failures:

1. **The Inability to Handle Semantic Overlap:** In classical probability, if an LLM generates Sentence A ("The dog ran") and Sentence B ("The hound sprinted"), these are treated as entirely separate, disjoint events in the probability space. To measure their semantic similarity classically, one must introduce an entirely separate mathematical step (e.g., cosine similarity of embeddings) *after* the probability distribution is calculated.  
2. **The Curse of Contextual Dimensionality:** To capture deep context classically, models require exponentially exploding joint probability distributions or massive Markov chain state spaces, which quickly become mathematically intractable.

### **The Elegance of the Quantum Framework**

The quantum approach possesses no unnecessary baggage; it is, in fact, the most elegant and parsimonious way to model LLM behavior.  
Instead of juggling two separate mathematical objects—a classical probability vector for likelihoods and a separate distance matrix for semantic similarities—the **density matrix** consolidates both into a single, unified object.

* It natively encodes classical uncertainty (the stochastic likelihood that the LLM will generate a specific sequence).  
* It natively encodes non-classical contextual overlap (the "coherences" or off-diagonal elements that mathematically prove two distinct sequences mean the same thing).

In short, retreating to purely classical methods would force us to use crude, mutually exclusive categories for language, abandoning the continuous, overlapping geometry that makes LLMs powerful in the first place.

## **4\. Justifying Quantum Metrics**

Once the distribution is represented as a density matrix, QIT provides metrics that classical statistics cannot offer:

### **Von Neumann Entropy**

Classical Shannon entropy measures the unpredictability of specific tokens. Von Neumann entropy measures the *diversity of the semantic space itself* by analyzing the eigenvalues of the density matrix.  
**How it works:** While Shannon entropy only looks at the diagonal elements (the raw probability of each discrete completion), Von Neumann entropy explicitly accounts for the off-diagonal "coherences" (the semantic overlap between different completions). The metric effectively asks: *"How many orthogonal (truly unique) semantic dimensions are being used by this distribution?"*  
**Interpreting the Values:**  
The value of Von Neumann entropy scales with the semantic spread of the LLM's outputs, ranging from ![][image1] to ![][image2] (where ![][image3] is the number of sampled dimensions).

* **Near 0 (A "Pure State"):** The LLM is outputting mathematically identical semantic content. Even if the surface tokens vary slightly, the embedding vectors point in the exact same direction. The system is highly focused and predictable in meaning.  
* **Low to Moderate:** The LLM is generating responses with a consistent core theme but minor semantic variations (e.g., slight shifts in tone, phrasing, or emphasis). There is high overlap (coherence) between the generated samples.  
* **High / Maximum (A "Maximally Mixed State"):** The LLM is outputting wildly diverse, semantically orthogonal responses. There is very little conceptual overlap between the completions. This might occur in highly open-ended creative prompts or if a model is heavily hallucinating.

**The Conceptual Payoff:** If an LLM generates 100 completely different sentences that all mean the exact same thing, classical Shannon entropy scores high (because the individual tokens are unpredictable), but Von Neumann entropy correctly scores near zero (because the underlying meaning is unchanged). This allows you to rigorously prove whether an architectural change (e.g., fp32 vs int8 quantization) degrades semantic diversity or merely alters specific token selection.

### **Trace Distance**

Trace distance calculates the absolute geometric distance between two semantic spaces.  
**How it works:** Trace distance subtracts one density matrix from another, computes the absolute value of the resulting matrix (via its eigenvalues), and halves the sum. It mathematically represents the maximum probability that an observer could successfully distinguish an output from Model A from an output from Model B based purely on their semantic embedding spaces.  
**Interpreting the Values:**  
The value is strictly bounded between 0 and 1\.

* **0 (Indistinguishable):** The two models output the exact same distribution of meanings.  
* **Intermediate Values (e.g., 0.3):** There is a partial semantic shift. Some concepts overlap, while others are unique to a specific model.  
* **1 (Perfectly Distinguishable/Orthogonal):** The semantic spaces have zero overlap. The models are outputting completely unrelated concepts.

**The Conceptual Payoff:** This allows researchers to answer the fundamental question: *"Are these two models fundamentally outputting different meanings, or just using different words?"* It provides a bounded, absolute magnitude of semantic shift between two implementations, such as checking if quantizing a model breaks its conceptual understanding.

### **Quantum Relative Entropy (QRE)**

Quantum Relative Entropy is the quantum analogue to Kullback-Leibler (KL) divergence, measuring the *asymmetric* information shift between two models.  
**The Mathematics of Quantum Surprisal:**  
To understand exactly what QRE is measuring, it is helpful to express it as an expected "surprisal." In classical information theory, the surprisal of an event with probability ![][image4] is ![][image5]. This measures how much information you gain—or how "surprised" you are—when that event occurs. Rare events yield a high surprisal, while guaranteed events yield zero surprisal.  
In our quantum framework, the surprisal of a semantic state is represented by the matrix observable ![][image6]. This expression is the linear algebraic generalization of classical surprisal, often called the **quantum surprisal operator** or **matrix surprisal**. Because the density matrix ![][image7] represents the entire probability distribution of the LLM's semantic states—including the overlaps and coherences between different outputs—![][image6] acts as an observable that encodes the surprise associated with every possible semantic dimension simultaneously.  
Here is how this matrix observable functions:

* **Eigenvalue Interpretation:** The matrix ![][image6] is computed by taking the matrix logarithm of the density matrix. Its eigenvalues are ![][image8], where ![][image9] are the eigenvalues of ![][image7]. In LLM terms, this computes the classical surprisal of the *principal, orthogonal semantic themes* (the eigenstates) of the model's outputs.  
* **Connection to Entropy:** Just as the weighted average of classical surprisals yields Shannon Entropy, the expected value of the quantum surprisal observable yields **Von Neumann Entropy**. Mathematically, taking the trace of ![][image7] multiplied by its surprisal (![][image10]) gives the exact Von Neumann Entropy formula (![][image11]) discussed earlier.

We can use this observable to mathematically decompose the QRE formula to reveal its direct relationship to surprisal:  
This translates perfectly to: **QRE \= (Quantum Cross-Entropy) \- (Von Neumann Entropy)**.

* **Term 1 (Quantum Cross-Entropy):** The expected value of Model B's surprisal (![][image12]) using Model A's distribution (![][image13]). This is the expected surprisal if you assume the LLM follows Model B's semantic distribution, but you are actually given Model A's distribution.  
* **Term 2 (Von Neumann Entropy):** The baseline expected surprisal (the inherent semantic diversity) of Model A on its own.

Therefore, QRE is simply the *excess expected surprisal*—the exact mathematical penalty (in bits or nats) of using the wrong semantic model to predict a distribution of meanings.  
**Conceptualizing Semantic Surprisal vs. Token Surprisal:** To help bridge the gap between classical information theory and this quantum-inspired framework, it is crucial to contrast **token surprisal** with **semantic surprisal**.  
In classical NLP, surprisal is measured in bits based on a discrete vocabulary. If an LLM is asked, *"What is the capital of France?"* and it outputs "T" (for Tokyo), the token surprisal is massive. It asks: *"How unexpected was that exact sequence of characters?"*  
**Semantic surprisal**, quantified by QRE and the ![][image6] observable, asks instead: *"How unexpected was that underlying concept, regardless of the words used to express it?"*

* **The "Paraphrase" Example:** Imagine prompting an LLM to explain a black hole.  
  * **Response A:** *"A region of spacetime where gravity is so strong that nothing can escape."*  
  * **Response B:** *"A cosmic sinkhole where the gravitational pull traps even light."*  
  * **Response C:** *"A delicious chocolate cake recipe."*

If we use classical token surprisal (like KL divergence) to compare Response A and Response B, the surprisal is quite high—the models chose completely different words, syntax, and string lengths. However, in our quantum-inspired framework, the density matrix ![][image7] creates a geometric map of expected meanings. Because A and B point in almost the exact same direction in the embedding space, they fall into the same "eigenstate" of the density matrix. Therefore, the **semantic surprisal of B given A is near zero**. Response C, however, points to an entirely orthogonal dimension in the semantic space. Calculating ![][image6] for Response C yields a massive semantic surprisal.

* **Interpreting the "Bits":** When we measure classical surprisal in bits, we quantify text string compression. When we measure semantic surprisal, we quantify **conceptual deviation**. A low semantic surprisal (few bits) means the model stayed within the expected themes—it might have used surprising words, but not surprising meanings. A high semantic surprisal means the model jumped into an orthogonal semantic dimension (e.g., hallucinating or changing the subject completely).

**The Conceptual Payoff:** Because QRE is asymmetric (![][image14]) and operates natively on continuous concepts, it serves as a powerful diagnostic tool. It can determine if an int8 quantized model has suffered "mode collapse"—losing the rich, diverse semantic nuances present in the original fp32 base model without explicitly introducing new, hallucinatory concepts.

## **5\. Precedent: Quantum Cognition and Information Retrieval**

It is highly recommended to remind the committee that you are not the first to do this. There are established subfields that use quantum mathematics for semantic processing:

* **Quantum Information Retrieval (QIR):** Researchers have long used Hilbert spaces and density operators to model document relevance and user queries, specifically because classical Boolean logic fails to capture semantic overlap.  
* **Quantum Cognition:** A rich field in psychology uses quantum probability to model human judgments that violate classical probability rules, mapping perfectly to how LLMs process contextual language.

## **Conclusion for the Committee**

Using quantum theory to analyze LLMs is not a metaphorical overreach; it is a mathematical upgrade. Classical probability forces us to treat language generation as a series of discrete, mutually exclusive events, which obscures the deep semantic invariants preserved by these models. By treating LLM semantic spaces as Hilbert spaces and their output distributions as density matrices, we adopt a mathematical language explicitly designed for contextuality, superposition, and complex correlations.

## **Appendix A: Detailed Mapping of Quantum and LLM Concepts**

To further clarify the methodological alignment, the following table and expanded notes directly map the structural features of Quantum Information Theory (QIT) to those of Large Language Models.

| Quantum Concept | LLM/NLP Equivalent | Mathematical Function / Purpose |
| :---- | :---- | :---- |
| **Quantum Particle** | **Token** | The fundamental discrete unit of the system whose state is probabilistic and entangled with others until measured. |
| **Hilbert Space** | **Embedding Space** | A high-dimensional, continuous vector space where geometric distance (inner product) equals conceptual similarity. |
| **Quantum State** | **Continuous Vector / Embedding** | The latent representation of a token or sequence before discrete realization. |
| **Superposition** | **Semantic Ambiguity** | Encoding multiple potential meanings simultaneously as a linear combination of basis concepts. |
| **Entanglement** | **Self-Attention Mechanism** | Contextual interdependence; the state of one unit is defined by its non-separable relationship with all other units. |
| **Measurement / Collapse** | **Decoding / Token Generation** | The irreversible projection of a continuous probability amplitude into a single, discrete, stochastic observation. |
| **Density Matrix** | **Distribution of Completions** | A unified representation of an ensemble of outputs that captures both probability and semantic overlap. |
| **Trace** | **Normalization** | Ensuring that the total probability across all semantic dimensions sums to 1\. |
| **Observables** | **Prompts or Probes** | The mechanism of "asking a question" to the system, forcing the superposition to resolve into a specific eigenstate. |

### **Expanded Analogues:**

* **Hilbert Space vs. Embedding Space:** Just as quantum mechanics replaced classical phase space with Hilbert space to handle continuous probability amplitudes, modern NLP replaced discrete, one-hot vocabulary vectors with dense, continuous embedding spaces.  
* **Measurement vs. Decoding:** In physics, the Schrödinger equation governs deterministic evolution, but measurement yields a random result. In LLMs, the forward pass is entirely deterministic, computing continuous probability distributions. Applying a decoding strategy introduces the probabilistic "collapse" into a discrete string.  
* **The Token as a Quantum Particle:** To summarize how these correspondences interact, consider the journey of a single LLM token:  
  1. **As a particle in a system (Entanglement):** The self-attention mechanism treats each token as a "particle," entangling its semantic state with all the other tokens in the prompt.  
  2. **Before measurement (The Quantum State):** Before a final word is chosen, this token exists as a continuous vector in a high-dimensional space, holding a superposition of possible meanings.  
  3. **Upon measurement (Wavefunction Collapse):** When the decoding algorithm runs, this token collapses from a wave-like mathematical probability into a discrete, observable word printed on the screen.

## **Appendix B: Mathematical Deep Dive: From Embeddings to Density Matrices**

When presenting this methodology, it is crucial to explain exactly how we transition from standard NLP embeddings to a valid quantum state. This process requires four distinct mathematical steps.

### **Phase 1: The Raw Embeddings**

We begin by generating a set of *N* completions from the LLM. Each completion is passed through an embedding model (like MPNet or EmbeddingGemma) to yield a dense vector in a high-dimensional space (e.g., 768 dimensions).

### **Phase 2: Normalization to the Unit Sphere**

Before constructing our matrix, every raw vector must be divided by its own length (its magnitude) to normalize it.  
**Why this matters:** In NLP, the magnitude (length) of a vector is often an artifact of string length or token frequency—factors we want to ignore. By mapping all vectors to the surface of a unit sphere, we isolate their *direction*. In this space, the relationship between two vectors is purely semantic, allowing the inner product to act as a perfect measure of cosine similarity.

### **Phase 3: The Gram Matrix**

Instead of calculating a standard covariance matrix (which looks at how *features* relate to other *features*), we calculate a Gram matrix, which looks at how *samples* relate to other *samples*.  
**Why this matters:** This creates an *N* by *N* grid. The diagonal of this grid compares every completion to itself (which always equals 1). The off-diagonals contain the cosine similarities of every completion against every other completion. It is a complete map of semantic overlap.

### **Phase 4: The Density Matrix and The Trace**

A map of similarities is mathematically interesting, but it is not a valid probability distribution until it is normalized. We accomplish this by dividing the entire Gram matrix by the number of samples, *N*.  
**Why we divide by N:** The Trace of a matrix is the sum of its diagonal elements. Because every vector is normalized, the diagonal of our Gram matrix is entirely made of 1s. Therefore, the trace of the Gram matrix is exactly *N*. By dividing the matrix by *N*, we force the Trace to equal exactly 1\.  
**The Conceptual Payoff:** In quantum mechanics and probability theory, a Trace of 1 is the mathematical equivalent of saying "the sum of all probabilities equals 100%." Dividing the Gram matrix by *N* bridges the gap between linear algebra and statistics. It transforms our raw grid of semantic similarities into a rigorous, valid probability distribution (a density matrix) across the semantic space, unlocking the use of quantum information metrics.

## **Appendix C: Defending the Specific Quantum Metrics vs. Classical Alternatives**

To rigorously counter the "baggage" critique, it is instructive to examine what occurs when we attempt to replace our chosen quantum metrics with purely classical equivalents. Measuring semantic diversity and geometric distinguishability in high-dimensional continuous spaces via classical statistics forces researchers to introduce severe mathematical contortions.

### **C.1 Von Neumann Entropy vs. Classical Shannon Entropy**

**The Quantum Metric:** Von Neumann Entropy calculates the entropy of the density matrix eigenvalue spectrum.  
**The Classical Alternative:** Shannon Entropy.  
**The Classical Baggage (Discretization Artifacts):** Shannon entropy strictly requires discrete, mutually exclusive bins. To apply Shannon entropy to a continuous 768-D embedding space, one must introduce a clustering algorithm (like K-means) *prior* to calculating the entropy. This introduces massive, unscientific baggage:

* It forces the researcher to make arbitrary decisions regarding hyperparameters (e.g., choosing the number of clusters, *k*).  
* It imposes hard, artificial boundaries on a continuous semantic space. If two semantically identical sequences fall just millimeters apart but cross a K-means cluster boundary, classical Shannon entropy treats them as entirely disjoint events, radically skewing the diversity measurement.

Von Neumann entropy completely bypasses this. By operating on the continuous eigenspectrum of the matrix, it respects the inner products (overlaps) natively, requiring no arbitrary clustering or artificial boundaries to measure the "true dimensionality" of the semantic space.

### **C.2 Trace Distance vs. Optimal Transport (Wasserstein Distance)**

**The Quantum Metric:** Trace Distance measures the maximum geometric distinguishability between two quantum states, natively bounded between 0 and 1\.  
**The Classical Alternative:** Wasserstein Distance (Earth Mover's Distance) or empirical Maximum Mean Discrepancy (MMD).  
**The Classical Baggage (Computational Explosion):** While MMD is excellent for binary hypothesis *detection* (answering "did the distribution change at all?"), measuring the absolute geometric *magnitude* of a semantic shift in classical continuous probability spaces typically requires Optimal Transport metrics like the Wasserstein distance.

* Calculating the Wasserstein distance requires computing the optimal routing of "probability mass" from one distribution to another, which scales abysmally (![][image15] or worse).  
* Furthermore, it is highly sensitive to the subjective choice of a ground cost metric.

Trace distance resolves this elegantly. Because the density matrices already encapsulate the aggregate structure of the sample spaces, trace distance simply computes the bounded magnitude of semantic shift using the matrix eigenvalues. It requires no routing optimization, vastly outperforming optimal transport in computational feasibility and mathematical simplicity.

### **C.3 Quantum Relative Entropy vs. Kullback-Leibler (KL) Divergence**

**The Quantum Metric:** Quantum Relative Entropy (QRE) measures the information divergence when substituting one density matrix for another.  
**The Classical Alternative:** Kullback-Leibler (KL) Divergence.  
**The Classical Baggage (The Curse of Dimensionality):** In discrete token spaces, KL divergence suffers from massive sparsity issues—if model B never predicts a token that model A predicts, the formula attempts to divide by zero, requiring arbitrary "smoothing" to function.

* If researchers attempt to apply KL divergence to the *continuous* embedding space instead, they must first calculate continuous probability density functions (PDFs) using Kernel Density Estimation (KDE) or Gaussian Mixture Models (GMMs).  
* In a 768-dimensional space, KDE suffers catastrophically from the "curse of dimensionality." Attempting to construct a valid probability volume in 768-D with limited samples (e.g., ![][image16]) results in almost purely noisy probability estimates.

Quantum Relative Entropy avoids the curse of dimensionality by abandoning the need to fit a continuous PDF. It constructs its valid density operator seamlessly from the finite, discrete ![][image17] Gram matrix, providing algebraic precision without requiring impossible density estimations.

## **Appendix D: Defense Presentation Slide Outline**

To help visual learners on the committee grasp these concepts without getting bogged down in equations, the following slide structure is recommended for the defense presentation. This outline directly supports the mathematical and conceptual arguments made above.  
**Slides 1-2: Confronting the "Baggage" Critique**

* **Visual Strategy:** Show a side-by-side comparison. On the left, a complex, tangled web representing purely classical Markov chains, arbitrary K-means clustering, and exploding joint probability distributions. On the right, a single, elegantly structured matrix representing the Density Matrix.  
* **Talking Point:** "We are importing linear algebra, not physics. Quantum frameworks offer a parsimonious, mathematically sound solution for modeling contextual semantic overlap that classical event sets cannot natively handle."

**Slides 3-4: Visualizing the Correspondences**

* **Visual Strategy:** A step-by-step diagram mapping physics to NLP. Show a quantum particle in a superposition next to a single LLM token vector prior to decoding.  
* **Talking Point:** Walk the committee through the token's lifecycle: First, it is entangled with the entire sequence via self-attention. Next, it exists in a superposition of meaning. Finally, it collapses into a discrete state upon measurement (decoding).

**Slides 5-7: The Mathematical Deep Dive (From Embeddings to Density Matrices)**

* **Visual Strategy (Slide 5):** An illustration of raw embeddings being projected onto a 3D unit sphere to demonstrate the normalization process (isolating direction/semantics from magnitude/length).  
* **Visual Strategy (Slide 6):** A simplified heatmap of the Gram matrix. Point out that the diagonal is solid (representing self-similarity) and the off-diagonals show semantic overlap between different completions.  
* **Visual Strategy (Slide 7):** The final formula showing the division by ![][image18] to achieve a Trace of 1\.  
* **Talking Point:** Emphasize that dividing by ![][image18] is the crucial step. "This is where the magic happens—it transitions the math from a simple geometric mapping of NLP embeddings into a rigorous, valid generalized probability distribution, bypassing the need for computationally heavy classical kernel density estimations."

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAWCAYAAAD5Jg1dAAAAhUlEQVR4XmNgGHrgLxD/h2IVNDk4AElyovH7kPhg8AgqgQzEsYiBBUDWogOQuBy6wHVkASgAia9GFziFLAAFIPEf6AJHkAWgABYCKAKnkQWgACT+CV3gPrIAFIDE56ALYAQFA0SMA1lgLVQQGURiEQMDkGA5Gl8DiY8CZjFAFHxAlxhWAABg9SkpOCtmvQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAYCAYAAAC4CK7hAAACOUlEQVR4Xu2Xz0tUURTHT0QFRWArF4GbzBC3bQOhRYu0XNq/YPRj5UYsopJs40YCEZn+gGjnD3QfWBBkf4G6cRFuNAgh6nznnfve9XvPezM6NlHMBw7c8zn33nfvm5n73oh06PBf8U3jNMsGnNHYYRnzUuOXxSTV/gRrGq9YOvyQYl33zL3WqOU9SmjXRnCdZrkkaX/OE9qxEVzjIcsKPku68DHHHaJdGzkK4avFeC6naiOjGv0sG/BM43yU35YGC3BA//ssJfM3WAa8jexrbFj7umR9uotynRHzgxrj1v6kMWvtAE6cqo30Sla/qzGvsW65B/wBywBv5Ke5mHOOQ36Tcu4D4L6wNAYkHVM2D8A8ZbVkI8iHozwAv2dtPAuQ43QJbJlj4GosDdT4DldtBPOU1eqFJ9YesvxaUc7hC6D9gHLvInDPWUr2+0OthzzcI3IBzONdow4KT63dZ/mtopzDC/1o+QuNRarFwC+xlOyT4DEXHBeDeUrrKOCkifO5KA/Ar1LeDOjnvWJ8l3SOD5FbiQtG5cGBAl5XAl/NxXgPI+TheL5MtZh3ko4FdyT1yIPjGoB7y3JKioHxBGDB8nCHvEnxpObxCNxpxhsPdjW2NZY1NjWuSNYXX9uuqF8ANZygJ8aE+Iu7KJm/St7rexxOap4cTPiYpYEaPtGYafE/qaOAo/8Ny1bBUx4L5v8WeH0ou2tlvllaHV9J/BtCzBwuJxx3MRh3iuXf5KzGe5YNwB8yjOvwT/Abwjm3SqpeE18AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAd0lEQVR4XmNgGNqADYj/I2GCAKRoEbogNkCUaf0MRCrE6T6Q4AIgNoKyQXgNsgIQAAlORuLnQMUYkcQYvkMFkUEnFjGsbsEQAxkNEohDFoSKbUQTwzANBEBiTFB2D7IgMkC2dguyhCNUAtkDIHoJED+FKRoFOAEA/hEmvhEXSisAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAhUlEQVR4XmNgGAXUBn+BOBPK5oXy5yOkIeA/En0GiKOh/BwkOQZJILaAskGCsTAJJDEwkILSNsiCUMAPFVNGFgQJoCv8jEUMLLAFixhWhaZIfGeoGAqohgrCJGShbCa4CihAVpTHADENKwAp2o0uiA2AFHKgC6KD50D8FIi/AnEPmtxgAAD0jyM9OqCmvwAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAYCAYAAABDX1s+AAACRklEQVR4Xu2XO0gcURSGDwqGFFZCAiGFID5SJ40xlRYq0cZK8NGbIhKwTRUL7cROiESwFSsJJOm19VFY2CooWCgqIZGQ3D9zL3v899zZXXcyatwPDjPnu6+Zu3fmzorUqFHjHnHsop5lCTZcNLP8F/xWkQffXMywLJO8rlGGJb/Bqh2n2vZl0SL5DIQx3rKsEPTRxTJrnkp+E1It45JNP6k8kfRB8Eg9Y0k8UOePXCyrHLyW9DGYhywUlfRzLWITcu5i25+/kKTO40LxX767uJRkBziSpM6AP04WqsmhdxY/JXnZovyjiz0XHT5vVPUC8A0ss8SakF+GwyrQbpRygHyOHIDfYun44o/9ktTBSgpsesfAdbPMEmtCkA+SA/Bn/tyaNOTsANwnllKoi5XG7Q4MB+DeswR1Li4qiBg8IWHJtysX0Dc8pM4DyKfIAfgPLBUoPzEc9w/gllhmCU9Im897lQvwRYY8PD4/VJkGZZ9ZKlD+3HCxCXnHMkuapXhg5AvkAPxXf44VOqHK0kA7vFgtXkrx+PjWYBeAf8UyS8IbXbNjuDeGQ44dqNVFE5VpVqS4bQCPM5chXyIX4LqZgs51YOsMLHq3rsoZbh+iT1fyWO0BPLZ4HPE/B8eRKzUKYLuN9XPj4MKmWTrmxb5oywH4TpYR1uTqj3ariN0gsMpmpXi36xG7boxK6ubOqdi7yr7EL5y99S0TY8zFLsvbBv634IZ0YNdII0wAVgs+vvAI4PO/FOVO3J0DL8ZVliX4byejRh78AfQevDfw9Xz3AAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAYCAYAAABA6FUWAAABf0lEQVR4Xu2WPUoEQRCFCw08gKCwGCyIf5GJIHgBRTQxMvACmph4BEORxczAyMA7eAAPoEfQxEQQFRURred2sTU11eMKs9Du9AeP7XpVs9PV29M7RJlMZlj5UhpqtqkBTU5TA5qcogY02aLqJrGdF6xpGFPjCda5ipMg1uQz6zqMl6hbM9lL//DK+mC1WffUrdkIn/u9siio22Oth/HA8Jr8dDz8WtrbMTFA3DFeDO9a69WG1yTiTeMB+E9h7C1EvxNFza3jla4dYb38QTFsk7Ld5pQn6IlsqbGA+MB4HqgbdTz7fbVhm5wN8aryBDsRiWXrvqtcjCPym4H3YM26aFP5pohPjQfgX4YxdtKuyvWLXShw7Hi1Mk/lG9w4Hk5B6yHGyTvDGje5GF6TshsGgtxQhL8B4Sx4VypvsdeL1nSRAfll1hvrIsSLhYqEwOQOrcmckL8goCqXJFWTjeViOyJZHsk/Te8o3gj85F75fgPvqfZ5XClUFIk1n8n8Z74BIFiFQuiMc5oAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAe0lEQVR4XmNgGAW0BizoAuggEYj/A7EhEF8E4t+o0hDQxABRhAxAfCs0MbAgExaxs8gCm6GC6AAktgtdAF1hNVSMA1kQJJCBLAAVQ9EsAhUoRhILgIqhgMtQwS9AvAGIn0L5GADDClwApCgLXRAd8DAQadp8BiIVDhAAANkLIXj010V/AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAYCAYAAACyVACzAAACWUlEQVR4Xu2Yv2sUQRTHXxR/dCkCBsTiUGNiZxERBGuDPxobLfIPaBEL/Q8iQa1CEMFCTrQ0WFpoL9gl2qULpIjERhKRKKLzvd3h3n337eyv2/WQ/cDj5n3fvHmzczc7uyfS0tLSUomvzg6yWIFDzrZZrIM/yprgvbOHLMb4eZzjQA4eO+uyWAe3pLnFyqpT5Ysrm1eIU9JMIdRYYJF4IuXnckfK5+bmhDRQRPLXQL81FnOSt0Zpjku4CLbpWRaJI6p9zNkr5YOrEq6hqboVL7E4TNIWa8/Zp7g9K1GfyX64xw9nv5x1nH2RqM+1+PNuv1vvtLJqWNyUqO9RDuQAeT9ZHCbWYv02NPx6tDZPPoC/TBqAvs5igLIXjRo8p6FiLRb866QB6Ltx21pQ+KwBaF0WU3gp6eOAkywoUMPMO+DsewFLgxfLb6NppXn0RdxQbQ/8+6QB6IssGryVaPuflijn/GC4B9fUoEYoXhlerDOxf1lpHv7Gve+35L6KaRDDQoT4KPbYRUCNojmF6EiyAPxnpAHo7+I2ftm3VSwE8kKvJBuSnMOOoeE+doU0TZGDpBQzkizw2dCshz74OCmnnE1QTLMqyVzPlqTHoN9Tbf1pgdgLFocFBteG49/zPNY+qDjD+d7mdKcYK/+CRHrai7WuO+5sydnrfjgB+upnvpEBE3vAomNF7IWxtKJkjZEV/2eEJmbFHkn4VM6DH9c6fPBI85TFUeGb2Kdf1j2oCpsSvTFYVB27dvAe6O8r3i4O9EhSx0VhzDEW/wcOO3vDYgXwhyLGbGlpkL/1m8DgykQJ9gAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAAAqElEQVR4XmNgGAWEwBkg/g/Eb9AlSAVrGCAGUQxAhuxGFyQVgAyh2DXZDBBDGNElSAUgQz6gC5IKKPYSMwPCEEU0ORgwQhdABhwMCBfgcs1MIC5BF4QBQQZUTT/R+ASBLANEgwiSGCtULA5JbCkQX0Piw4EqA0RxKLoEA6qXnJDEMABIcDO6IBTsYkBosoHSWA0hBYC804suSCpAjjmyAUhzCrrgKEAFAN2dKN/FRUVaAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHYAAAAYCAYAAAAvWQk7AAADP0lEQVR4Xu2Zy6tPURTHl0deUZQMjG7+AI+RwshAkoHHTChGkgnKgKQMlJKilIHyKkbKc2JyI2UgAxkwMfEYmEghr8L5/s7et/37WmvtczjHPTfnU6vfXt+19tprnV+/+/udc0V6enp6ev4jfrLwBzRRY6LgzfqwsBEWNVCkqj0Ne+rgNVmVJmpopLN1Da8nLzYGkg4qGm/+oWg57hd2j8WaNFHDQ5u1C4wWdoPFhGzPWoI1rKZ51M3XaKKGhzVrF/D6Qmwliyl3WBB7WE2z2CP18jWaqJHDmrULoK/9LAa2S6bvbSyIPSxrM8lPQe5NFmvSRI0c1qwRXMA6rC/sAIsOU1lIuC5+b15MJTdsjL8Ir7j4nA9/FmkayNtd2Lqw5liVGn+DNSu0tWF9LvgMtLOFLQ9r2KTwmmOHlHnLCntS2Pfh8AB8cLxaiE1j0cMaNiXNOZWsI+xrcA6fy/E24DOjtpq0t0GP8L4F5Hscld9z4a8gDXBeitanCzetkcvxYgDxV4rGF69t+Mzb5EfmSKlPDz7vi9pi0jSQN1nRHpMG+IwUxA6z6KE1zeRyvBhAfIqipfu8Gp9q2MKwR0M70zoX+plkneYtIt/iluh50O6yKHpuBLELLHpw0xq5HC92QvQ4tHfktw3PwX4K9GPkjxa2L6y3JDELrf6hoM0gHXBuCmJ7WfTQDmdyObkYx08qGvttwL0cIT+yRob1z8m6DqixS9G0M4GlA8RWsejhHRTJ5SC2mcWAthf+VkWzajSF1cs1RXuf+K8Ley7ld+qI+Ld+kflS1knvTTcETWOj2DHgxcaIA2p2OckDHL8yHB7wsrCvLAawB7cIXwq7GvwlQxklXo0m4DlSvgXtQXh9NBwewPu1Oil41o74RynvUd8E3wLX5wOLAdzmeHtbw/r5f1p0XcOq0QWsvrw314tpINd6gIFf77gFGxfQ2DxFqzsc1xhvNok9w1KxY9DxMKYKc8WuA7xY62jNwec/7R5ajS6AnnayKKWu/TqeLfXmQK71vY3HwM9Y/NfguenxxK8zXIRrdIX4T4poeOCCR4oa56X67LhDuMRiQtU6rVPnE2rRRI2JwkUWEjrzpvb09DTBL8tFTjCTa37AAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGcAAAAYCAYAAADnNePtAAADFUlEQVR4Xu2ZS6hNURjHP+/3QJQkMUCYyUyJUsrIK5KBSMk75THwzLtMJDLjGshYRKJ0S1EkyVhKKSQDz0L4/met5a79P9/e55571rl3X+1ffe39/f/f2mvt853H3vuIVFRUVFT0MrM1Oln8Dxii8ZbF1PxpIpplgPRsXMxJ6Zr/EHl9zRmNDhZTMVPcSQ+LtPBi7I20UV6bEGndodXGxJSxOSDlOWbYpLGCtMPiJtxJ+haNVaQVsUTSLryszdkqac/zH1dZUA6Km2w7G8p5FgrAMdaz2AJlbQ5oS3PesaAcEDfZNjaUZ5SPoDymmQUPZsGgqDlzNRaz2ICjGiNZLKCoFmubz2I72C9uMnxc84CPeOm3N/w2hnOLDeLq5mg81/iZtTNYzVnp9XHiGoz9R5kKkWVeX6ixz+8/0bjg9xvxTeO1xkRx9Wuzdg3oP1hsB91pDggNAueifbCacotjUl+DfB5pAW7OAq8x0M5Svohya5wF6i5H+WSvMXhjWXpyetIc5pTkewH4Aw3tKWkBbk7e/LE+yO+P7bJrnwJrHLNH6uumGRroEFuvgZP82kQUkaI5eLfleeCm2D60uyx64OFKMs6tY3yQrI79HZRb4xjUfCTtvteZ42LryUnRnI2S7wFrbLgQGU56AN4RyvkY4Itk9cc+P6Fxi7wiUIf7QNas8bfF1pOTojnh5jYPeJsNrdEYXGHFuVXPulXTiOVij4OGJjN4jGPVJydFc0CeN16ctzvSwhVVEfDx9CIQrs4YaFMpX6MxS2NSpBfxW+qPXXS+0K+wmJIwuRUM+9eydg3o8WOhwAtxHr5+rmu88Xke4eLCWg/uPZDjGJ/8/pTIB3jSweMRRb+98F/57Wm/fZCpyALfOtfS8l3jDotS/wK3k/A7xowRp09nwwNvBosFWHOUHmvR0Bp9ZaYCc+1i0QPvEovKUrHXncdnjYss9gfea6yL8tHS3Im3Cp6iYz7c78TgUUveOn5JvmfRTG3piBffQXlv8VDcvCHiJwgM/E4Wc0At/rPq1/RFQ9rNPY2hLFZUVLTKX8B3H8KVROoXAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAYCAYAAABDX1s+AAABwUlEQVR4Xu2VzytEURTHLxZKWVhKdoh/QFFK5MdOrGThP7CTrZQoGxt22NkoO8pKiqSUhcVsLJRsbMQCJeF8zbnNfV/nzYyaGW9xP3V693zuuW/OO9PMcy4SiUSywVcQESUOhIgDIeJAiFIDmWVRgiWJJpZF+EttTUgbCNy4rrc0Z+AGJRZ0fSmxoetSvErcSbS6fP1Mcvv/sAaCfIjcg3oP1sOU833SQN1OkLerywT8IAeUe5pd3jdqjnVLYfvn27bOMfPud12H4XxfYVwlKoh6iZcy40bPWPBAOA+B3wzWc7SXdi4ENY/kjtWH9BkO+TO5isMPwnkI/ArlyxKHui4H1HUbjs9jaLvkUHNNruJwM4uUe0Zd0r8F63KZdPa9/WDZhYwYrirwQLzbN9xTkN9LTEv0SLQFvhifzv4sdgAOQ8dP/kNiLbldHXwzVlPv6s70ilcqw+cReIA0sH+r11W9niYqCnA/Y4bLFFZz/k3UyRsK9rpYGuCFsUeu39mfmQmmXHpz8NsshQmXfoY5kqgjh7Pr5DIFGmwgN6DeAv8DaXsheIugDtcLiZzmvWFRVjl3+WZ9FPsGsX/CMhKJ1JxvXsipgpz6+toAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAAA1UlEQVR4XmNgGAWjYGgDFnQBSkAiEP8HYkMgvgjEv1GlSQdNDBADkQGIb4UmRhIAGcCERewsmhgyQHcECtjMgF0BSGwXuiAU9DBg1wMHIEl0BdVQMQ40cRiYxACR50aXgAGQZAYWMXSLYOALlAbJ+yFLwIAIA0SyGEksACqGDTADsTiUDVIzEUkODi4zQCRBtm8A4qdQPi4AkuMEYjYo+wqqNATg8yY6iEHjH2fAoRckmIUuiAMcQON3MmAxlAebIBZwiQHTR+uQxEC5Dw7mQwVHAXUAAHR3NyEks6ghAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKsAAAAYCAYAAACFtg3CAAAFGklEQVR4Xu2aS8huYxTHl3suAyJKSpSJYkLushMGjCRSMhAjZCJmRgYuA0UhKee4TERuAxODg2IgkyPJpRzJPSO33J39/573ed/n+7//tZ59e08n7V+tvm//13rWWvv5nr2/Z+/3NZuZmZmZmZmZmZT/WKhwLwsFKpfShnJIa9+xOBHXsLCP+bG1g1gcgZr3s6x/jSNau5FF0/lDMCCyR1ehkt4FLY05gMUFKp/SMtwv27ur0CUPtraDxQmI+qzBfbP9tAqVvNHa/Swu+MzW87FduYxeAZ052/ovVuT5m8UFqkYIrkhvUD4ZxdutvcVihZyvIT2jaimt5HPzY7z+lTYGdVEMAX09wWLLF5Z8J5Ge6XI+3lw8bEl/knQVe471W6z3mV8X7GrtVRYjomQAvj0sWjxGgav3ZUvjvK2Ayqm0kqh/z3eraX0o/7IwACyWqCfvXKDdwaIAcbgDK1RuPgbnWr/FmvOqXJnItwaCo8lWxW4XWo0cj5/vlI4ClVNpJfD/wuIC1XvG0/vyKQsDiXoFt1ny82KJxmSusBSHPbtC1eZjcL6t1/f4efFT5S6B704WFcdaCr6aHQWqGI5fIy3i/daOXPyu8mWUrrTM4Zb857Gj5SJLvhfYsQC+i1kcwF8sDCSaF3ChJf/NhXbVQqvxh/lx2E8qn9IusG6L9cDWvlr8XjuvVyz2L9lt9UBVDMd4yutKucFW+TJKV1oG+2blf96SjjuKB/x/stiT71kYAfrB/tHjcUsx5cMQ3myo82e8Ofd0oHTcALos1nJsVAPkG06VWiKgYvg4gmNVvozSlZbJub61dCXD8gMjtioRXS5UkGt0tTe3RvUjP+RE5Pys4TxqIA53128szdHXlrZ+eMuQ/+MxXAvgP1FtseIV3i3Fseqbqfm3QFC0X83bhH9I75S85eTWniItal7pSsvAN2S/CnZY7K/h1R1CrVegYnCM84i4zFLcwexoudSS7zp22HotcInVFyuPi97WZGp+O95SULRfxUJWiZSmyBOsTKF0pYGjLPm8fWdUB+CNROSvMeWHC7VeT7Hk30U6NO/NSgZ76ii3V1tpjcWL9SNb5WM7rohjVK1tfGJxEJqCXz3tRuMyO1s7jUXzJwcoXWngPfN9IKoDXrfYHzHVQ1UGfTzCYoF3LtBwHhHe2IznV1pj8WLFVoO51lKu69lRoGptw2syE/k9vcSL6ZtXaSDKc6rFftD14eQuYT8ILdsZaVhnsE2K+vjYfD/02h0eMd6rQmxl4L+cHaZrNuYvVhUPTrDk4+1giTd2iffHPNOSru6oGfijz8JV3gzyen6lKw1A/51FS6+xvHMrgX8nix2o5e1L1GvkAy9a7L/Bkh9P3MweSz68OlKovI3pxYoPG75ksQC5vIsK21BVa4s8AZ79tgp1QWPqls+5MFmZfIWxlfAxYI3HK7tpGe2DuMNY7EDfj5g9uGdlpy+jfRDHfGjruZThfaiHytvY9sX6ga3nLGEfjOccN5z8AcJGyA9nU6NyKm0KhuQdMmbTbKonlbcxfWcdA+qoNxWTgiLHsDgSNUFKGwuu5MdYrIBvij3H4n7AA639yuIEqHlvbNrFerTpOpOziUIqn9LGMiTnkDH7ik30pnI2Nu1iRQ21n94Id1u6sqdCTZDSxoB83vdpI+5hYT9jE/PENDbdYn2otWdY3DTPsjACfIWQUdpQ8NR6KIv/E3BeL7E4Ajw8MSfasAtd8TQLMzMzMzObZC/7CAwUvZkbjQAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAYCAYAAAC4CK7hAAAB6ElEQVR4Xu2XzytFQRTHjwViRRKyVWRDWSilV3ZkzUIkNkIh/gMlbImdJ/8AC6EsvH+B8hfYSHaUHwvMMXNz+t4z982te5+UT53unO935szMfe9N84j+yYU+EycmOtD4S5yL9qOJW5FXnG4TJRQDeTfx4NpFE5/CY6pN3IMWBBfiWDOxKnIfVZTsp4HrLKFo2Ca7ySC2yBaqQ8PwRv7F+vQ0bJKt04+GIGieYyrfkf0j0IadnhU7Jj5QdMxTmbl6yXZYQQPQvmKcT4OWlgLkOIckyVMXqKH1w1xSg4IHrjHl2u0u98HeIIoM/6DZPEVDIc1GWO9xzygfNTEntIgbE5OuzV6n8BD2+ZSLoS1Oo5Xifccgl8y4J46JtGXQQrmmeL1vtIk0nsn2WxDahtOQC/dsIt1nbQjFQIqk1wzeiNbvQNEk2tsbULQ0rJNnvLZA5IpsnwLos073odXWtDSckWf8K3kMRz1ZvwQ600XJY9mbULRF0U4LX1W849jgDSHNZL19NAS+oo0U99qE1kL2sEgLjz9EURJ95CMmxkXO96gkuE8timRvtLgRhrVdE3doBOKbLwaf/Q0oJvBCP6dUJdBeTmbkWlzwZGIPxSzh/xLRFSNPKvLC8p4k5PeaGXlt5pLCL6D//DpfB1yS5ChaCcwAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAYCAYAAAC2odCOAAAB0UlEQVR4Xu2XzytEURTHT0R+lYVY2dpIUSJJUv4J/wFlY8NGNrJip/xYshIrRZRIdsrO2srCSimRFHHOe/fOu/c4546paa5yP/Vt3v2ce8Y48+b1HkAikUj8XVoxX04+/XKGW7eJySoXjDnMIaaHFxgbmGUuQ7xAeABdmHcua8gYZh/Cn5GgWq853jVrzgz4no5PnbUKbZwyrwesRmxhmrmsIdOYJswjyP84Qf5WcJeCqxNcWd7Mq/ZNSS4G5YbUwdy18ZZNtraQu+fSZRgzbo5vIG8YLMoZ0hvHQBvSJMh+G3wfOgkkX+KZraWGc7aOhTakHZA9XZirMiRe5A1rmAZnHeIEiv7fphK0IV2A7Bch9/Z6qv1NzZew1yNLC+QN9gwLNtcYbUhnIHs7JPsla8PQfMYIZoJL8JvU5ghoQ1oH2a+A77VhaD7jlQvDLORN9FM7ZrWYaEPqB9nvQRWGpBagaKznhQB00zlfYSpBGxJBfpS5J+MtC2xtIXfFJdEJcoPlDsL1GISGRI9U/LGK9i4JbsBZNxrn0Wakm3ZvR8GP5kjwz2sz5G4y7gjys/8D8+CXM7oh30ePOn3mmO6z/h38ZliCHu7p15RIJBKJxD/gGyP6tdKVYu5EAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAYCAYAAABEHYUrAAABOklEQVR4Xu2UwUrDQBCGB8RDreBJX0QoPZWevCsI4sm7j+ID+A6ePeoDiBfP4lOIUnooOpvskM3fZLPbTAKF+eAn3Znsnz9DN0SGYewzU9ZfoE29XRD2RWOinvGH4jedsdZYHBm1jM7gxl+foOd4ZE2wODJqGVf+2ja5ptrYqGScsRb+9zuVm86rdkGSkeeDdYHFgBwvQS3jN6ybJvcC6y4+WVdYpG3fVNQy4iY0emAdButUvljXwRqfkwPu3TmjnAXhiEojmSY+KAd54T4eDpWMc9YSi1SfXJJRBLf/DosZqGX8xYLnnkoD9/d4hl4OEsJ9VG7DRgZqGWMTkckdYCMR9HYvHJ7hVNAnJDnjKcWN3HmL9WO07XtjXWIxQu+Mx1RNRHRSu6MiatRC155XLDQwdEbDMAzDGIJ/tvF8pvn7nOgAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA1UlEQVR4Xu2QsQ4BQRCGp1JQiUeQaPSiUEnU4j3UolZS6BQ8gkIkFAreRKdEIhoazGVncrv/zjWic1/yJbv/P7u3OaKcb9iwO3ELnbJm5+yUXbCzsA55i0UsyGXad6ELGLIlSoctkryNIaKH9aKR1ylZHwjQoaasrUMHDCz23lovqnjZmC14e5MBhUNLil9lvTDCGvrZRRNyeV/2d6/L5IGBoK/qsQ3oIlpsB0PhRu6iFxYWTwwA/FcmR3JDZSw8kv6KoVJlL+xJPLOrYCKlztYwzPl7PrpUOGDX90mhAAAAAElFTkSuQmCC>