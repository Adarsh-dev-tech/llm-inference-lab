# Theory Reference: Transformer Architectures and Self-Attention Mechanics

Welcome! This guide is written from the ground up to explain the Transformer architecture. Even if you have zero prior background in Machine Learning (ML) or deep learning, this document will build the intuition from absolute first principles (word embeddings, vectors) and guide you up to the advanced mathematical equations and tensor shapes that power modern models like GPT and Qwen.

---

## 1. The Pre-Requisites: How Computers Understand Language

To understand the Transformer, we must first understand how computers process human language. Computers cannot read words; they can only process numbers and perform arithmetic.

### Concept A: Tokenization (Breaking Text Into Units)
When you feed a sentence into an AI model, the first step is **Tokenization**. The text is split into small pieces called **Tokens**. A token can be a whole word, a part of a word (subword), or even a single character.
*   *Example*: The sentence `"Inference optimization"` might be split into tokens: `["In", "ference", " optim", "ization"]`.
*   Each unique token is mapped to a specific integer ID in a massive dictionary (the vocabulary). For instance, `"In"` might be ID `410`, and `"ference"` might be ID `12803`.

### Concept B: Word Embeddings (Mapping Meaning to Space)
An integer ID (like `12803`) is just a label; it doesn't carry any meaning. To represent the meaning of a word, machine learning uses **Vectors** (lists of floating-point numbers). We call these **Embeddings**.

Imagine mapping words onto a multi-dimensional map of meaning:
*   In a simple 2D map with axes for **"Feline-ness"** (how cat-like a word is, on a scale from 0 to 1) and **"Size"**:
    *   `"Housecat"` might be at coordinates `[0.9, 0.1]` (High feline-ness, small size).
    *   `"Tiger"` might be at coordinates `[0.9, 0.9]` (High feline-ness, large size).
    *   `"Dog"` might be at coordinates `[0.1, 0.2]` (Low feline-ness, small size).
*   By representing words as coordinates, the computer can calculate the mathematical distance between them. Similar words will point in similar directions.
*   In modern LLMs like Qwen2.5-7B, instead of a 2D map, we use a **3,584-dimensional map**. The vector for each token is a list of 3,584 numbers, capturing extremely subtle semantic traits. We refer to this vector length as the **Hidden Dimension ($C$)**.

```
              Embedding Space Intuition (e.g., 2D slice)
                 ^ Feline-ness
                 |
                 |   * Tiger [0.9, 0.9]
                 |
                 |   * Housecat [0.9, 0.1]
                 |
                 |                         * Dog [0.1, 0.2]
                 +------------------------------------------> Size
```

---

## 2. The Core Problem: Modeling Sequences of Words

Before the Transformer was invented in 2017, the state-of-the-art models for text were **Recurrent Neural Networks (RNNs)**. 

### How RNNs Worked (Sequential Reading)
RNNs read text the way humans do: one word at a time, from left to right.
*   When reading `"The dog chased the cat"`, the RNN reads `"The"`, updates its internal memory (hidden state), then reads `"dog"`, updates its memory, and so on.
*   **The Sequential Bottleneck**: Because token $t$ cannot be processed until token $t-1$ is finished, RNNs cannot perform parallel calculations on GPU hardware. Training on large datasets is extremely slow.
*   **Context Loss**: For very long sentences, the network's "memory" fades. By the time the RNN reaches the end of a long paragraph, it has mathematically "forgotten" the beginning.

### The Transformer Solution (Parallel Processing)
The Transformer architecture (Vaswani et al., 2017) solved this by discarding recurrence entirely.
*   Instead of sequential processing, the Transformer ingests **the entire sequence of tokens simultaneously**.
*   To calculate how words relate to each other across the sequence, it uses the **Self-Attention Mechanism**.

---

## 3. The Self-Attention Intuition: The Search Engine Analogy

Self-attention allows each word in a sequence to look at every other word, evaluate their relevance, and update its own representation (embedding vector) with context.

### The Query, Key, and Value Analogy
To perform self-attention, the Transformer projects every token vector into three separate representations: **Queries ($Q$)**, **Keys ($K$)**, and **Values ($V$)**. Think of this like a database search engine:

1.  **Query ($Q$)**: The search term you type. It represents: *"What information am I (the current token) looking for in this sentence?"*
2.  **Key ($K$)**: The indexing tags of the database files. It represents: *"What characteristics do I (other tokens) possess, and what context can I offer?"*
3.  **Value ($V$)**: The actual content of the files. It represents: *"If I am relevant to you, what actual semantic meaning/information do I bring?"*

```
     Token Vector (Embedding)
             |
     +-------+-------+
     |       |       |
     v       v       v
   Query    Key    Value
    (Q)     (K)     (V)
     |       |       |
     v       v       v
  [Search] [Tags] [Content]
```

### Contextual Disambiguation Walkthrough
Let's see self-attention in action with the word **"bank"** in two different contexts:

#### Sentence 1: `"The bank approved the business loan."`
1.  The token `"bank"` projects a Query: *"I need context related to actions, finances, or decisions."*
2.  All other tokens project their Keys:
    *   `"The"` Key: *"I am a grammatical article."* (Low relevance)
    *   `"approved"` Key: *"I am a verb denoting authorization."* (High relevance)
    *   `"loan"` Key: *"I am a noun denoting financial borrowing."* (Very High relevance)
3.  The model calculates similarity scores (dot-products) between the Query of `"bank"` and the Keys of all other words. The scores for `"loan"` and `"approved"` are extremely high.
4.  These similarity scores are turned into percentages (using a function called **Softmax**), yielding:
    *   `"loan"`: 70% attention
    *   `"approved"`: 20% attention
    *   Others: 10% attention
5.  The model takes the **Values (semantic content)** of `"loan"` (finance, debt) and `"approved"` (authorization), multiplies them by their attention percentages, and adds them to the embedding vector of `"bank"`.
6.  **Result**: The representation of `"bank"` shifts in coordinate space towards "financial institution".

#### Sentence 2: `"The bank of the river was muddy."`
1.  The token `"bank"` projects the same Query: *"I need context related to physical locations, water, or nature."* (Adjusted by other surrounding signals).
2.  The token `"river"` projects its Key: *"I am a physical body of flowing water."* (Very High relevance).
3.  The token `"muddy"` projects its Key: *"I am an adjective denoting wet earth."* (High relevance).
4.  The attention score for `"river"` and `"muddy"` dominates.
5.  The value of `"river"` (flowing water, geography) is merged into the embedding of `"bank"`.
6.  **Result**: The representation of `"bank"` shifts towards "sloped earth near water".

### Step-by-Step Numerical Attention Walkthrough

To see how these concepts translate into raw arithmetic under the hood in a model like **Qwen2.5-7B**, let's trace the math with concrete numbers using a simplified sequence: `["The", "bank", "approved"]` (sequence length $T = 3$).

#### Qwen's Scale vs. Our Simplified Walkthrough
In the actual Qwen2.5-7B architecture:
*   **Hidden Dimension ($C$)** = `3584` (each word/token is represented by a list of 3,584 numbers).
*   **Attention Heads ($H_Q$)** = `28`.
*   **Head Dimension ($d_k$)** = $3584 / 28 =$ `128`.
*   **Grouped-Query Attention (GQA)**: 28 Query heads share 4 Key-Value heads (meaning 7 Query heads group together to share a single Key-Value head).

To make the calculations easy to follow, we will scale this down to:
*   **Hidden Dimension ($C$)** = `4` (each token is a vector of 4 numbers).
*   **Head Dimension ($d_k$)** = `3` (Query, Key, and Value vectors are lists of 3 numbers).

---

#### 1. Setting Up the Inputs (Embeddings & Normalization)
Suppose our 3 tokens have passed through Qwen's word embedding layer and RMSNorm, giving us three token vectors ($X_{\text{The}}, X_{\text{bank}}, X_{\text{approved}}$) of size $C=4$:

$$X_{\text{The}} = [1.0, 0.0, 0.0, 0.0]$$
$$X_{\text{bank}} = [0.0, 1.0, 1.0, 0.0]$$
$$X_{\text{approved}} = [0.0, 1.0, 0.0, 1.0]$$

---

#### 2. Projecting to Queries (Q), Keys (K), and Values (V)
Qwen projects these hidden states using trained weight matrices. Let's define simple projection matrices $W^Q$, $W^K$, and $W^V$ (each of shape $C \times d_k$, or $4 \times 3$):

$$W^Q = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}, \quad W^V = \begin{bmatrix} 0.5 & 0 & 0 \\ 0 & 0.5 & 0 \\ 0 & 0 & 0.5 \\ 0 & 0.5 & 0 \end{bmatrix}$$

We calculate $Q$, $K$, and $V$ by multiplying each token vector by these matrices:

##### **Queries (Q)**: *"What am I looking for?"*
*   $q_{\text{The}} = X_{\text{The}} \cdot W^Q = [1.0, 0.0, 0.0]$
*   $q_{\text{bank}} = X_{\text{bank}} \cdot W^Q = [0.0, 1.0, 1.0]$
*   $q_{\text{approved}} = X_{\text{approved}} \cdot W^Q = [0.0, 2.0, 0.0]$

##### **Keys (K)**: *"What traits do I offer?"*
*   $k_{\text{The}} = X_{\text{The}} \cdot W^K = [0.0, 1.0, 0.0]$
*   $k_{\text{bank}} = X_{\text{bank}} \cdot W^K = [1.0, 0.0, 1.0]$
*   $k_{\text{approved}} = X_{\text{approved}} \cdot W^K = [2.0, 0.0, 0.0]$

##### **Values (V)**: *"What content do I contain?"*
*   $v_{\text{The}} = X_{\text{The}} \cdot W^V = [0.5, 0.0, 0.0]$
*   $v_{\text{bank}} = X_{\text{bank}} \cdot W^V = [0.0, 0.5, 0.5]$
*   $v_{\text{approved}} = X_{\text{approved}} \cdot W^V = [0.0, 1.0, 0.0]$

> **Rotary Position Embedding (RoPE) Note**: In the actual Qwen model, right after calculating $Q$ and $K$ and before the dot-product, RoPE rotations are applied to the vectors to encode positional information. We skip this here to keep the arithmetic clear.

---

#### 3. Calculating Attention Scores for "bank"
Let's calculate how the second token, `"bank"`, distributes its attention. We take the Query of `"bank"` ($q_{\text{bank}} = [0.0, 1.0, 1.0]$) and compute the dot product (similarity) with the Keys ($k$) of all tokens:

1.  **Similarity to "The"**:
    $$\text{Score}_{\text{bank} \rightarrow \text{The}} = q_{\text{bank}} \cdot k_{\text{The}} = (0.0 \times 0.0) + (1.0 \times 1.0) + (1.0 \times 0.0) = 1.0$$
2.  **Similarity to "bank" (self-attention)**:
    $$\text{Score}_{\text{bank} \rightarrow \text{bank}} = q_{\text{bank}} \cdot k_{\text{bank}} = (0.0 \times 1.0) + (1.0 \times 0.0) + (1.0 \times 1.0) = 1.0$$
3.  **Similarity to "approved"**:
    $$\text{Score}_{\text{bank} \rightarrow \text{approved}} = q_{\text{bank}} \cdot k_{\text{approved}} = (0.0 \times 2.0) + (1.0 \times 0.0) + (1.0 \times 0.0) = 0.0$$

---

#### 4. Applying the Causal Mask
Because Qwen is a decoder-only autoregressive model, the token `"bank"` (at position $t=2$) cannot see the future token `"approved"` (at position $t=3$). We apply a causal mask, setting future scores to $-\infty$:
*   $\text{Score}_{\text{bank} \rightarrow \text{The}} = 1.0$ (Allowed)
*   $\text{Score}_{\text{bank} \rightarrow \text{bank}} = 1.0$ (Allowed)
*   $\text{Score}_{\text{bank} \rightarrow \text{approved}} = -\infty$ (Masked)

---

#### 5. Scaling and Softmax
To stabilize training gradients, we scale the scores by dividing by the square root of the head dimension ($\sqrt{d_k} = \sqrt{3} \approx 1.732$):
1.  **Scaled Scores**:
    *   $\text{Scaled Score}_{\text{bank} \rightarrow \text{The}} = 1.0 / 1.732 \approx 0.577$
    *   $\text{Scaled Score}_{\text{bank} \rightarrow \text{bank}} = 1.0 / 1.732 \approx 0.577$
    *   $\text{Scaled Score}_{\text{bank} \rightarrow \text{approved}} = -\infty$
2.  **Softmax Conversion**:
    $$\text{Weight}_i = \frac{e^{\text{ScaledScore}_i}}{\sum_j e^{\text{ScaledScore}_j}}$$
    *   Exponentials: $e^{0.577} \approx 1.78$, $e^{0.577} \approx 1.78$, $e^{-\infty} = 0.0$.
    *   Sum of Exponentials: $1.78 + 1.78 + 0.0 = 3.56$.
    *   **Attention weights**:
        *   To `"The"`: $1.78 / 3.56 = 0.5$ (**50%**)
        *   To `"bank"`: $1.78 / 3.56 = 0.5$ (**50%**)
        *   To `"approved"`: $0.0 / 3.56 = 0.0$ (**0%**)

---

#### 6. Weighted Sum of Values
We multiply these percentages by the Value vectors ($v$) of the respective tokens:

$$\text{Output}_{\text{bank}} = (0.5 \times v_{\text{The}}) + (0.5 \times v_{\text{bank}}) + (0.0 \times v_{\text{approved}})$$
$$\text{Output}_{\text{bank}} = 0.5 \cdot [0.5, 0.0, 0.0] + 0.5 \cdot [0.0, 0.5, 0.5] + 0.0 \cdot [0.0, 1.0, 0.0] = [0.25, 0.25, 0.25]$$

This vector $[0.25, 0.25, 0.25]$ is the final output of this single head for the token `"bank"`. In Qwen, all 28 heads are concatenated back into a size-3584 vector, which is multiplied by the Output Projection Matrix ($W^O$) and added back to the residual stream.

---

### Addressing Key Conceptual Confusions

#### Q: Logically, "bank" should have more similarity to "approved" than to "the". Why isn't that happening here?
There are two reasons:
1. **The Causal Mask (Decoder-Only Nature)**: Qwen is a decoder-only model, which means a token is prohibited from looking at future tokens during generation. When calculating the attention scores *for* `"bank"` (index 2), the word `"approved"` (index 3) lies in the future. Because of the causal mask, the similarity score for `"approved"` is overridden to $-\infty$ (meaning exactly $0\%$ attention). However, when the model moves to the next token, `"approved"`, it *is* allowed to look backward, and `"approved"` will attend heavily to `"bank"`, pulling the financial meaning of `"bank"` into `"approved"`'s representation.
2. **Simplified Weights**: In a real, pre-trained model, the projection weights $W^Q$ and $W^K$ are trained on massive datasets so that relevant words project similar vectors. In our toy walkthrough, we chose simple, arbitrary matrices $W^Q$ and $W^K$ just to keep the manual multiplication arithmetic clean.

#### Q: What is the purpose of the final vector $[0.25, 0.25, 0.25]$ we calculated for "bank"?
This vector represents the **context update** (or context delta) gathered by this attention head. It contains $50\%$ of the visual/semantic information from `"The"` and $50\%$ from `"bank"` itself. It is a package of new contextual information that will be used to enrich the original token vector of `"bank"` before it is processed by the Feed-Forward Network (FFN).

#### Q: If our input hidden dimension ($C$) is 4, why is this output vector's size only 3?
This is a standard feature of Multi-Head Attention:
1. **Dimension Splitting**: We project our 4-dimensional input into smaller $3$-dimensional heads ($d_k = d_v = 3$) so that each head can process a different semantic sub-space.
2. **Re-Projection**: To merge this $3$-dimensional output back into our original $4$-dimensional residual stream, the model multiplies it by the **Output Projection Matrix ($W^O$)**. 
3. **The Math**: Since we have $H = 1$ head of size $d_v = 3$ and we need to map it back to $C = 4$, the matrix $W^O$ will have a shape of $[3, 4]$. Multiplying our vector of size $[1, 3]$ by $W^O$ of size $[3, 4]$ yields a vector of size $[1, 4]$, which can then be directly added back to the original $4$-dimensional input token vector via the **Residual Connection**:
   $$X_{\text{updated\_bank}} = X_{\text{bank}} + (\text{Output}_{\text{bank}} \cdot W^O)$$

#### Q: Why do we transpose the Key matrix ($K^T$) in the self-attention formula?
There are two reasons:
1. **Mathematical Shape Alignment**: 
   * The Query matrix $Q$ has a shape of $[T, d_k]$ (Sequence Length $T$ by Head Dimension $d_k$).
   * The Key matrix $K$ also has a shape of $[T, d_k]$.
   * If you try to multiply $Q \times K$ directly, you are multiplying $[T, d_k] \times [T, d_k]$, which is mathematically impossible because the inner dimensions ($d_k$ and $T$) do not match.
   * Transposing $K$ to $K^T$ flips its shape to $[d_k, T]$. Now, we can multiply them: $[T, d_k] \times [d_k, T]$, resulting in a square matrix of shape $[T, T]$.
2. **Calculating All Pairwise Dot Products**:
   * A dot product between two vectors (e.g. $q_{\text{bank}}$ and $k_{\text{The}}$) requires multiplying their corresponding elements and adding them up: $q \cdot k = \sum q_i k_i$.
   * Since $Q$ and $K$ store their vectors as *rows*, transposing $K$ turns its rows into columns. Multiplying the matrix $Q$ by the transposed matrix $K^T$ is a highly optimized way for hardware (like GPUs) to calculate the dot product between every single Query row and every Key column at once. The resulting $[T, T]$ matrix contains the raw attention score for every token pair in the sequence.

#### Q: You said the Query matrix $Q$ has shape $[T, d_k]$. Since $T=3$ and $d_k=3$ in our example, why were our queries, keys, and values vectors of size $1 \times 3$ instead of matrices of size $3 \times 3$?
This is the difference between the **entire sequence (batch) view** and the **single-token view** we used for our walkthrough:
1. **The Entire Sequence View (Matrices)**: When the model runs on hardware, it stacks all token vectors together into matrices so it can calculate everything in parallel. Stacking our 3 tokens (`"The"`, `"bank"`, `"approved"`) gives us:
   * **Query Matrix $Q$** (shape $3 \times 3$, or $[T, d_k]$):
     $$Q = \begin{bmatrix} q_{\text{The}} \\ q_{\text{bank}} \\ q_{\text{approved}} \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 1.0 \\ 0.0 & 2.0 & 0.0 \end{bmatrix}$$
   * **Key Matrix $K$** (shape $3 \times 3$, or $[T, d_k]$):
     $$K = \begin{bmatrix} k_{\text{The}} \\ k_{\text{bank}} \\ k_{\text{approved}} \end{bmatrix} = \begin{bmatrix} 0.0 & 1.0 & 0.0 \\ 1.0 & 0.0 & 1.0 \\ 2.0 & 0.0 & 0.0 \end{bmatrix}$$
   * **Value Matrix $V$** (shape $3 \times 3$, or $[T, d_v]$):
     $$V = \begin{bmatrix} v_{\text{The}} \\ v_{\text{bank}} \\ v_{\text{approved}} \end{bmatrix} = \begin{bmatrix} 0.5 & 0.0 & 0.0 \\ 0.0 & 0.5 & 0.5 \\ 0.0 & 1.0 & 0.0 \end{bmatrix}$$
2. **The Single-Token View (Walkthrough)**: In our step-by-step example, we focused solely on calculating the output **for the single token `"bank"`**. Therefore, we only extracted the single row from $Q$ belonging to `"bank"` ($q_{\text{bank}} = [0.0, 1.0, 1.0]$, shape $1 \times 3$), but we still compared it against all the Key vectors stacked in $K$ (shape $3 \times 3$). If you compute the full matrix product $Q K^T$:
   $$Q K^T = \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 1.0 \\ 0.0 & 2.0 & 0.0 \end{bmatrix} \times \begin{bmatrix} 0.0 & 1.0 & 2.0 \\ 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 1.0 & 2.0 \\ 1.0 & 1.0 & 0.0 \\ 2.0 & 0.0 & 0.0 \end{bmatrix}$$
   Notice that the second row of the resulting matrix is exactly $[1.0, 1.0, 0.0]$, which are the three similarity scores we calculated manually for `"bank"`!

---

## 4. Structural Layout of a Transformer Block

In modern large language models, multiple transformer layers are stacked on top of each other. A single transformer layer (block) contains two main processing sub-layers: **Self-Attention** and the **Feed-Forward Network (FFN/MLP)**, structured with **Pre-Layer Normalization (Pre-LN)** and **Residual Connections**.

```
                      Input / Residual Stream
                                 |
                                 v
                       +-------------------+
                       |   RMS / LN Norm   |
                       +-------------------+
                                 |
              +------------------+------------------+
              |                                     |
              v                                     |
     +-----------------+                            |
     |  Self-Attention |                            |
     +-----------------+                            |
              |                                     |
              +----------------->(+) <--------------+  (Residual Add)
                                 |
                                 v
                       +-------------------+
                       |   RMS / LN Norm   |
                       +-------------------+
                                 |
              +------------------+------------------+
              |                                     |
              v                                     |
     +-----------------+                            |
     |    FFN / MLP    |                            |
     +-----------------+                            |
              |                                     |
              +----------------->(+) <--------------+  (Residual Add)
                                 |
                                 v
                        Next Layer Input
```

### Sub-Component 1: Layer / RMS Normalization
Normalization is like scaling values. Without normalization, as numbers are multiplied repeatedly through deep networks, the values could grow exponentially large (exploding gradients) or shrink to zero (vanishing gradients), which halts learning. 
*   **Pre-LN** normalizes the inputs *before* they enter the attention and FFN blocks.
*   **RMSNorm** (Root Mean Square Normalization) is a computationally cheaper variant of LayerNorm that scales vectors based on their root mean square, skipping the mean calculation to speed up GPU execution.

### Sub-Component 2: Residual (Skip) Connections
Notice the line in the diagram that bypasses the Self-Attention and FFN blocks entirely, merging back via an addition sign `(+)`. This is a **Residual Connection**.
*   Instead of forcing the representation to completely change at every block, the layer computes a *modification delta* ($\Delta X$) and adds it to the original input:
    $$X_{\text{out}} = X_{\text{in}} + \text{Layer}(X_{\text{in}})$$
*   This acts as an "express lane," allowing raw information to flow through the network undisturbed while individual layers focus on adding small contextual corrections.

### Sub-Component 3: Self-Attention Sub-Layer
This is where tokens exchange context information across positions (as explained in Section 3).

### Sub-Component 4: Feed-Forward Network (FFN / MLP)
Once self-attention has finished passing context between tokens, each token's vector is processed through a **Feed-Forward Network** (sometimes called Multi-Layer Perceptron / MLP).
*   **Crucial Detail**: The FFN processes each token **independently** of other tokens. 
*   **Intuition**: If Self-Attention is where tokens "talk to each other" to gather information, the FFN is where each token goes to its "private thinking space" to compile and store that gathered context into its hidden representation.
*   Modern models use **SwiGLU** activation layers within the FFN, which perform gated matrix multiplication to capture non-linear relationships.

---

## 5. Positional Embeddings: Why Sequence Order Matters

Because the self-attention formula processes all tokens simultaneously, it is mathematically permutation-invariant (order-independent). 
*   To self-attention, `"Dog eats food"` and `"Food eats dog"` are exactly identical because the similarity calculations do not encode position.
*   To solve this, we add positional information to the token vectors.

### Types of Positional Encodings:
1.  **Absolute Positional Encodings**: Add a unique sine/cosine wave pattern of numbers to the embedding vector based on its absolute index (e.g., adding a specific vector pattern for token index `1`, another pattern for index `2`).
2.  **Relative / Rotary Positional Embeddings (RoPE)**: Instead of adding static vectors, RoPE projects Query and Key vectors and rotates them in 2D coordinate slices by an angle proportional to their sequence index. This mathematically preserves the *relative distance* between tokens, which makes it much easier for the model to generalize to long sequence windows.

---

## 6. Architectural Comparison of Transformer Variants

The original Transformer has been adapted into three major configuration branches based on how attention is masked:

```
   Encoder-Only (BERT)             Decoder-Only (GPT, Qwen)          Encoder-Decoder (T5, BART)
+------------------------+        +------------------------+        +------------------------+
|   Bidirectional Attn   |        |      Causal Attn       |        |   Bidirectional Attn   |
| (Full Context Visible) |        |   (Future Masked Out)  |        |        (Source)        |
+------------------------+        +------------------------+        +----------+-------------+
            |                                 |                                |
            v                                 v                                v
   Representation/Class              Autoregressive Text              Cross-Attention Link
   (Classification, NER)             (Generation, Chat)               (Translation, Summarize)
```

### 1. Encoder-Only (e.g., BERT)
*   **How it works**: Uses **Bidirectional Attention**. Every token can look at every other token in both directions (past and future).
*   **Intuition**: Think of this as a "Reader" or "Editor." It is ideal for analyzing complete sentences.
*   **Use Cases**: Sentiment analysis, classifications, finding names in text (NER).

### 2. Decoder-Only (e.g., GPT, Claude, Llama, Qwen)
*   **How it works**: Uses **Causal Masking**. During generation, a token at index $t$ is blocked from looking at future tokens $t+1 \dots T$. It can only look at its past.
*   **Intuition**: Think of this as a "Writer." It generates text autoregressively (predicting one token at a time, feeding its own output back as the next input).
*   **Use Cases**: Conversational assistants, code generation, creative writing.

### 3. Encoder-Decoder (e.g., T5, BART)
*   **How it works**: Uses a bidirectional Encoder to process the input prompt, and passes the output to a causal Decoder via **Cross-Attention** layers to generate a response.
*   **Intuition**: Think of this as a "Translator." The encoder reads the input (e.g. in French), and the decoder writes the output (e.g. in English) while constantly looking back at the encoder's thoughts.
*   **Use Cases**: Translation, document summarization.

---

## 7. Multi-Head Attention (MHA) Mathematical Formulation

For readers ready for the formal mathematical execution steps, here is the complete formulation of scaled dot-product attention:

### Scaled Dot-Product Attention
Given input matrices $Q$ (Queries), $K$ (Keys), and $V$ (Values):
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

Where:
*   $Q \in \mathbb{R}^{T \times d_k}$ (Queries matrix)
*   $K^T \in \mathbb{R}^{d_k \times T}$ (Transposed Keys matrix)
*   $\sqrt{d_k}$ scales down the values before softmax. If $d_k$ is large, dot products grow large, pushing softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ stabilizes learning.
*   $M$ is the attention mask. For causal generation, $M$ zeroes out future values:
    $$M_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
    Because $\text{e}^{-\infty} = 0$, applying softmax to these masked indices drops their attention weights to exactly $0$, preventing the model from seeing future tokens.

### Multi-Head Attention (MHA)
Instead of calculating attention once over the full hidden dimension ($C$), the model projects $Q$, $K$, and $V$ into $h$ low-dimensional subspaces (heads), performs attention on each subspace independently, and concatenates the results:
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

Where:
*   $W_i^Q \in \mathbb{R}^{C \times d_k}$
*   $W_i^K \in \mathbb{R}^{C \times d_k}$
*   $W_i^V \in \mathbb{R}^{C \times d_v}$
*   $W^O \in \mathbb{R}^{h d_v \times C}$ (Output projection)

---

## 8. Decoder-Only Tensor Shape Trace (Forward Pass)

The following table traces a batch of sequences through a single transformer block in a decoder-only architecture. 

### Dimension Variables:
*   **$B$ (Batch Size)**: Number of independent sequences processed concurrently.
*   **$T$ (Sequence Length)**: Number of active tokens in the sequence.
*   **$C$ (Hidden Dimension)**: Number of features per token embedding (e.g., 3584).
*   **$H$ (Number of Heads)**: Partition count of the attention layer (e.g., 28 heads).
*   **$D$ (Head Dimension)**: Hidden dimension per head, where $D = C / H$ (e.g., $3584 / 28 = 128$).
*   **$V_{\text{vocab}}$ (Vocabulary Size)**: Size of the token ID vocabulary space (e.g., 151,936).

| Step | Operation Description | Input Tensor Shape | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **1. Token IDs** | Input sequence of integer token identifiers | N/A | $[B, T]$ |
| **2. Embeddings** | Map Token IDs to dense embedding vectors + Position Embeddings | $[B, T]$ | $[B, T, C]$ |
| **3. Layer Norm** | RMSNorm or LayerNorm prior to attention | $[B, T, C]$ | $[B, T, C]$ |
| **4. QKV Projections**| Linear projections to query, key, value matrices: $X W^Q, X W^K, X W^V$ | $[B, T, C]$ | $Q$: $[B, T, C]$<br>$K$: $[B, T, C]$<br>$V$: $[B, T, C]$ |
| **5. Multi-Head Split**| Reshape and transpose to isolate attention heads | $Q, K, V$: $[B, T, C]$ | $Q, K, V$: $[B, H, T, D]$ |
| **6. Attention Score** | Scaled dot-product: $Q K^T / \sqrt{D}$ | $Q$: $[B, H, T, D]$<br>$K^T$: $[B, H, D, T]$ | $[B, H, T, T]$ |
| **7. Causal Mask** | Apply upper-triangular masking ($-\infty$ to future tokens) + Softmax | $[B, H, T, T]$ | $[B, H, T, T]$ |
| **8. Weighted Values**| Apply attention weights to value vectors: $\text{softmax}(\dots) V$ | Attention: $[B, H, T, T]$<br>$V$: $[B, H, T, D]$ | $[B, H, T, D]$ |
| **9. Concat Heads** | Transpose and reshape to merge head outputs | $[B, H, T, D]$ | $[B, T, C]$ |
| **10. Out Projection**| Linear projection $W^O$ back to residual stream | $[B, T, C]$ | $[B, T, C]$ |
| **11. Residual Add** | Add attention block output back to original block input | Block Input: $[B, T, C]$<br>Attn Out: $[B, T, C]$ | $[B, T, C]$ |
| **12. Layer Norm** | RMSNorm or LayerNorm prior to Feed-Forward Network (FFN) | $[B, T, C]$ | $[B, T, C]$ |
| **13. MLP Expansion** | FFN expansion layer (often SwiGLU: $(X W^{\text{gate}} \cdot \text{silu}(X W^{\text{up}}))$) | $[B, T, C]$ | $[B, T, 4C]$ (or $[B, T, 8/3 C]$) |
| **14. MLP Down** | Project FFN output back to hidden dimension | $[B, T, 4C]$ | $[B, T, C]$ |
| **15. FFN Residual** | FFN Residual add | Block Input: $[B, T, C]$<br>FFN Out: $[B, T, C]$ | $[B, T, C]$ (Next layer input) |
| **16. LM Head** | Linear unembedding projection to map back to vocabulary space | Final Block Out: $[B, T, C]$ | $[B, T, V_{\text{vocab}}]$ (Output Logits) |

---

## Related Learnings
*   [Learning Report: Transformer Weight Distribution, KV Caching, and Context Limits](../learnings/learning-transformer-internals.md)
