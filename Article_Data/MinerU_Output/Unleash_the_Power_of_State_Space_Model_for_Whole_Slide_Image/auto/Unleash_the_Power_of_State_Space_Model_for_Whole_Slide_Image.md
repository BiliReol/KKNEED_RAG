# Unleash the Power of State Space Model for Whole Slide Image With Local Aware Scanning and Importance Resampling

Yanyan Huang , Weiqin Zhao , Yu Fu , Lingting Zhu , and Lequan Yu , Member, IEEE

Abstract— Whole slide image (WSI) analysis is gaining prominence within the medical imaging field. However, previous methods often fall short of efficiently processing entire WSIs due to their gigapixel size. Inspired by recent developments in state space models, this paper introduces a new Pathology Mamba (PAM) for more accurate and robust WSI analysis. PAM includes three carefully designed components to tackle the challenges of enormous image size, the utilization of local and hierarchical information, and the mismatch between the feature distributions of training and testing during WSI analysis. Specifically, we design a Bi-directional Mamba Encoder to process the extensive patches present in WSIs effectively and efficiently, which can handle large-scale pathological images while achieving high performance and accuracy. To further harness the local information and inherent hierarchical structure of WSI, we introduce a novel Local-aware Scanning module, which employs a local-aware mechanism alongside hierarchical scanning to adeptly capture both the local information and the overarching structure within WSIs. Moreover, to alleviate the patch feature distribution misalignment between training and testing, we propose a Test-time Importance Resampling module to conduct testing patch resampling to ensure consistency of feature distribution between the training and testing phases, and thus enhance model prediction. Extensive evaluation on nine WSI datasets with cancer subtyping and survival prediction tasks demonstrates that PAM outperforms current state-of-the-art methods and also its enhanced capability in modeling discriminative areas within WSIs. The source code is available at https://github.com/HKU-MedAI/PAM.

Index Terms— Whole slide image, state space model, importance resampling, scanning.

Yu Fu is with the School of Information Science and Engineering, Lanzhou University, Lanzhou 730000, China (e-mail: fuyu@lzu.edu.cn). Digital Object Identifier 10.1109/TMI.2024.3475587

## I. INTRODUCTION

W HOLE Slide Images (WSIs) contain rich histopatho-logical information of tissue sections and are routinely logical information of tissue sections and are routinely used in clinical practice. With recent advances in deep learning, computational WSIs have garnered significant attention for their potential in automated and objective diagnosis, prognosis, and therapeutic-response prediction in the medical imaging community [1], [2], [3], [4], [5]. However, the large size of WSIs and the requirement for detailed pixel-level annotations present significant challenges in designing computational architectures [6], [7], [8]. To address this issue, Multiple Instance Learning (MIL)-based approaches have been proposed for weakly-supervised WSI analysis [2], [9], [10]. These approaches divide each WSI into patches, analyze each individually, and then aggregate the results for slide-level prediction.

Numerous research efforts have focused on efficiently aggregating patch-level information for WSI analysis [2], [11], [12]. Particularly, some methods involve adopting Graph Neural Networks (GNN) [13], [14], [15] or utilizing Transformer [16], [17] to model WSIs. Although these methods have achieved remarkable success in deriving expressive WSI representations, their reliance on fully connected graphs and self-attention mechanisms incurs a quadratic complexity relative to bag size. This leads to substantial computational overhead, particularly as a WSI bag typically comprises tens of thousands of patch instances, which results in a complexity of $O ( N ^ { 2 } )$ . To mitigate this complexity, GNN and Transformer-based MIL methods often adopt the bag-based strategy [13], [16], which, however, limits their capacity to directly model the global features of WSI from each patch instance due to restricted global receptive fields.

Moreover, WSIs are commonly stored at several levels with various resolutions, resulting in a hierarchical structure containing different pathological information [18]. For example, patch-level images encompass find-grained cells and tumor cellularity information [19], [20], [21], while region-level images mainly characterize the tissue information, such as the extent of tumor-immune localization [16], [22], [23]. The utilization of this hierarchical information poses challenges to model design. Furthermore, the various characteristics of feature distribution across WSIs can potentially result in mismatches between the training and testing data distributions, which may limit the model’s predictive accuracy.

To address the above challenges, we aim to design a novel WSI analysis model that efficiently models the relationships among patches from different regions and scales while maintaining alignment between training and testing WSI data to enhance model predictions. Inspired by recent advances in state space models [24], [25], we propose a new Mambabased framework, Pathology Mamba (PAM), for effective WSI analysis. First, we design a Local-aware Scanning (LAS) module to transform patches into an input sequence, enhancing local information capture through a local-aware scanning mechanism and enabling both flat and hierarchical WSI modeling. To model the relationship between each instance of the aforementioned sequence, we introduce a novel Bi-directional Mamba Encoder (BiME) for direct and efficient WSI analysis, which can model the sequence in a scanning manner and facilitate interactions between elements through a condensed hidden state. Furthermore, to align the feature distribution of testing and training data, we develop a novel Test-time Importance Resampling (TIR) module, which resamples patches of testing data to match the desired target feature distribution of training data. To evaluate the effectiveness of PAM, we conduct extensive experiments on nine slide-level analysis tasks, including cancer subtyping and survival prediction. The experiment results reveal that PAM outperforms current state-of-the-art methods across all tasks. Extensive ablation analysis confirm the efficacy of each module in PAM. And the visualization of heatmaps also demonstrates the enhanced capability of PAM in modeling discriminative areas within WSIs.

## II. RELATED WORK

## A. Multiple Instance Learning for WSI Analysis

Deep learning-based MIL approaches have been extensively researched in WSI analysis. Recently, attention-based aggregators and bag-based methods have demonstrated significant potential in enhancing MIL. Attention-based methods [2], [26] usually utilize the patch features to calculate the corresponding attention scores as the weight for each patch to get the global feature and the prediction results. For instance, Ilse et al. [26] proposed an attention-based MIL model, grounded in the contribution of each instance to the bag embedding. Lu et al. [2] showed that only a global pooling operator needs to be trained for weakly-supervised WSI classification by using ResNet-50 for instance-level feature extraction. However, these methods usually fail to model the relationships among patches. Bagbased methods [11], [12] usually split patches into several sub-groups, model the information of instances and get the representation feature for each bag, and then aggregate these representations to obtain the global feature and the final prediction results. While these methods can model relationships within each bag, they lack the direct capability to capture the relevance between patches across different bags.

## B. GNN and Transformer-Based WSI Analysis

The widespread adoption of GNN and Transformer in computer vision has led to their incorporation into many

MIL methods for WSI modeling. Considering the pyramidal structures of WSIs, Li et al. [27] proposed a multi-scale MIL model that patches at two distinct scales and concatenates features pyramidally. Hou et al. [13] proposed a hierarchical graph neural network to model the pyramid structure of WSI. Chen et al. [16] proposed a Transformer [28] based hierarchical network to leverage the natural hierarchical structure in WSI. Recently, Huang et al. [29] devised a hierarchical interaction Transformer to model the bi-directional feature interaction across scales, enhancing WSI analysis performance. However, the nature of the fully connected graph and self-attention mechanism requires quadratic complexity in terms of the number of instances. To avoid this issue, these methods usually follow the idea of the bag-based method, which restricts their ability to model the global feature of WSI from each instance directly.

## C. State Space Models

State Space Models (SSM) have been widely utilized across various research fields, including computational neuroscience and control theory. Structured state space sequence models (S4) [30], [31] are a promising class of architectures for sequence modeling in deep learning which can address the computational inefficiency of Transformers on long sequences with linear or near-linear scaling in sequence length. Recently, SSM architectures have achieved promising performance in widespread domains such as audio [32], [33] and computer vision [34], including H3 [25], Hyena [35], RetNet [36], and RWKV [37]. Considering that each WSI can be segmented into a lengthy sequence of patches, incorporating SSMs within the MIL framework for WSIs is a logical step. Fillioux et al. [38] proposed the use of SSMs as a multiple instance learner to model the information of digital pathology. However, this work does not take the importance of scanning direction and strategy into account, as well as the feature mismatch between training and testing.

## III. METHOD

The framework of the proposed Pathology Mamba (PAM) is demonstrated in Fig. 1 (a). It comprises three main components: Local-aware Scanning (LAS) module scans WSI patches using a flat or hierarchical strategy and incorporates a local-aware mechanism to form them into an input sequence; Bi-directional Mamba Encoder (BiME) models the WSI patch sequence effectively by incorporating a selective state space model, aiding downstream tasks like cancer subtyping and survival prediction; Test-time Importance Resampling (TIR) module resamples the input sequence based on importance weights, which improves the model’s generalization and mitigating the feature distribution gap between the training and testing phases.

## A. Preliminaries

1) State Space Models: State space model (SSM) represent a promising approach for capturing dependencies in long-range sequence modeling. Using a continuous function x(t) as input,

![](images/6d9d540c74620dd9d8717c727287997f555ec861321abfc222a0d05a0c1475ca.jpg)  
Fig. 1. The framework of proposed Pathology Mamba (PAM). The scale of regions and patches has been intentionally enlarged for clarity. In the experiments, resolutions of regions and patches are set to 4096 × 4096 and 512 × 512, respectively. (a) The architecture and comprehensive workflow of the proposed PAM. The local-aware scanning strategy can be flat scanning or hierarchical scanning. (b) The details of Bi-directional Mamba Encoder. (c) Depiction of S6 block. (d) The illustration of the Test-time Importance Resampling module.

these models facilitate sequence-to-sequence transformations in two stages, utilizing four key parameters 1, A, B, C

$$
\begin{array} { r } { h ^ { \prime } ( t ) = A h ( t ) + B x ( t ) , } \\ { y ( t ) = C h ( t ) + D x ( t ) , } \end{array}\tag{1}
$$

where h(t) denotes the hidden state, and the parameter D can be omitted for exposition since Dx can be viewed as a skip connection. Furthermore, the SSM is discretized by step size 1 to be applied on the discrete input sequence $[ { \pmb x } _ { 0 } , { \pmb x } _ { 1 } , \ldots ]$ instead of continuous function x(t). Through the application of the discretization rule, continuous parameters A, B, C are converted into discrete counterparts A, B, C

$$
{ \overline { { A } } } = e ^ { \Delta A } , ~ { \overline { { B } } } = ( \Delta A ) ^ { - 1 } ( e ^ { \Delta A } - I ) \cdot \Delta B , ~ { \overline { { C } } } = C .\tag{2}
$$

The linear recurrence mode of the SSM can be represented as

$$
\begin{array} { l } { { \pmb { h } } _ { t } = \overline { { { \pmb { A } } } } { \pmb { h } } _ { t - 1 } + \overline { { { \pmb { B } } } } { \pmb { x } } _ { t } , } \\ { { \pmb { y } } _ { t } = { \pmb { C } } { \pmb { h } } _ { t } , } \end{array}\tag{3}
$$

where $\scriptstyle { x _ { t } }$ and $\mathbf { } y _ { t }$ represent the input sequence and output sequence, respectively, and $\mathbf { } _ { \pmb { h } _ { t } }$ represents the hidden state. To improve computational efficiency, the recurrent SSM can be converted to a global convolution mode by unrolling

$$
\begin{array} { r } { \overline { { K } } = ( C \overline { { B } } , C \overline { { A } } \overline { { B } } , \ldots , C \overline { { A } } ^ { k } \overline { { B } } , \ldots ) , } \\ { y = x * \overline { { K } } , \qquad } \end{array}\tag{4}
$$

where $\overline { { K } }$ represents the SSM convolution kernal.

Based on the basic SSM, the Structured State Space Model (S4) [30] incorporates HiPPO matrix to overcome the exploding gradients problem and addresses computational challenges by introducing a special representation and algorithm under the Diagonal Plus Low-Rank (DPLR) assumption.

2) Selective State Space Models: By incorporating a selection mechanism into S4 models to control information propagation or interaction along the sequence dimension and making their parameters that affect interactions along the sequence input-dependent, Gu and Dao [24] proposed selective state space (S6) models. Additionally, their study introduced a hardware-aware algorithm that leverages the memory hierarchy of modern hardware to address computational inefficiency challenges.

In S6, the parameters 1, B, C are functions of the input, as shown in Fig. 1 (c), which suggests that S6 recognizes the context information contained in the input and ensures the adaptability of weights in S6. Furthermore, they also proposed Mamba by integrating selective SSMs into a simplified endto-end neural network without attention, which demonstrates near-linear computational complexity and achieves performance on par with the Transformer architecture.

## B. Local-Aware Scanning

Given a WSI, the foreground tissue of the original WSI is initially segmented into M non-overlapping regions with 4096 × 4096 pixels in size. Subsequently, each region is subdivided into 64 patches, each measuring 512 × 512 pixels.

![](images/48315abdd411b5983b80494708a35523e5ca25d52ae9117efce018173b3a8baa.jpg)  
Fig. 2. Illustration of different scanning strategies for WSI analysis.

A pretrained feature extractor is then employed to generate a feature representation vector for each region and patch,

$$
\mathcal { R } = \{ R _ { 0 } , R _ { 1 } , \ldots , R _ { N - 1 } \} \in \mathbb { R } ^ { M \times E } .\tag{5}
$$

Each region feature vector $\pmb { R } _ { i } \in \mathbb { R } ^ { 1 \times E }$ corresponds a patch feature sequence,

$$
\mathcal { P } _ { i } = \{ P _ { 0 } ^ { i } , P _ { 1 } ^ { i } , \ldots , P _ { 6 3 } ^ { i } \} \in \mathbb { R } ^ { 6 4 \times E } ,\tag{6}
$$

where $P _ { \mathrm { ~ \it ~ i ~ } } ^ { i } \in \mathbb { R } ^ { 1 \times E }$ represents the feature vector of j -th patch from i-th region, and E denotes the embedding dimension.

To acquire the feature sequence in a format conducive to the state space model’s processing and analysis requirements, a specific scanning strategy is required to generate the input sequences for the state space model after obtaining sequences of featured regions and their corresponding featured patches. The vanilla scanning processes the input patches line by line according to their spatial positions, as shown in Fig. 2 (a). However, this vanilla scanning fails to capture the local organization within the WSI at the patch level since some patches with close spatial positions may be far apart after scanning.

To alleviate this issue and ensure that the spatial and sequential information inherent to the regions and patches is effectively captured and represented, we introduce a new Local-aware Scanning (LAS) module that comprises local-aware flat scanning and local-aware hierarchical scanning. The local-aware flat scanning is demonstrated in Fig. 2 (b). Initially, all patches from the first region are sequentially processed using vanilla scanning. Subsequently, patches from the remaining regions are scanned in sequence, ensuring a structured approach to processing spatial and local information across WSI regions. The scanned patches are then formulated to an input sequence: $I _ { p _ { \ldots } = \{ P _ { 0 } ^ { 0 } , P _ { 1 } ^ { 0 } , \ldots , P _ { 6 3 } ^ { 0 } , \ldots , P _ { 0 } ^ { M - 1 } , P _ { 1 } ^ { M - 1 } , \ldots , P _ { 6 3 } ^ { M - 1 } \} }$ ∈ R(M·64)×E .

To further leverage the inherent hierarchical structure within WSIs, we also design a local-aware hierarchical scanning strategy to effectively capture WSIs’ multi-scale structural information, ranging from broad architectural patterns to fine cellular details. This strategy provides a rich and nuanced input for analysis while preserving the local information modeling capability of local-aware flat scanning. It is similar to the Breadth First Scan (BFS) method, which offers a straightforward approach to navigating the hierarchical structure of WSIs, as shown in Fig. 2 (c). Specifically, it initially scans the features at the region level and arranges all featured regions into a sequence to comprehensively model macroscopic-level features. After that, it processes featured patches at a more microscopic level using local-aware flat scanning and then arranges patches according to their regions’ scanning order. This process produces a feature sequence that incorporates both region and patch information, serving as the input: $I _ { h } \ =$ $\{ \check { R } _ { 0 } , \dotsc , \check { R } _ { M - 1 } ^ { \mathrm { ~ ~ } } , \pmb { P } _ { 0 } ^ { 0 } , \dotsc , \pmb { P } _ { 6 3 } ^ { 0 } , \dotsc , \pmb { P } _ { 0 } ^ { \check { M } - 1 } , \dotsc , \pmb { P } _ { 6 3 } ^ { \check { M } - 1 } \} \quad \in$ R(M+M·64)×E .

## C. Bi-Directional Mamba Encoder

Following the LAS module, we obtain a sequence of formulated feature vectors. We then propose a Bi-directional Mamba Encoder (BiME) to model the sequence directly and effectively, as demonstrated in Fig. 1 (b). Specifically, this encoder incorporates a layer normalization [39], a linear layer, and a 1D convolution layer with subsequent SiLU activation [40] to generate the input matrix x,

$$
\begin{array} { r } { \pmb { x } = \mathrm { S i L U } ( \mathrm { C o n v } ( \operatorname { L i n e a r } ( \operatorname { L N } ( I ) ) ) ) \in \mathbb { R } ^ { M \times E } . } \end{array}\tag{7}
$$

Additionally, a separate linear layer followed by the SiLU activation is utilized to generate the gate sequence z,

$$
z = \mathrm { S i L U } ( \mathrm { L i n e a r } ( \mathrm { L N } ( I ) ) ) \in \mathbb { R } ^ { M \times E } .\tag{8}
$$

Subsequently, the S6 block is utilized to model the sequential information of the input matrix x. To model the relationship within the sequence more comprehensively, we further introduce a bi-directional scanning process including both forward and backward scans to enhance the modeling capability. Different from the approach in Vmamba [41] that uses separate S6 blocks for forward and backward processing, our proposed BiME employs a shared-parameter strategy for both directional scans, which acts as an effective augmentation technique to enhance the model’s generalization capabilities and predictive accuracy due to the non-directional nature of WSIs.

Following this, we obtain the forward scan feature sequence matrix $y _ { f } .$ . The details of the S6 block are illustrated in Fig. 1(c). First, the trainable parameter matrices $\pmb { A } \in \mathbb { R } ^ { E \times S }$ and $\textbf { \textit { D } } \in \ \mathbb { R } ^ { E }$ are initialized, where S denotes the state dimension. Then, two linear layers are applied to the input matrix x, yielding matrices $\pmb { { B } } ^ { \mathrm { ~ \bar { ~ } ~ } } \in \mathbb { R } ^ { M \times S }$ and $\textbf { \textit { C } } \in \mathbb { R } ^ { M \times S }$ , respectively. Additionally, two more linear layers are applied to x, resulting in the matrix $\pmb { \Delta } \in \mathbb { R } ^ { M \times E }$ . Besides, the discrete parameters A and B are obtained by applying Equ (2). Finally, the output sequence matrix is obtained using the SSM

$$
\begin{array} { r } { \pmb { y } _ { f } = \mathsf { S 6 } ( \pmb { x } ) = \mathsf { S S M } ( \overline { { \pmb { A } } } , \overline { { \pmb { B } } } , \pmb { C } ) ( \pmb { x } ) \in \mathbb { R } ^ { M \times E } . } \end{array}\tag{9}
$$

Similar to the forward process, we can also get the backward scan feature sequence matrix $y _ { b }$ by inputting the reversed input sequence x to the same S6 block. And $y _ { b }$ is then reversed to combine with $y _ { f }$ to obtain the bi-directional scan sequence matrix y,

$$
\begin{array} { r } { \pmb { y } _ { b } = \mathsf { S 6 } ( \mathbf { R e v e r s e } ( \pmb { x } ) ) \in \mathbb { R } ^ { M \times E } , } \\ { \pmb { y } = \pmb { y } _ { f } + \mathrm { r e v e r s e } ( \pmb { y } _ { b } ) \in \mathbb { R } ^ { M \times E } . } \end{array}\tag{10}
$$

The resulting sequence matrix y is then element-wise multiplied by the gate sequence matrix z, and a subsequent linear layer is applied to produce the final output feature sequence O,

$$
\mathcal { O } = \mathrm { L i n e a r } ( y \odot z ) \in \mathbb { R } ^ { M \times E } .\tag{11}
$$

## D. Test-Time Importance Resampling

While the BiME module can effectively model WSIs, it also encounters challenges with mismatched feature distributions between training and testing data, which may hinder the model’s capability to perform prediction in the testing phase. Moreover, as WSIs are segmented into many patches, features across these patches often show considerable similarity and redundancy. Consequently, we employ a patch sampling approach during both training and testing to downsample the original sequence of patches and align the feature distributions of training and testing data. Specifically, in diagnostic WSI, cancerous regions typically occupy significant areas. Thus, sampling a fixed number of patches (in this study, $k = 4 0 0 0 )$ ensures a certain proportion of patches contain cancerous cells. However, due to variability among WSIs, the proportion of different cell types within a testing WSI may significantly differ from other training WSIs, which may potentially position it as an outlier in the feature space. This discrepancy could negatively impact the model’s predictive performance on that specific WSI.

Therefore, we focus on leveraging prior knowledge from the training data to align the feature distribution of the testing data with that of the training data, thus enhancing the testing stability. To this end, we propose a Test-time Importance Resampling (TIR) strategy based on the Sampling-Importance Resampling algorithm [42], [43]. Specifically, we denote $\pmb { p } _ { f e a t }$ and $\textbf { \textit { q } } _ { f e a t }$ as the feature distributions of training and testing WSIs, respectively, and the goal is to select patches from the testing WSI with features that are approximately distributed according to the training feature distribution.

As illustrated in Fig. 1 (d), distinct strategies are employed for the training and testing phases. During the training phase, a random sampling approach is adopted to select k patches from each WSI, which not only reduces computational load but also facilitates data augmentation. Additionally, we maintain and update a record of the feature distribution $\hat { p } _ { f e a t }$ of the training data, which can be regarded as the mean of the feature vector for each patch in the training set. During the testing phase, for each slide, we first obtain the mean of the feature vector for each patch in the testing set to represent the feature distribution $\hat { \pmb q } _ { f e a t }$ of the testing slide. Next, for feature vector $\mathbf { \nabla } P _ { i }$ of patch i, we calculate the cosine similarity between this feature vector and $\hat { p } _ { f e a t }$ to obtain the likelihood of this feature belonging to the distribution of training set $\hat { \pmb { p } } _ { f e a t } ( \pmb { P } _ { i } )$ . Similarly, we can determine the likelihood of this feature belonging to the distribution of testing set $\hat { \pmb q } _ { f e a t } ( \pmb { P } _ { i } )$ . Particularly, the importance weights $w _ { i }$ for each featured patch $\mathbf { \nabla } P _ { i }$ are computed as

$$
w _ { i } = \frac { \hat { \pmb { p } } _ { f e a t } ( \pmb { P } _ { i } ) } { \hat { \pmb q } _ { f e a t } ( \pmb { P } _ { i } ) } .\tag{12}
$$

Subsequently, k featured patches are sampled without replacement from the testing WSI, based on the following probabilities:

$$
P r o b _ { i } = \frac { w _ { i } } { \sum _ { j = 1 } ^ { M } w _ { j } } .\tag{13}
$$

Selecting the top-k patches based solely on the highest probabilities is sub-optimal, as it favors patches with high probabilities and neglects potentially relevant patches with lower probabilities that may align with the target training feature distribution. To address this limitation, we utilize the Gumbel top-k procedure [44], [45] to introduce randomness into the sampling process by adding Gumbel noise to the probabilities, which ensures that patches with lower probabilities also have a chance to be sampled. This strategy balances the trade-off between exploiting the most informative patches and exploring potentially informative ones with lower initial probabilities. Specifically, in the Gumbel top-k procedure, noise $g _ { i }$ is sampled from the IID standard Gumbel distribution and added to each log-importance weight to determine the score for each patch,

$$
s _ { i } = \log { w _ { i } } + g _ { i } = \log { \frac { \hat { p } _ { f e a t } ( \pmb { P } _ { i } ) } { \hat { \pmb { q } } _ { f e a t } ( \pmb { P } _ { i } ) } } + g _ { i } .\tag{14}
$$

After that, patches with the top k scores are selected and reordered according to the original scanning sequence.

## E. Objective Function

The BiME module outputs a processed feature sequence O, which is then aggregated (using max pooling in this study) to obtain a global feature vector. This vector is further passed into an MLP to generate the final prediction. To evaluate the performance of PAM, we conducted experiments on cancer subtyping tasks and survival prediction tasks, and we chose different objective functions for these two tasks. For cancer subtyping tasks, the cross entropy (CE) loss is adopted. While for survival prediction tasks, the negative log-likelihood (NLL) loss [49] is adopted, in line with prior studies [16], [50].

## IV. EXPERIMENTS

## A. Experimental Settings

To fully investigate the performance of the proposed PAM, we experiment on four slide-level classification tasks and five survival prediction tasks across different organ types.

1) Comparisons: We conduct an extensive comparison of our approach against a range of leading Whole Slide Image (WSI) analysis approaches to comprehensively demonstrate the advantages of the proposed PAM for WSI analysis. The methods compared include four traditional MIL methods (CLAM [2], DS-MIL [27], HIPT [16], and DTFD-MIL [11]), two Transformer-based MIL methods (TransMIL [3] and HIT [29]), two GNN-based MIL methods (Patch-GCN [46] and WiKG [47]) and two SSM-based MIL methods (S4-Model [38] and MambaMIL [48]). Since the original HIPT has pretrained parameters, we adopt the HIPT\_N in the experiment, which is the simplified version of HIPT. We followed the reproduction strategies and hyperparameter settings used in the original papers to reproduce the competing models, with the exception of the input dimension, which was adjusted to 768 to match the feature dimension of the preprocessed datasets.

2) Datasets of Slide-Level Classification Tasks: The slidelevel classification tasks include cancer subtyping and lymph node metastases detection task. The cancer subtyping tasks were conducted on four well-known WSI datasets from The Cancer Genome Atlas (TCGA) project: Invasive Breast Carcinoma (BRCA), Esophageal Carcinoma (ESCA), Non-Small Cell Lung Cancer (NSCLC), and Renal Cell Carcinoma (RCC). The lymph node metastases detection task was conducted on the CAMELYON16 [51] dataset. Detailed descriptions of these datasets are provided below:

1) BRCA includes nine disease subtypes. This study focuses on the two most prevalent subtypes: Invasive Ductal Carcinoma (IDC, 726 slides from 694 cases) and Invasive Lobular Carcinoma (ILC, 149 slides from 143 cases), resulting in a total of 875 diagnostic WSIs.

2) ESCA consists of two disease subtypes: Adenocarcinoma (AC, 66 slides from 65 cases) and Squamous Cell Carcinoma (SCC, 90 slides from 89 cases), with a total of 156 diagnostic WSIs.

3) NSCLC includes two disease subtypes: Lung Adenocarcinoma (LUAD, 492 slides from 430 cases) and Lung Squamous Cell Carcinoma (LUSC, 466 slides from 432 cases), totaling 958 diagnostic WSIs.

4) RCC comprises three disease subtypes: Kidney Chromophobe Renal Cell Carcinoma (CHRCC, 118 slides from 107 cases), Kidney Clear Cell Renal Cell Carcinoma (CCRCC, 498 slides from 492 cases), and Kidney Papillary Renal Cell Carcinoma (PRCC, 289 slides from 267 cases), amounting to 905 diagnostic WSIs.

5) CAMELYON16 is a dataset for the detection of lymph node metastases in women with breast cancer. It consists of 399 WSIs, with 159 slides containing nodal metastases and 240 slides without metastases.

3) Datasets of Survival Analysis: The survival prediction tasks were conducted on five WSI datasets from the TCGA project. Slides with available prognostic data were selected and only one slide was chosen from each case. Detailed descriptions are provided below:

1) BRCA is the same dataset used in cancer subtyping, including a total of 697 slides.

2) LUAD is a subset of the NSCLC dataset and this collection includes 430 slides.

3) CCRCC is a subset of the RCC dataset, comprising 492 slides in total.

4) PRCC is another subset of the RCC dataset, including 267 slides.

5) BLCA consists of slides diagnosed with Bladder Carcinoma, totaling 383 slides.

## B. Implementation Details

1) Feature Extraction: To obtain region images and their corresponding patch images, this study commences by segmenting the foreground tissue, as introduced in [2]. Subsequently, tiles of 4096 × 4096 pixels are extracted from the segmented areas at 20× magnification to serve as regionlevel images, with the background discarded. Each region-level image is then subdivided into 64 non-overlapping 512 × 512 pixel patches to create patch-level images. ConvNeXt [52] pretrained on ImageNet is employed as the feature extractor to generate 768-dimensional feature vectors for both region and patch levels. For fair performance comparison, the same pre-extracted feature vectors are used across all evaluated methods.

2) Training and Evaluation: Model parameters are optimized using the Adam optimizer [53] with a batch size of one. A learning rate of $1 \times 1 0 ^ { - 5 }$ is set, with linear decay and early stopping implemented to prevent overfitting. To demonstrate PAM’s superiority, extensive experiments were conducted on slide-level classification and survival prediction tasks. Slidelevel classification performance is assessed using accuracy (ACC) and the area under the curve (AUC) of the receiver operating characteristic. Survival prediction performance is primarily assessed using the concordance index (c-Index). Results with means and standard deviations are obtained from 10 repeated runs of 10-fold cross-validation.

## C. Results of Slide-Level Classification

The comparison results for classification tasks are presented in Table I. Overall, PAM consistently outperforms other methods in AUC metric across all tasks. Compared to the best-performing baseline in each task, PAM shows performance increases of 3.85%, 2.50%, 0.83%, 0.30%, and 0.86% in AUC for the BRCA, ESCA, NSCLC, RCC, and CAMELYON16 datasets, respectively.

When comparing SSM-based methods (S4-Model and MambaMIL) with Transformer-based and GNN-based methods, SSM-based methods generally exhibit superior performance across most tasks. This superiority is attributed to the SSM’s capability to model long-range dependencies between patches and the hierarchical information modeling of WSIs, which is essential for accurate WSI analysis. Conversely, the use of Nyström self-attention to approximate standard self-attention in TransMIL and the bag-based design of HIT hinder their ability to effectively model WSIs. Additionally, the focus of GNN-based methods on modeling the local structure of WSIs using graphs limits their overall performance. As PAM is also an SSM-based method, it outperforms both the S4-Model and MambaMIL in all tasks, primarily due to the local-aware scanning and hierarchical modeling of the LAS module, as well as the importance resampling of the TIR module.

## D. Ablation Analysis

1) Ablation Study: To demonstrate the effectiveness of Local-aware Scanning (LAS) module, incorporating hierarchical information, and Test-time Importance Resampling (TIR) module, we conducted an ablation study and present the results in the Table II. It was observed that PAM’s performance (AUC) decreased across all tasks when vanilla scanning was performed (w/o LAS), hierarchical information was ignored (w/o Hie), all patches were inputted without resampling (w/o TIR), or the BiME module was replaced with a single-directional Mamba encoder (w/o BiME). The observed decrease when ignoring hierarchical information underscores the effectiveness of leveraging WSIs’ inherent multi-scale structure, enabling the model to discern and utilize both macroscopic and microscopic patterns for accurate cancer subtyping. The improvement observed with the use of TIR can be attributed to two main factors. First, the random sampling process during training serves as data augmentation to enhance the model’s generalizability. Second, the importance resampling during testing based on feature distribution also ensures that the sampled patches’ feature distribution aligns with that of the training data, thereby improving the model’s predictive stability. This combination of techniques optimizes model performance by leveraging the training data’s diversity and the consistency in feature distribution between the training and testing phases. The performance gain from using LAS can be attributed to its ability to enable the model to focus on overarching patterns across the entire WSI and to identify and leverage specific, localized features critical for accurate analysis and predictions. Additionally, the effectiveness of the bi-directional design in BiME is also demonstrated, as it enables the model to capture information from both directions.

TABLE I  
RESULTS OF DIFFERENT METHODS ON SLIDE-LEVEL CLASSIFICATION TASKS
<table><tr><td rowspan="2">Method</td><td colspan="2">BRCA</td><td colspan="2">ESCA</td><td colspan="2">NSCLC</td><td colspan="2">RCC</td><td colspan="2">CAMELYON16</td></tr><tr><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td></tr><tr><td>CLAM [2]</td><td> $0 . 8 4 5 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $0 . 8 4 3 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 9 4 4 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 9 3 9 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 9 8 6 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 1 8 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 8 8 5 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 8 6 3 { \scriptstyle \pm 0 . 0 4 6 }$ </td></tr><tr><td>DS-MIL [27]</td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 8 5 9 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 1 0 1 }$ </td><td> $\overline { { 0 . 7 8 0 { \pm 0 . 1 2 4 } } }$ </td><td> $0 . 9 3 7 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 8 7 1 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 8 9 7 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 8 5 9 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 8 1 3 { \scriptstyle \pm 0 . 0 5 4 }$ </td></tr><tr><td>HIPT [16]</td><td> $0 . 8 7 2 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 8 6 7 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 9 1 7 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td> $0 . 7 8 7 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $0 . 9 5 1 \pm 0 . 0 2 1$ </td><td> $0 . 8 8 2 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 9 8 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 1 2 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 9 0 9 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 8 3 { \scriptstyle \pm 0 . 0 4 0 }$ </td></tr><tr><td>DTFD-MIL [11]</td><td> $0 . 8 8 1 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 8 6 5 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 9 2 0 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 1 1 7 }$ </td><td> $0 . 9 4 6 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 8 8 2 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 9 8 6 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 2 4 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 9 1 7 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 8 5 7 { \scriptstyle \pm 0 . 0 4 4 }$ </td></tr><tr><td>TransMIL [3] a</td><td> $0 . 8 4 9 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 8 4 0 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 9 2 0 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 7 8 7 { \scriptstyle \pm 0 . 1 1 7 }$ </td><td> $0 . 9 5 0 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 8 7 9 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 9 2 7 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 6 8 }$ </td></tr><tr><td>HIT [29] a</td><td> $\underline { { 0 . 8 8 3 \pm 0 . 0 6 7 } }$ </td><td> $0 . 8 7 1 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 9 5 0 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 8 4 0 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $0 . 9 5 0 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 8 7 2 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $\underline { { 0 . 9 9 1 \pm 0 . 0 0 5 } }$ </td><td> $\underline { { 0 . 9 4 0 { \scriptstyle \pm 0 . 0 2 0 } } }$ </td><td> $0 . 8 6 1 \pm 0 . 0 3 7$ </td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 4 0 }$ </td></tr><tr><td>Patch-GCN [46] b</td><td> $0 . 8 5 2 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 8 4 6 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 9 0 9 { \scriptstyle \pm 0 . 0 8 2 }$ </td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 7 7 }$ </td><td> $0 . 9 4 9 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 8 8 2 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 1 8 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 8 9 2 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 0 3 5 }$ </td></tr><tr><td>WiKG [47] b</td><td> $0 . 8 4 1 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.957±0.048</td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td>0.944±0.037</td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $0 . 9 8 6 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 2 1 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $\underline { { 0 . 9 2 1 \pm 0 . 0 3 2 } }$ </td><td> $0 . 8 5 2 { \scriptstyle \pm 0 . 0 3 9 }$ </td></tr><tr><td>S4-Model [38] c</td><td> $0 . 8 6 9 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $\underline { { 0 . 8 7 3 \pm 0 . 0 2 5 } }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $0 . 7 8 7 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $0 . 9 5 5 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 8 8 3 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $\underline { { 0 . 9 9 1 \pm 0 . 0 0 4 } }$ </td><td> $0 . 9 3 8 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 9 1 0 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 8 6 0 { \scriptstyle \pm 0 . 0 6 1 }$ </td></tr><tr><td>MambaMIL [48] c</td><td> $0 . 8 6 6 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 8 5 2 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $\underline { { 0 . 9 5 9 \pm 0 . 0 3 9 } }$ </td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 7 8 }$ </td><td> $\underline { { 0 . 9 5 6 \pm 0 . 0 2 2 } }$ </td><td> $\underline { { 0 . 8 8 5 \pm 0 . 0 4 9 } }$ </td><td> $0 . 9 8 8 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 1 9 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 9 0 3 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 0 5 0 }$ </td></tr><tr><td>PAM (Ours)</td><td> $\mathbf { 0 . 9 1 7 { \scriptstyle \pm 0 . 0 3 9 } }$ </td><td> $\mathbf { 0 . 8 8 0 \pm 0 . 0 3 2 }$ </td><td> $\mathbf { 0 . 9 8 3 \pm 0 . 0 3 2 }$ </td><td> $\mathbf { 0 . 9 2 7 { \scriptstyle \pm 0 . 0 6 0 } }$ </td><td> $\mathbf { 0 . 9 6 4 } \pm \mathbf { 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 9 0 9 { \scriptstyle \pm 0 . 0 4 6 } }$ </td><td> $\mathbf { 0 . 9 9 4 } \pm \mathbf { 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 9 5 2 } \pm \mathbf { 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 9 2 9 } \pm \mathbf { 0 . 0 3 8 }$ </td><td> $\mathbf { 0 . 8 8 5 \pm 0 . 0 2 4 }$ </td></tr></table>

We highlight the best result in bold and the second-best result with an underline. The sample size of TIR is $k = 4 0 0 0 .$ $/ ^ { \mathrm { ~ b ~ } } / ^ { \mathrm { ~ c ~ } }$ represent Transformer / Graph / SSM based methods respectively.

TABLE II  
RESULTS OF ABLATION STUDY. THE AUC PERFORMANCE IS REPORTED  
TABLE III  
RESULTS OF ABLATION STUDY. THE AUC PERFORMANCE IS REPORTED
<table><tr><td>Method</td><td>BRCA</td><td></td><td>ESCA</td><td>NSCLC</td><td>RCC</td><td></td></tr><tr><td>PAM</td><td>0.917</td><td></td><td>0.983</td><td>0.964</td><td>0.994</td><td></td></tr><tr><td>- w/o LAS</td><td>0.899</td><td>↓1.96%</td><td>0.968↓1.53%</td><td>0.959</td><td>↓0.52%</td><td>0.991↓0.30%</td></tr><tr><td>- w/o Hie</td><td>0.912</td><td>↓0.55%</td><td>0.977 ↓0.61%</td><td>0.960↓0.41%</td><td>0.994 -</td><td></td></tr><tr><td>- w/o TIR</td><td></td><td></td><td>0.903 ↓1.53% 0.976 ↓0.71% 0.960 ↓0.41%</td><td></td><td></td><td>0.992↓0.20%</td></tr><tr><td>- w/o BiME</td><td>0.904</td><td>↓1.42%</td><td>0.980 ↓0.31%</td><td>0.961</td><td>↓0.31% 0.993</td><td>↓0.10%</td></tr></table>

<table><tr><td>Method</td><td>BRCA</td><td>ESCA</td><td>NSCLC</td><td>RCC</td></tr><tr><td>Original LAS</td><td> $0 . 9 1 7 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 9 8 3 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td>0.964±0.019</td><td> $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>w/ Rotation 90°</td><td>0.914±0.043</td><td> $0 . 9 8 3 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 9 6 3 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td rowspan="2">w/ Rotation 180°</td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td></tr><tr><td> $0 . 9 1 6 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 9 8 1 \pm 0 . 0 2 2$ </td><td> $0 . 9 6 5 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td>0.994±0.004</td></tr><tr><td rowspan="2">w/ Rotation 270°</td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td></tr><tr><td>0.914±0.031</td><td> $0 . 9 8 5 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td></td><td>0.962±0.026 0.994±0.005</td></tr><tr><td rowspan="2">w/ Flip</td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$   $0 . 9 8 5 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td></tr><tr><td> $0 . 9 1 6 { \scriptstyle \pm 0 . 0 2 4 }$   $( p > 0 . 0 5 )$ </td><td> $( p > 0 . 0 5 )$ </td><td> $0 . 9 6 3 { \scriptstyle \pm 0 . 0 2 6 }$   $( p > 0 . 0 5 )$ </td><td> $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 5 }$   $( p > 0 . 0 5 )$ </td></tr></table>

TABLE IV  
FURTHER ANALYSIS OF TIR MODULE. THE AUC PERFORMANCE IS REPORTED
<table><tr><td rowspan="2">Method</td><td colspan="4">BRCA</td></tr><tr><td> $k = 1 0 0 0$ </td><td> $k = 2 0 0 0$ </td><td> $k = 4 0 0 0$ </td><td>k = 6000</td></tr><tr><td>PAM -w/ TIR</td><td>0.909</td><td>0.911</td><td>0.912</td><td>0.908</td></tr><tr><td>PAM -w/ TRS</td><td>0.902 ↓0.77%</td><td>0.906 ↓0.55%</td><td>0.910 ↓0.22%</td><td>0.902 ↓0.66%</td></tr></table>

2) Further Analysis of LAS Module: To further investigate the robustness of the LAS module to rotation and flip transformations, we conducted additional experiments, and the results are presented in Table III. We applied rotation and flip operations to each WSI, conducted feature extraction, and performed local-aware scanning on the rotated or flipped WSIs. The results show that the LAS module is insensitive to rotation and flip transformations, as the performance changes are not statistically significant across all tasks $( i . e . , p > 0 . 0 5 )$ This insensitivity is attributed to the lack of orientation-based inductive bias within WSIs. Therefore, regardless of rotation or flipping, PAM can effectively model the features of WSIs.

3) Further Analysis of TIR Module: We conducted additional experiments to compare the performance of TIR and TRS, and the results are reported in Table IV. TRS denotes Test-time Random Sampling, which replaces importance resampling in TIR with random sampling. It is observed that TIR consistently outperforms TRS across different sample sizes (k), which demonstrates the superiority of utilizing importance resampling during the testing phase. Moreover, the effectiveness of TIR at various sampling sizes and the influence of integrating TIR with other models were also investigated to demonstrate the superiority of the TIR module. The results are depicted in Fig. 3. The mean number of patches in the BRCA dataset is around 12,700, and All denotes using all of the patches without TIR. Since most methods only input patch-level information, we implemented PAM with flat scanning and did not compare methods with hierarchical structures (i.e., HIPT and HIT) in this experiment for fair comparison. First, it is observed that incorporating TIR significantly enhances the performance of PAM across all sampling sizes. Furthermore, the integration of TIR with most methods leads to varying degrees of performance improvement. Notably, sampling sizes around 4,000 generally yield better performance.

![](images/0ef375df06332b607c8bf5f7991f32341e62e85bcfee6fd7113a62c510640f94.jpg)  
Fig. 3. The AUC performance of different models in the BRCA dataset at different sample sizes combining the TIR method.

Importance Weight High Weight Patches Low Weight Patches  
![](images/9e6a690692d3f9cb58fcf9147675a294cc8be3401d74f706f90167f19aa78e8d.jpg)  
Fig. 4. The visualization of importance weights and some samples with high or low weights.

We also visualized the importance weights of several WSIs with TIR in Fig. 4, particularly those with a large area of adipose cells. It is observed that different cell types possess varying importance weights, with high weights primarily assigned to cancer cells and normal cell tissues, whereas low weights are predominantly associated with adipose cells.

4) Key Parameter Settings in PAM: We also conduct experiments to investigate performance changes in PAM across various parameter settings and identify the optimal parameters. These experiments are conducted on the validation set, and results are reported on the test set to avoid data leakage. We primarily focus on two parameters: the embedding dimension E and the state dimension S. The results are presented in Fig. 5. For the embedding dimension, it is observed that as the dimension increases from 128 to 1024, the variation trend of the AUC is more stable compared to the ACC, showing an initial increase followed by a decrease. Notably, both AUC and ACC reach their peak when the dimension is set to 640. The influence of state dimension on PAM’s performance is less pronounced, with optimal performance achieved at a state dimension of 16. Consequently, in our experiments, the embedding dimension and state dimension were set to 640 and 16, respectively.

![](images/8cd211429174273ef67f1ec2c731b11e88aa848aa1b8b789cfe83ce2440424a6.jpg)  
Fig. 5. The impact of embedding dimension and state dimension to the performance of PAM for TCGA-BRCA subtyping task.

## E. Results of Survival Prediction

To comprehensively evaluate the effectiveness of PAM, we also conducted experiments on survival prediction tasks, and the results are presented in Table V. Overall, PAM achieves the highest C-Index scores in the BRCA, LUAD, PRCC and BLCA datasets. Specifically, it achieves a C-Index increase of 0.44%, 0.80%, 0.42%, 3.63% on BRCA, LUAD, CCRCC, and BLCA datasets, respectively. It is also noted that the S4-Model [38] achieves suboptimal performance in two cancer types. This shows the effectiveness of state space model-based methods in comprehensively modeling important long-range dependencies between WSI instances, which is advantageous for survival prediction. Moreover, the incorporation of an enhanced state space model in PAM further enhances its performance. Additionally, the local-aware scanning mechanism enables PAM to focus on crucial local information while retaining the capability to model WSI’s global information.

## F. Visualization

1) Comparing With CLAM: To demonstrate PAM’s effectiveness and superiority in modeling WSIs, heatmaps generated by PAM were visualized and compared with those produced by CLAM. We utilize all input patches to conduct this analysis without employing TIR. Specifically, to generate the heatmaps of PAM, the matrix $\pmb { \Delta } \in \bar { \mathbb { R } } ^ { M \times E }$ was averaged along its second dimension, yielding a one-dimensional vector of length M, representing the weight value for each patch. The matrix 1 controls the balance between how much to focus or ignore the current input, which is similar to the gates in

TABLE V  
RESULTS ON SURVIVAL PREDICTION TASKS. THE C-INDEX METRIC IS REPORTED
<table><tr><td>Method</td><td>BRCA</td><td>LUAD</td><td>CCRCC</td><td>PRCC</td><td>BLCA</td></tr><tr><td>CLAM [2]</td><td>0.665±0.039</td><td>0.566±0.069</td><td>0.682±0.057</td><td>0.622±0.039</td><td>0.633±0.077</td></tr><tr><td>DS-MIL [27]</td><td>0.625±0.029</td><td>0.601±0.068</td><td>0.655±0.057</td><td>0.567±0.061</td><td>0.634±0.065</td></tr><tr><td>HIPT [16]</td><td>0.650±0.040</td><td>0.560±0.066</td><td>0.671±0.081</td><td>0.563±0.067</td><td>0.595±0.085</td></tr><tr><td>TransMIL [3] a</td><td>0.660±0.053</td><td>0.581±0.083</td><td>0.700±0.073</td><td>0.574±0.058</td><td>0.609±0.077</td></tr><tr><td>HIT [29] a</td><td>0.656±0.033</td><td>0.581±0.067</td><td>0.684±0.058</td><td>0.572±0.057</td><td>0.617±0.085</td></tr><tr><td>Patch-GCN [46] </td><td>0.687±0.039</td><td>0.582±0.075</td><td>0.701±0.063</td><td>0.564±0.045</td><td>0.595±0.068</td></tr><tr><td>WiKG [47] b</td><td>0.680±0.030</td><td>0.589±0.081</td><td>0.703±0.056</td><td>0.568±0.052</td><td>0.568±0.052</td></tr><tr><td>S4-Model [38] c</td><td>0.677±0.038</td><td>0.623±0.074</td><td>0.715±0.056</td><td>0.582±0.062</td><td>0.591±0.075</td></tr><tr><td>MambaMIL [48]</td><td>c0.665±0.044</td><td>0.588±0.052</td><td>0.684±0.058</td><td>0.603±0.060</td><td>0.629±0.069</td></tr><tr><td>PAM</td><td>0.690±0.036</td><td>0.628±0.084</td><td>0.718±0.049</td><td>0.622±0.038</td><td>0.657±0.080</td></tr></table>

We highlight the best result in bold and the second-best result with an underline. The sample size of TIR is k = 4000.  
a  /  represent Transformer / Graph / SsM based methods respectively.

![](images/123ad2c89bf46fec22db1cfaa200877e8a2005c4cc0c512278cbf8d0e6f4453e.jpg)  
Fig. 6. The heatmap visualization of PAM and CLAM. Compared to CLAM, PAM more comprehensively focuses on cancerous regions of WSIs, which enhances WSI analytical performance.

RNN. A significant 1 refreshes the state h and prioritizes the present input x, whereas a minimal 1 maintains the state continuity and diminishes the relevance of the current input. The heatmaps of several samples are illustrated in Fig. 6. Notably, the comparison of PAM and CLAM heatmaps reveals that PAM more comprehensively focuses on cancerous regions of WSIs, facilitating thorough and effective modeling that enhances WSI analytical performance. Furthermore, the demarcation between cancerous and non-cancerous regions in PAM’s heatmaps is more pronounced, highlighting PAM’s enhanced capability to identify cancerous areas within WSI and underscoring its superior performance.

2) PAM Enables More Comprehensive and Holistic Modeling of WSIs: We also observe an interesting phenomenon: PAM can model not only the cancerous regions of WSIs but also the non-cancerous areas, including normal tissue and background regions, as illustrated in Fig. 7. Given the similarity between the matrix 1 and the multi-head self-attention mechanisms in Transformer [28], where different feature channels focus on various sequential dependencies, the K-means algorithm was applied to cluster matrix 1 along its second dimension. This approach reveals two main clusters: the first closely resembles the overall heatmaps and primarily focuses on cancerous regions and the second contrasts with the first by mainly targeting non-cancerous normal tissues and background areas. This mechanism may contribute to PAM’s comprehensive modeling capabilities for WSIs.

Overall  
Cancer Related  
![](images/3de2416e42365fba0cb7e978f7f07eb1f25e0809a87a18fe7feb1fd0842ab868.jpg)  
Fig. 7. Visualization of the overall heatmaps, corresponding cancer-related heatmaps and cancer-unrelated heatmaps.

TABLE VI  
SPACE AND TIME EFFICIENCY ANALYSIS
<table><tr><td>Method</td><td>GPU Memory (GB)</td><td>Training Time (S)</td></tr><tr><td>CLAM [2]</td><td>2.78</td><td>30.07</td></tr><tr><td>DS-MIL [27]</td><td>1.10</td><td>25.92</td></tr><tr><td>HIPT [16]</td><td>2.37</td><td>31.90</td></tr><tr><td>DTFD-MIL [11]</td><td>1.13</td><td>60.09</td></tr><tr><td>TransMIL [3] a</td><td>20.30</td><td>53.54</td></tr><tr><td>HIT [29] a</td><td>7.31</td><td>63.84</td></tr><tr><td>Patch-GCN [46] b</td><td>16.70</td><td>56.93</td></tr><tr><td>WiKG [47] b</td><td>23.14</td><td>86.30</td></tr><tr><td>S4-Model [38] c</td><td>21.85</td><td>49.01</td></tr><tr><td>MambaMIL [48]c</td><td>10.92</td><td>88.60</td></tr><tr><td>PAM w/o TIR</td><td>14.37</td><td>60.07</td></tr><tr><td>PAM</td><td>1.82</td><td>40.88</td></tr></table>

a / b / c epresent Transformer / Graph / M based methods respectively.

## V. DISCUSSION

In this paper, we propose Pathology Mamba (PAM) as a novel framework for WSI analysis, which is capable of modeling WSIs efficiently and effectively through the Mamba-based module with the incorporation of local-aware scanning mechanism, and able to enhance prediction ability by keeping the alignment between feature distributions of training and testing. More specifically, PAM possesses a global receptive field for WSIs and retains the ability to effectively capture local information and hierarchical information using the Local-aware Scanning (LAS) module and Bi-directional Mamba Encoder (BiME). Additionally, PAM maintains feature distribution alignment between the training and testing phases by utilizing the Test-time Importance Resampling (TIR) module.

TABLE VII  
RESULTS ON CANCER SUB-TYPING TASKS BY USING CTRANSPATH IMAGE ENCODER
<table><tr><td rowspan="2">Method</td><td colspan="2">BRCA</td><td colspan="2">ESCA</td></tr><tr><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td></tr><tr><td>CLAM [2]</td><td>0.925±0.047</td><td>0.887±0.033</td><td>0.988±0.018</td><td>0.946±0.048</td></tr><tr><td>DS-MIL [27]</td><td>0.928±0.044</td><td>0.896±0.022</td><td>0.991±0.026</td><td>0.933±0.049</td></tr><tr><td>HIPT [16]</td><td>0.927±0.035</td><td>0.884±0.032</td><td>0.988±0.018</td><td>0.927±0.053</td></tr><tr><td>DTFD-MIL [11]</td><td>0.925±0.043</td><td>0.905±0.020</td><td>0.979±0.022</td><td>0.933±0.040</td></tr><tr><td>TransMIL [3]</td><td>0.924±0.039</td><td>0.894±0.022</td><td>0.988±0.018</td><td>0.900±0.071</td></tr><tr><td>HIT [29]</td><td>0.925±0.039</td><td>0.899±0.030</td><td>0.996±0.011</td><td>0.953±0.029</td></tr><tr><td>Patch-GCN [46]</td><td>0.924±0.048</td><td>0.898±0.028</td><td>0.991±0.018</td><td>0.947±0.038</td></tr><tr><td>WiKG [47]</td><td>0.917±0.046</td><td>0.903±0.023</td><td>0.994±0.011</td><td>0.940±0.066</td></tr><tr><td>S4-Model [38]</td><td>0.927±0.043</td><td>0.904±0.025</td><td>0.994±0.011</td><td>0.947±0.055</td></tr><tr><td>MambaMIL [48]</td><td>0.920±0.049</td><td>0.897±0.021</td><td>0.985±0.019</td><td>0.947±0.025</td></tr><tr><td>PAM (Ours)</td><td>0.937±0.040</td><td>0.912±0.031</td><td>0.998±0.005</td><td> $\mathbf { 0 . 9 6 7 { \scriptstyle \pm 0 . 0 4 3 } }$ </td></tr></table>

We conducted time and space efficiency analysis of PAM and compared it with other methods. The results are shown in Table VI. It is important to note that since the computational complexity of Transformers is quadratic with respect to sequence length, existing Transformer-based methods often incorporate various techniques to reduce computational complexity, such as Nyström self-attention (TransMIL [3]) and bag-based design(HIT [29]), which may hinder their ability to effectively and directly model WSIs. And GNN-based methods model WSIs using a graph structure, which may lead to feature oversmoothing and difficulty in capturing long-range dependencies in WSIs. In contrast, the proposed Mamba-based method can model long-range dependencies between patches of WSIs directly and effectively, without introducing significant additional computational complexity. Additionally, the GPU memory consumption and training time of the proposed PAM decrease significantly when incorporating the TIR module, which improves the model’s efficiency while achieving better performance by ensuring consistency in feature distribution between the training and testing phases and augmenting the training data.

In this work, we select ConvNeXt pretrained on ImageNet as the feature extractor. For more comparative analysis, the performance of different methods using the feature extractors that were trained on pathology images (e.g., CTransPath [54]) was also investigated. The results presented in Table VII reveal significant performance improvements for all methods after adopting CTransPath as the feature encoder. Notably, PAM also consistently outperforms the other methods under this setting, which demonstrates that PAM can consistently achieve great performance when using different feature extractors.

We believe that the proposed PAM model has the potential to serve as an auxiliary tool in routine clinical diagnostics. The PAM model can provide pathologists with a more accurate and efficient diagnosis of WSIs, which can help reduce the workload of pathologists and improve the efficiency of clinical diagnostics. In addition, the PAM model can also be used to assist pathologists in identifying regions of interest in WSIs, which can help pathologists focus on the most important regions.

The proposed framework has several limitations. In the survival prediction experiments, PAM’s reported performance was achieved using local-aware flat scanning. Although hierarchical information integration boosts cancer subtyping task performance, we do not observe a substantial improvement in survival prediction tasks. This suggests that while hierarchical information offers discriminative features across scales, it might influence the model’s capacity for a global representation of cancer infiltration at a singular scale. In future work, we will explore modeling the hierarchical structure of WSIs using more effective scanning strategies or feature fusion methods to promote the ability of PAM for survival prediction tasks.

## VI. CONCLUSION

This paper introduces Pathology Mamba (PAM) as an effective solution for Whole Slide Images (WSIs) analysis. The proposed framework aligns the feature distribution between the training and testing phases, captures long-range dependencies and local information within the slides, and leverages the hierarchical nature of WSIs. Specifically, we introduce a Local-aware Scanning strategy to manage the flat or hierarchical nature of WSIs effectively. The hierarchical scanning integrates information from cellular details to broader intratumoral features, which is crucial, especially for cancer subtyping tasks. Moreover, a Test-time Importance Resampling paradigm was proposed to align the feature distribution between training and testing data, thereby facilitating model prediction. Furthermore, a new Bi-directional Mamba Encoder was devised to model WSIs effectively and efficiently through the incorporation of a selective state space model. The efficacy of PAM was demonstrated through experiments on nine slidelevel tasks, including cancer subtyping and survival prediction tasks, where PAM showed superior performance over existing methods. The results indicate that PAM is a promising framework for WSI analysis.

## REFERENCES

[1] T. C. Cornish, R. E. Swapp, and K. J. Kaplan, “Whole-slide imaging: Routine pathologic diagnosis,” Adv. Anatomic Pathol., vol. 19, no. 3, pp. 152–159, 2012.

[2] M. Y. Lu, D. F. K. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood, “Data-efficient and weakly supervised computational pathology on whole-slide images,” Nature Biomed. Eng., vol. 5, no. 6, pp. 555–570, Mar. 2021.

[3] Z. Shao et al., “TransMIL: Transformer based correlated multiple instance learning for whole slide image classification,” in Proc. Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 2136–2147.

[4] J. Tanizaki et al., “Report of two cases of pseudoprogression in patients with non–small cell lung cancer treated with nivolumab—Including histological analysis of one case after tumor regression,” Lung Cancer, vol. 102, pp. 44–48, Dec. 2016.

[5] Y. Fu et al., “OTFPF: Optimal transport based feature pyramid fusion network for brain age estimation,” Inf. Fusion, vol. 100, Dec. 2023, Art. no. 101931.

[6] W. Lu, S. Graham, M. Bilal, N. Rajpoot, and F. Minhas, “Capturing cellular topology in multi-gigapixel pathology images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), Jun. 2020, pp. 260–261.

[7] S.-C. Huang et al., “Deep neural network trained on gigapixel images improves lymph node metastasis detection in clinical settings,” Nature Commun., vol. 13, no. 1, pp. 1–14, Jun. 2022.

[8] S. Javed, A. Mahmood, N. Werghi, K. Benes, and N. Rajpoot, “Multiplex cellular communities in multi-gigapixel colorectal cancer histology images for tissue phenotyping,” IEEE Trans. Image Process., vol. 29, pp. 9204–9219, 2020.

[9] G. Campanella et al., “Clinical-grade computational pathology using weakly supervised deep learning on whole slide images,” Nature Med., vol. 25, no. 8, pp. 1301–1309, 2019.

[10] A. Shmatko, N. Ghaffari Laleh, M. Gerstung, and J. N. Kather, “Artificial intelligence in histopathology: Enhancing cancer research and clinical oncology,” Nature Cancer, vol. 3, no. 9, pp. 1026–1038, Sep. 2022.

[11] H. Zhang et al., “DTFD-MIL: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 18802–18812.

[12] N. Hashimoto et al., “Multi-scale domain-adversarial multiple-instance CNN for cancer subtype classification with unannotated histopathological images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 3851–3860.

[13] W. Hou et al., “H2-MIL: Exploring hierarchical representation with heterogeneous multiple instance learning for whole slide image analysis,” in Proc. AAAI Conf. Artif. Intell., 2022, vol. 36, no. 1, pp. 933–941.

[14] Y. Guan et al., “Node-aligned graph convolutional network for wholeslide image representation and classification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 18791–18801.

[15] T. H. Chan, F. J. Cendra, L. Ma, G. Yin, and L. Yu, “Histopathology whole slide image analysis with heterogeneous graph representation learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 15661–15670.

[16] R. J. Chen et al., “Scaling vision transformers to gigapixel images via hierarchical self-supervised learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 16144–16155.

[17] Z. Wang, L. Yu, X. Ding, X. Liao, and L. Wang, “Lymph node metastasis prediction from whole slide images with transformer-guided multiinstance learning and knowledge transfer,” IEEE Trans. Med. Imag., vol. 41, no. 10, pp. 2777–2787, Oct. 2022.

[18] F. Ghaznavi, A. Evans, A. Madabhushi, and M. Feldman, “Digital imaging in pathology: Whole-slide imaging and beyond,” Annu. Rev. Pathol., Mech. Disease, vol. 8, no. 1, pp. 331–359, Jan. 2013.

[19] S. Graham et al., “Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images,” Med. Image Anal., vol. 58, Dec. 2019, Art. no. 101563.

[20] P. Pati et al., “Hact-net: A hierarchical cell-to-tissue graph neural network for histopathological image classification,” in Uncertainty for Safe Utilization of Machine Learning in Medical Imaging, and Graphs in Biomedical Image Analysis. Berlin, Germany: Springer, 2020, pp. 208–219.

[21] S. Abousamra et al., “Multi-class cell detection using spatial context representation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 3985–3994.

[22] K. AbdulJabbar et al., “Geospatial immune variability illuminates differential evolution of lung adenocarcinoma,” Nature Med., vol. 26, no. 7, pp. 1054–1062, Jul. 2020.

[23] N. Brancati et al., “BRACS: A dataset for BReAst carcinoma subtyping in H&E histology images,” Database, vol. 2022, Oct. 2022, Art. no. baac093.

[24] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” 2023, arXiv:2312.00752.

[25] D. Y. Fu et al., “Hungry hungry hippos: Towards language modeling with state space models,” in Proc. 11th Int. Conf. Learn. Represent., 2023, pp. 1–27.

[26] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in Proc. Int. Conf. Mach. Learn., 2018, pp. 2127–2136.

[27] B. Li, Y. Li, and K. W. Eliceiri, “Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 14318–14328.

[28] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1–11.

[29] Y. Huang, W. Zhao, S. Wang, Y. Fu, Y. Jiang, and L. Yu, “ConSlide: Asynchronous hierarchical interaction transformer with breakup-reorganize rehearsal for continual whole slide image analysis,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 21292–21303.

[30] A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,” 2021, arXiv:2111.00396.

[31] A. Gu et al., “Combining recurrent, convolutional, and continuous-time models with linear state space layers,” in Proc. Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 572–585.

[32] K. Goel, A. Gu, C. Donahue, and C. Ré, “It’s raw! Audio generation with state-space models,” in Proc. Int. Conf. Mach. Learn., 2022, pp. 7616–7633.

[33] G. Saon, A. Gupta, and X. Cui, “Diagonal state space augmented transformers for speech recognition,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Jun. 2023, pp. 1–5.

[34] E. Nguyen et al., “S4ND: Modeling images and videos as multidimensional signals with state spaces,” in Proc. Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 2846–2861.

[35] M. Poli et al., “Hyena hierarchy: Towards larger convolutional language models,” 2023, arXiv:2302.10866.

[36] Y. Sun et al., “Retentive network: A successor to transformer for large language models,” 2023, arXiv:2307.08621.

[37] B. Peng et al., “RWKV: Reinventing RNNs for the transformer era,” 2023, arXiv:2305.13048.

[38] L. Fillioux, J. Boyd, M. Vakalopoulou, P.-H. Cournède, and S. Christodoulidis, “Structured state space models for multiple instance learning in digital pathology,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent., 2023, pp. 594–604.

[39] J. Lei Ba, J. Ryan Kiros, and G. E. Hinton, “Layer normalization,” 2016, arXiv:1607.06450.

[40] S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,” Neural Netw., vol. 107, pp. 3–11, Nov. 2018.

[41] Y. Liu et al., “VMamba: Visual state space model,” 2024, arXiv:2401.10166.

[42] D. B. Rubin, “Using the sir algorithm to simulate posterior distributions,” in Proc. 3rd Valencia Int. Meeting. Oxford, U.K.: Clarendon Press, Jun. 1988, pp. 395–402.

[43] S. M. Xie, S. Santurkar, T. Ma, and P. S. Liang, “Data selection for language models via importance resampling,” in Proc. Adv. Neural Inf. Process. Syst., vol. 36, 2024, pp. 1–27.

[44] W. Kool, H. van Hoof, and M. Welling, “Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 3499–3508.

[45] C. Kim, A. Sabharwal, and S. Ermon, “Exact sampling with integer linear programs and random perturbations,” in Proc. AAAI Conf. Artif. Intell., 2016, vol. 30, no. 1, pp. 1–9.

[46] R. J. Chen et al., “Whole slide images are 2D point clouds: Contextaware survival prediction using patch-based graph convolutional networks,” in Proc. Int. Conf. Med. Image Comput. Comput. Assist. Intervent., Strasbourg, France. New York, NY, USA: Springer, Sep. 2021, pp. 339–349.

[47] J. Li et al., “Dynamic graph representation with knowledge-aware attention for histopathology whole slide image analysis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024, pp. 11323–11332.

[48] S. Yang, Y. Wang, and H. Chen, “MambaMIL: Enhancing long sequence modeling with sequence reordering in computational pathology,” 2024, arXiv:2403.06800.

[49] S. G. Zadeh and M. Schmid, “Bias in cross-entropy-based training of deep survival networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 9, pp. 3126–3137, Sep. 2021.

[50] R. J. Chen et al., “Multimodal co-attention transformer for survival prediction in gigapixel whole slide images,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Oct. 2021, pp. 4015–4025.

[51] B. E. Bejnordi et al., “Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer,” JAMA, vol. 318, no. 22, pp. 2199–2210, Dec. 2017.

[52] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2022, pp. 11976–11986.

[53] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2014, arXiv:1412.6980.

[54] X. Wang et al., “Transformer-based unsupervised contrastive learning for histopathological image classification,” Med. Image Anal., vol. 81, Oct. 2022, Art. no. 102559.