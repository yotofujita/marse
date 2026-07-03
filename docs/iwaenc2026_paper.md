# MASKED AUTOREGRESSIVE SPEECH ENHANCEMENT

# WITH CONTINUOUS NEURAL AUDIO CODEC REPRESENTATIONS

# Yoto Fujita^1 ,^2 Simon Leglaive^1 Laurent Girin^2

(^1) CentraleSupelec, IETR (UMR CNRS 6164), France ́
(^2) Univ. Grenoble Alpes, CNRS, Grenoble-INP, GIPSA-lab, France

## ABSTRACT

Previous work on speech enhancement (SE) based on masked gener-
ative modeling relied on discrete token representations of audio sig-
nals, obtained using neural audio codecs (NACs). However, a recent
study has shown that continuous latent representations of NACs can
be advantageous for SE in terms of speech quality and intelligibil-
ity. In this work, we propose masked autoregressive SE (MARSE),
a unified probabilistic framework for SE based on iterative decoding
of masked clean speech frames using continuous NAC representa-
tions of speech. In particular, we investigate a set of different decod-
ing policies, ceteris paribus, that is, using the same DNN (a Con-
former model), the same NAC (the DAC codec) and the same train-
ing setup. The results show that the proposed framework enables a
flexible trade-off between SE performance and computational cost.
Audio examples and code are available online.^1

Index Terms— Speech enhancement, masked generative mod-
eling, neural audio codec, autoregressive modeling, speech represen-
tations.

## 1. INTRODUCTION

Speech enhancement (SE) in noise has a long history of conventional
signal processing techniques [1, 2] and has been largely revisited in
the last decade with deep neural networks (DNNs) [3]. DNN-based
SE has been increasingly performed in learned latent spaces instead
of using conventional representations such as the short-time Fourier
transform. In particular, the compact yet expressive discrete latent
representations provided by neural audio codecs (NAC), which con-
sist of sequences of vector quantizer codes called tokens, have re-
cently facilitated SE based on token sequence modeling with Trans-
formers [4], in the line of speech language models [5]. The tokens
corresponding to a clean speech signal frame at time t are decoded
(i.e., predicted) using the noisy signal and possibly previously de-
coded speech tokens. Token-based SE has been explored with differ-
ent decoding strategies: non-autoregressive decoding [6, 7], causal
autoregressive decoding [6, 8, 9], or non-causal masked generative
decoding [10, 11, 12, 13].
This paper aims at both investigating SE models that operate on
continuous NAC representations and extending the study of decod-
ing policies for SE. This is motivated by the following three points.
First, although the above-mentioned token-based SE methods are
reported to exhibit good speech quality, they can suffer from lim-
ited preservation of acoustic details that matter for speech quality
and intelligibility, leading to reduced performance on some down-
stream tasks such as automatic speech recognition (ASR) [10]. Sec-
ond, it has been recently shown in [6] that using a continuous NAC

(^1) https://yotofujita.github.io/marse
**T T T T**
Concatenated noisy and clean speech representaiton
MSE-based loss over the masked positions
Trainable Conformer
Noisy speech
Noisy speech representationMask sequence
Partially-masked clean speech representation
Fully-decoded clean speech representation
Predicted clean speech
× N
**MARSE inference with iterative decoding of masked frames MARSE training**
Pretrained NAC encoder
Pretrained NAC decoder
Pretrained NAC encoderPretrained NAC encoder
_Partially mask_
Trainable Conformer
Fully-decoded clean speech representation
Noisy speech Clean speech
Fig. 1. Overview of the proposed MARSE framework applied on a
continuous NAC representation (left: inference; right: training).
representation—that is, embedding vectors at the output of the NAC
encoder and before the quantizer—can substantially improve speech
intelligibility and quality. Third, most of the token-based methods
mentioned above focus on a specific decoding policy with a specific
experimental setup, and the trade-off between performance and com-
putational cost regarding decoding policy has not been explored.
Independently of the SE problem, a unified generative frame-
work called masked autoregressive (MAR) has recently been pro-
posed for image generation [14]. Here, decoding is formalized as an
iterative unmasking of masked frames, and this framework actually
encompasses the different policies explored in token-based SE. Im-
portantly, MAR has been applied in [14] on continuous vector repre-
sentations of images. In this paper, inspired by both MAR and token-
based masked generative modeling of SE, we propose a unified prob-
abilistic framework for SE called masked autoregressive speech en-
hancement (MARSE), in which iterative decoding is applied on con-
tinuous NAC representations and defined as an autoregressive (AR)
process on blocks of signal frames. We compare several decoding
policies as variants of the number of iterations and/or the definition
of the blocks. We study their effect on speech quality, intelligibility,
and computational cost, by a fair comparison using the same network
architecture (in the present case, a Conformer-based model [15]), the
same NAC (in the present case, Descript Audio Codec or DAC [16]),
and the same training setup. Experiments show that the MARSE
framework allows for a flexible trade-off between SE performance
(speech quality and intelligibility) and computational cost.


## 2. METHOD

In a probabilistic framework, SE can be defined as the problem of es-
timating a conditional distribution pθ(x| y) of parameters θ, where
x, y ∈ A denote the clean and noisy speech signals respectively,
defined in some representation domainA. Inspired by [6], we con-
sider a model that operates on the continuous latent representation
of a pretrained NAC, before quantization. We thus have x = {xt∈
RD}Tt=1and y = {yt∈ RD}Tt=1, where T denotes the number
of time frames and D the NAC’s latent space dimension. Deep
learning-based SE methods rely on DNNs to parametrize the distri-
butionpθ(x| y). In a supervised setting, the parametersθ are learned
using a labeled dataset of paired clean and noisy speech signals, typ-
ically by minimizing the negative log-likelihood − lnpθ(x | y)
averaged over the training examples.

2.1. Masked autoregressive speech enhancement

Within this probabilistic framework, defining pθ(x | y) involves
(i) defining the distribution of the clean speech variables xt,
t ∈ { 1 ,...,T}, and their temporal dependencies; and (ii) defining
the DNN architecture that implements this probabilistic model. In
this work, we extend the MAR framework of [14] to the SE problem
(see Section 1), resulting in the general class of MARSE models,
which allows for arbitrary assumptions about these dependencies
and tackles SE as an iterative decoding process, as illustrated in the
left part of Fig. 1. In MARSE, the conditional distribution of x given
y is defined by:

```
pθ(x| y) =
```
## YN

```
i=
```
```
pθ
```
## 

```
xM(i)| y, xV (i)
```
## 

## (1)

## =

## YN

```
i=
```
## Y

```
t∈M(i)
N(xt;fθ,t(y, xV (i)), I), (2)
```
where N ∈ { 1 ,...,T} is the number of iterations in the decoding
process and

- xM(i)denotes the set of predicted frames at iteration i, with

```
M(i)⊆{ 1 ,...,T} and such that
```
## SN

```
i=1M(i) ={^1 ,...,T};
```
- xV (i)denotes the set of visible (i.e. previously decoded) frames at
    the beginning of iteration i, with V (1) = ∅, V (i + 1) = V (i) ∪
    M(i) and M(i)∩ V (i) =∅ for i≥ 1.
- and fθ: RD×T× RD×T 7→ RD×Tdenotes a neural network
    that processes the noisy speech y and visible clean speech frames
    xV (i)to predict xM(i). Inspired by [6], we use a Conformer-based
    network taking as input the temporal concatenation of noisy and
    partially-masked clean speech representations. The masked clean
    speech frames at positions t∈{ 1 ,...,T}\V (i) are here replaced
    by a learnable mask vector.

By construction, the collection of index sets{M(i)}Ni=1is a disjoint
partition of{ 1 ,...,T}. The factorization in (1) is therefore an appli-
cation of the chain rule of probability that constitutes an autoregres-
sive model over blocks of frames, which forms a proper probability
density function (pdf) over x as long as each conditional factor in
(1) is also a valid pdf. Eq. (2) shows that within a block, the vectors
xtare assumed to be mutually conditionally independent.

2.2. Decoding policies for inference

In MARSE, a fundamental aspect of the model is the definition of
the collection of index sets {M(i)}Ni=1, which will be referred to
as the decoding policy in the following sections. This policy di-
rectly relates to the assumed temporal dependencies between the

```
clean speech frames. The flexibility of MARSE comes from the
possibility of designing various decoding policies and, in particular,
choosing an arbitrary number of iterations N ∈ { 1 ,...,T}. Be-
low, we present causal and non-causal decoding policies, which can
all be characterized in two steps: (i) defining the number of frames
decoded in parallel at iteration i, which is equal to the cardinal of
M(i) in (1), denoted by card(M(i)); (ii) defining the indices of the
frames to predict, that is, specifying the set M(i) given its cardinal.
In this work, following [17], the cardinal of M(i) is fixed accord-
ing to a predefined cosine schedule. More precisely, card(M(i)) =
ni+1− niframes, where ni= card(V (i)) =
```
## 

## T

## 

```
1 − γ(i−N^1 )
```
## 

## ,

```
with γ : r ∈ [0, 1]7→ cos
```
```
π
2 r
```
## 

```
∈ [0, 1]. Here, nidenotes the num-
ber of predicted frames up to, and not including, iteration i, such that
n 1 = card(V (1)) = 0. Then, variants of the decoding policy are
obtained by specifying different strategies for defining M(i) at each
iteration and/or by choosing a different number of iterations.
Causal decoding. In the block-wise causal AR decoding policy,
the set of indices to predict is defined by M(i) = {ni+ 1,ni+
2 ,...,ni+1}. Thus, each block of frames consists of a subset of
consecutive frames, and the blocks follow a causal ordering along
the iterations. In the limit case where N = 1, we have M(1) =
{ 1 ,...,T} and V (1) =∅, i.e. all frames are decoded all at once and
the decoding is not really AR anymore (we thus can refer to it as non-
AR). As can be seen from (2), this non-AR decoding assumes that all
variables xt, t∈{ 1 ,...,T}, are mutually independent. It constitutes
the most naive temporal model. In the limit case where N = T ,
we recover the conventional frame-wise causal AR decoding with
M(i) = {i} and V (i) = { 1 ,...,i − 1 } for all i ∈ { 1 ,...,T}.
This policy is the most expressive in terms of temporal modeling,
as no conditional independence assumptions are made. However, it
requires T forward passes in the neural network fθ, which can be
computationally expensive.
Non-causal decoding with oracle index selection. Contrary to the
previous policy, in the non-causal decoding policy a masked frame
at a given time step can be decoded conditioned on frames in the fu-
ture. The intuition behind a non-causal policy for SE is that frames
associated with high signal-to-noise ratio can be decoded first, and
then used to decode noisier frames from the past. In the non-causal
decoding policy with oracle index selection, at iteration i, we first
decode all the frames that remain to be decoded. Then, we use
the ground-truth clean speech to select, among those frames, the
card(M(i)) ones that have the lowest prediction error (in terms of
squared error). This decoding policy is not realistic, as it requires
the availability of the ground-truth clean speech, but it is meant to
provide a form of upper-bound on the performance of MARSE with
non-causal decoding.
Non-causal decoding with random index selection. We also con-
sider the most naive non-causal decoding policy based on random
index selection, as used in [14]. In this policy, given its cardinal, at
each iteration i∈{ 1 ,...,N} the set M(i) is filled with indices cho-
sen randomly within the set of remaining masked frames{ 1 ,...,T}\
V (i), following a uniform distribution. This policy can be inter-
preted as randomly partitioning{ 1 ,...,T} into N blocks of size de-
fined by the cosine schedule function γ, and modeling the proba-
bilistic dependencies between these blocks with (1).
```
```
2.3. Training
```
```
The training of the MARSE model is illustrated in the right part of
Fig. 1. Given a labeled dataset of noisy-clean speech pairs (y, x),
it is done by optimizing the negative log-likelihood defined from (2)
```

(averaged over the training set). This is equivalent to minimizing the
following MSE-based loss function:

```
L(θ; x) = EU(i;{ 1 ,...,N})
```
##  X

```
t∈M(i)
```
(^) xt− fθ,t

## 

```
y, xV (i)
```
## ^

## 

```
2
```
## 

## , (3)

where summation over i ∈ { 1 ,...,N} has been equivalently re-
placed by an expectation. In masked generative modeling, it is com-
mon to approximate the expectation using one single sample i [17].
We follow this approach, where the number of masked frames is
computed using the same cosine schedule as defined before for the
decoding policies, i.e. card(M(i)) = ⌊γ(r)· T⌋ where r = (i−
1)/N. In practice, r is sampled from U([0, 1[), which is consis-
tent with r = (i− 1)/N and i ∼ U({ 1 ,...,N}) as N → +∞.
Given card(M(i)), the set M(i) is then defined randomly within
{ 1 ,...,T}, following a uniform distribution. Finally, V (i) is set to
{ 1 ,...,T}\ M(i).

2.4. Mitigating exposure bias with quantization

At inference, for each iteration i ∈ { 1 ,...,N} of the decoding pro-
cess, we compute the prediction of the clean speech frames xM(i)=
{fθ,t(y, xV (i))}t∈M(i), which corresponds to the mean of the Gaus-
sian distributions in (2). In the next iteration, these predicted frames
are reinjected into the model as visible frames: xV (i+1)= xV (i)∪
xM(i)with xV (1)= ∅. This is different from the training process,
where the frames xV (i)in the loss function (3) are obtained from
ground-truth clean speech. This mismatch between training and test-
ing conditions, commonly referred to as exposure bias [18], can re-
sult in error accumulation during decoding iterations. To mitigate
this problem, as proposed in [6], we leverage the fact that we work
in the latent space of a NAC to quantize the visible frames xV (i)us-
ing the pretrained NAC quantizer, before feeding them to the model
fθ. This is done both at training and inference.

## 3. EXPERIMENTS

This section describes the experiments we conducted to assess
MARSE on continuous NAC representations with the different de-
coding policies, under the same general configuration (in terms of
NAC representation, dataset, DNN architecture, training setup, and
evaluation metrics).

3.1. Experimental setup

NAC representation. Regarding the NAC representation, we used
DAC [16], with continuous latent representations of dimension D =
1024. DAC is one of the most widely used NACs, based on a convo-
lutional encoder-decoder architecture and a 12-stage residual vector
quantizer (RVQ), whose codebooks each contain 1024 codewords.

Data. For MARSE training and validation, we used the 212-hour
train-360 and 11-hour dev subsets of Libri1Mix, respectively.
This dataset is obtained by discarding one of the two speakers in
Libri2Mix mixtures [21]. Each subset of Libri1Mix contains paired
noisy and clean utterances at 16 kHz generated by mixing clean
speech from LibriSpeech [22] with noise from WHAM! [23]. We
used the 11-hour test subset of Libri1Mix to evaluate in-domain
performance. To further evaluate out-of-domain performance, we
created a 4-hour synthetic dataset, here referred to as LibriDE-
MAND, by mixing clean speech from the train-clean-
subset of LibriSpeech with noise from DEMAND [24] with a very
similar procedure to Libri1Mix. For training, a 1-second clip is

```
randomly cropped from each utterance to form a sequence of em-
bedding vectors with length T = 50. At inference, longer utterances
are segmented into 1 s-segments that are processed independently
by the model.
```
```
DNN architecture. We used a Conformer architecture [15] with 16
Conformer blocks, hidden dimension set to 384, 12 attention heads,
each of dimension 32, convolution kernel of size 10, expansion fac-
tor for convolution module set to 2, and expansion factor for feed-
forward module set to 4. To align the dimension of DAC embedding
vectors with the hidden dimension of the Conformer, two learnable
linear layers are applied before and after the Conformer.
```
```
Training setup. The model was trained by multi-GPU batch gradi-
ent descent, using the AdamW optimizer [25] with a learning rate
of 10 −^3 , beta coefficients of (0. 9 , 0 .95), and weight decay of 0. 05.
The batch size was set to 128 for each GPU and the number of
epochs was fixed to 300. A cosine decay learning rate scheduler
with warmup was applied, where the learning rate linearly increased
from 0 to the target value over 10 epochs and followed a cosine de-
cay schedule over the remaining 290 epochs. The multi-GPU train-
ing was implemented with the distributed data parallel module of
PyTorch across four NVIDIA RTX A100 GPUs with 40GB VRAM.
```
```
Evaluation metrics. The enhanced speech quality was measured
using the non-intrusive DNSMOS P.835 SIG, BAK and OVRL
scores [26] (between 1. 0 and 5. 0 , the higher the better), which mea-
sure respectively the quality of the speech signal, the intrusiveness of
background noise, and the overall quality. Speech intelligibility was
measured by the differential word error rate (dWER in %, the lower
the better) between the transcriptions obtained from the enhanced
and ground-truth speech, using a pretrained ASR model based on
wav2vec 2.0 [27].^2 As for computational cost analysis, we measured
the amount of giga floating-point operations (GFLOPs).
```
```
Baselines. As conventional SE baselines, we used ConvTasNet [19]
and DPTNet [20], using the publicly-available models pretrained on
Libri1Mix.^3 We also compare MARSE against two NAC-based SE
models from [6] that we retrained. Those do not use masking at
training or inference, although they have a specific decoding policy:
(i) C-NAR is a non-AR model using the ‘all at once’ decoding corre-
sponding to N = 1, and (ii) C-AR is a frame-wise causal AR model,
corresponding to N = T. Concretely, C-NAR was trained and in-
ferred using a single-step noisy-to-clean mapping, whereas C-AR
was trained with a next-frame prediction objective and inferred via
frame-wise causal AR decoding of clean representations. Note that
we do not use token-based baselines, as it has been shown in [6] that
C-NAR and C-AR significantly outperform their counterparts using
token representations.
```
```
3.2. Results
```
```
Table 1 provides the results obtained by the proposed MARSE model
with the three decoding policies presented in Section 2.2, using N =
10 decoding steps, as well as those obtained by the baselines. Causal
decoding is denoted ‘MARSE-causal’. Non-causal decoding with
random index selection is denoted ‘MARSE-NC-random’ and con-
stitutes the practical non-causal baseline. Non-causal decoding with
oracle index selection is denoted ‘MARSE-NC-oracle’ and is re-
ported as an upper bound. Regarding the baselines, the SE quality
of the conventional methods (ConvTasNet and DPTNet) was glob-
```
(^2) https://huggingface.co/facebook/wav2vec2-base-960h
(^3) https://huggingface.co/JorisCos


```
Libri1Mix LibriDEMAND
Model SIG↑ BAK↑ OVRL↑ dWER↓ GFLOPs↓ SIG↑ BAK↑ OVRL↑ dWER↓ GFLOPs↓
Noisy 2.46 1.81 3.08 30.19 - 2.55 2.10 1.92 21.77 -
ConvTasNet [19] 3.47 4.10 3.22 9.79 49 3.41 3.85 3.02 10.49 51
DPTNet [20] 3.44 4.08 3.16 10.05 12 3.38 3.84 2.98 9.57 12
C-NAR [6] 3.60 4.08 3.32 12.84 1235 3.55 3.76 3.08 9.61 1334
C-AR [6] 3.64 4.11 3.37 20.89 3856 3.55 3.76 3.09 15.91 4166
MARSE-causal 3.62 4.10 3.34 12.68 1912 3.54 3.80 3.11 9.35 2065
MARSE-NC-random 3.62 4.09 3.34 13.39 1912 3.54 3.83 3.12 10.13 2065
MARSE-NC-oracle 3.61 4.08 3.34 12.87 1912 3.57 3.83 3.14 8.86 2065
```
Table 1. Performance obtained by the proposed MARSE model for N = 10 iterations and the baselines on the Libri1Mix in-domain dataset
and LibriDEMAND out-of-domain dataset. The Metrics are described in the text. The scores for input noisy speech (‘Noisy’ item) are
provided as a lower reference. Best and second-best scores in each column are bold and underlined.

```
1 5 10 20 30 40 50
Number of decoding steps
```
```
3.
```
```
3.
```
```
3.
```
```
3.
```
```
3.
```
```
DNSMOS OVRL
C-NAR
C-AR
MARSE-causal
```
```
MARSE-NC-random
MARSE-NC-oracle
```
Fig. 2. DNSMOS OVRL score obtained by the proposed MARSE
model (for the three decoding policies) and the C-AR and C-NAR
baselines on a subset of 300 samples from Libri1Mix test, as a
function of the number of decoding iterations.

ally lower than for the other models, but their computational cost
was very reasonable. The C-AR and C-NAR baselines showed con-
trasting performance on the in-domain Libri1Mix. C-AR exhibited
higher quality (SIG = 3. 64 , BAK = 4. 11 , OVRL = 3. 37 ) but slower
inference (GFLOPs = 3856 ), while C-NAR exhibited lower qual-
ity (SIG = 3. 60 , BAK = 4. 08 , OVRL = 3. 32 ) but faster inference
(GFLOPs = 1235 ). This observation is consistent with the assump-
tion behind these models: C-AR fully models the causal dependency
of speech frames at the expense of computational cost, as it requires
T forward passes in the model for inference, while C-NAR inde-
pendently models each frame for the benefit of computational cost,
as it requires a single forward pass in the model for inference. In
general, the MARSE model, which assumes inter-block dependency
but inner-block independence, demonstrated intermediate SE per-
formance between C-NAR and C-AR, and its computational cost
was also in between (e.g., SIG = 3. 62 , BAK = 4. 10 , OVRL = 3. 34 ,
GFLOPs = 1912 for MARSE-causal). This provides an illustration
of the flexibility of the proposed MARSE framework in terms of
trade-off between SE performance and computational cost.

Another noticeable result shown in Table 1 is that the MARSE
models exhibited intelligibility clearly better than C-AR and even
competitive with C-NAR on both the in-domain Libri1Mix (e.g.,
dWER is 12. 68 % for MARSE-causal, 20. 89 % for C-AR, and
12. 84 % for C-NAR) and out-of-domain LibriDEMAND (e.g.,
dWER is 9. 35 % for MARSE-causal, 15. 91 % for C-AR, and
9. 61 % for C-NAR). Among the three MARSE models, the prac-

```
tical MARSE-causal performed the best on in-domain data, whereas
the non-practical MARSE-NC-oracle exhibited the best scores on
out-of-domain data. This is likely due to the accumulated error
along decoding iterations being larger on out-of-domain data than
on in-domain data, requiring a ‘reliable’ decoding policy. Unsur-
prisingly, MARSE-NC-random was slightly lower than the other
two MARSE models. These results suggest that better non-causal
frame-selection strategies could be worth exploring in future work.
Fig. 2 displays the DNSMOS OVRL score obtained by the pro-
posed MARSE model (for the three decoding policies) as a function
of the number of decoding iterations N ∈{ 1 , 5 , 10 , 20 , 30 , 40 , 50 }.
To limit computational cost and duration of the experiments, these
scores were evaluated on a subset of 300 samples randomly selected
from Libri1Mix test, therefore they can be different from those of
Table 1. We can see in Fig. 2 that the OVRL score of MARSE-causal
is quite close to that of C-NAR for N = 1 and to that of C-AR for
N = 50. This result is consistent with the fact that for N = 1, both
MARSE and C-NAR employ ‘all-at-once’ frame decoding, and for
N = 50, both MARSE-causal and C-AR employ frame-wise causal
AR decoding. Between these two ‘extreme’ decoding policies, the
MARSE-causal performance improves between C-NAR and C-AR
as the number of iterations increases. This confirms the flexibility of
the MARSE framework for exploring variants of decoding policies
and enabling a trade-off between SE performance and computational
cost. In particular, we see in Fig. 2 that the performance of MARSE-
causal starts to stagnate in the range 20 – 40 iterations. Setting N in
this range would thus provide a good trade-off between performance
and computational cost. Note that all MARSE models were trained
with the same strategy presented in Section 2.3.
```
## 4. CONCLUSION

```
In this work, we proposed the MARSE framework and explored
different iterative decoding policies for SE using continuous NAC
representations. The experimental results showed the ability of this
framework to provide a flexible trade-off between SE performance
and computational cost. In future work, we will focus on improving
non-causal decoding policies, which may be beneficial in terms of
generalization, as suggested by the results obtained with MARSE-
NC-oracle on the out-of-domain dataset.
```
```
Acknowledgement. This work was performed using computa-
tional resources from the Mesocentre Paris-Saclay, as part of the
ANR-funded DEESSE project (ANR-24-CE23-2229).
```

## 5. REFERENCES

```
[1] J. Benesty, S. Makino, and J. Chen, Speech enhancement.
Springer Science & Business Media, 2006.
[2] P. C. Loizou, Speech enhancement: Theory and practice.
CRC press, 2013.
[3] D. Wang and J. Chen, “Supervised speech separation based
on deep learning: An overview,” IEEE Trans. Audio, Speech,
Lang. Proc., vol. 26, no. 10, pp. 1702–1726, 2018.
[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones,
A. Gomez et al., “Attention is all you need,” Adv. Neural In-
form. Proc. Syst. (NeurIPS), vol. 30, 2017.
[5] P. Mousavi, G. Maimon, A. Moumen, D. Petermann, J. Shi,
H. Wu et al., “Discrete audio tokens: More than a survey!”
Trans. Mach. Learn. Res. (TMLR), Jun. 2025.
[6] S. Kammoun, X. Alameda-Pineda, and S. Leglaive, “Modeling
strategies for speech enhancement in the latent space of a neu-
ral audio codec,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc.
(ICASSP), 2026.
[7] Z. Wang, X. Zhu, Z. Zhang, Y. Lv, N. Jiang, G. Zhao, and
L. Xie, “SELM: Speech enhancement using discrete tokens
and language models,” in IEEE Int. Conf. Acoust., Speech, Sig.
Proc. (ICASSP), 2024.
[8] J. Yao, H. Liu, C. Chen, Y. Hu, E. Chng, and L. Xie, “GenSE:
Generative speech enhancement via language models using hi-
erarchical modeling,” in Int. Conf. Learn. Rep. (ICLR), 2024.
[9] H. Xue, X. Peng, and Y. Lu, “Low-latency speech enhance-
ment via speech token generation,” in IEEE Int. Conf. Acoust.,
Speech, Sig. Proc. (ICASSP), 2024.
```
[10] H. Yang, J. Su, M. Kim, and Z. Jin, “Genhancer: High-fidelity
speech enhancement via generative modeling on discrete codec
tokens,” in Interspeech, 2024.

[11] X. Li, Q. Wang, and X. Liu, “MaskSR: Masked language
model for full-band speech restoration,” in Interspeech, 2024.

[12] J. Zhang, J. Yang, Z. Fang, Y. Wang, Z. Zhang, Z. Wang
et al., “AnyEnhance: A unified generative model with prompt-
guidance and self-critic for voice enhancement,” IEEE Trans.
Audio, Speech, Lang. Proc., vol. 33, pp. 3085–3098, 2025.

[13] T. H. Pham, T. D. Nguyen, P. T. Tran, J. S. Chung, and
D. D. Nguyen, “MAGE: A coarse-to-fine speech enhancer with
masked generative model,” in IEEE Int. Conf. Acoust., Speech,
Sig. Proc. (ICASSP), 2026.

[14] T. Li, Y. Tian, H. Li, M. Deng, and K. He, “Autoregressive
image generation without vector quantization,” in Adv. Neural
Inform. Proc. Syst. (NeurIPS), 2024.

[15] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu,
W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer:
Convolution-augmented Transformer for speech recognition,”
in Interspeech, 2020.

[16] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar,
“High-fidelity audio compression with improved RVQGAN,”
Adv. Neural Inform. Proc. Syst. (NeurIPS), vol. 36, 2023.

[17] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Free-
man, “MaskGIT: Masked generative image transformer,” in
IEEE/CVF Conf. Computer Vision Pattern Recog. (CVPR),
2022.

```
[18] M. Ranzato, S. Chopra, M. Auli, and W. Zaremba, “Sequence
level training with recurrent neural networks,” in Int. Conf.
Learn. Rep. (ICLR), 2016.
[19] Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing ideal
time–frequency magnitude masking for speech separation,”
IEEE Trans. Audio, Speech, Lang. Proc., vol. 27, no. 8, pp.
1256–1266, 2019.
[20] J. Chen, Q. Mao, and D. Liu, “Dual-path Transformer net-
work: Direct context-aware modeling for end-to-end monaural
speech separation,” in Interspeech, 2020.
[21] J. Cosentino, M. Pariente, S. Cornell, A. Deleforge, and E. Vin-
cent, “LibriMix: An open-source dataset for generalizable
speech separation,” preprint arXiv:2005.11262, 2020.
[22] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Lib-
riSpeech: An ASR corpus based on public domain au-
dio books,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc.
(ICASSP), 2015.
[23] G. Wichern, J. Antognini, M. Flynn, L. R. Zhu, E. McQuinn,
D. Crow et al., “WHAM!: Extending speech separation to
noisy environments,” in Interspeech, 2019.
[24] J. Thiemann, N. Ito, and E. Vincent, “The Diverse Environ-
ments Multi-channel Acoustic Noise Database (DEMAND):
A database of multichannel environmental noise recordings,”
Proc. Meetings on Acoustics, vol. 19, no. 1, p. 035081, 2013.
[25] I. Loshchilov and F. Hutter, “Decoupled weight decay regular-
ization,” in Int. Conf. Learn. Rep. (ICLR), 2018.
[26] C. Reddy, V. Gopal, and R. Cutler, “DNSMOS P.835: A non-
intrusive perceptual objective speech quality metric to evaluate
noise suppressors,” in IEEE Int. Conf. Acoust., Speech, Sig.
Proc. (ICASSP), 2022.
[27] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec
2.0: A framework for self-supervised learning of speech rep-
resentations,” in Adv. Neural Inform. Proc. Syst. (NeurIPS),
vol. 33, 2020.
```

