# PML&DL Assignment III
## Artemii Bykov, Maxim Salo, BS17-DS01
###### tags: `PML&DL`, `ML`, `Supervised learning`, `NN`, `FairSeq`, `Machine Translation`

### Self-made Transformer Model
In the beginning, we tried to implement the Transformer model ourselves. We decided to use PyTorch for this purpose. To start with, we used the PyTorch transformer tutorial.

#### Architecture
The overall architecture consists of the YouTokenToMe tokenizer that implements fastBPE, positional encoding and transformer module. Moreover, we used cross-entropy loss, Adam optimizer and StepLR learning rate scheduler.

#### Training
For training purposes, we used the proposed Yandex dataset with 1 million samples. However, training the model on the whole dataset takes a lot of time, even using Colab with GPU (we have no GPUs to train on our machines 😭). That is why we tried only up to 10 epochs of training. Also, we tried to train the model on 10k samples of the dataset with more than 20 epochs.

#### Problems and solutions
As a result, we have a model that converges in train loss (and sometimes in validation loss), but it cannot produce any meaningful translations at all.

We think that the problem can be with the learning rate and the number of epochs with the dataset size. 

We tried to tune the learning rate and check it with the same set of other parameters; however, it gave no tremendous results (except that the initial learning rate must be at least 0.1 or lower).

If we talk about the number of epochs and the dataset size, we have no opportunity to check the higher number of epochs as it just takes a vast amount of time.

### FairSeq Model
In the end, we decided to switch to the pre-trained model and searched for the WMT competition participants. We've found the Facebook AI research lab solution to the competition: the FairSeq model that led the WMT19 conference.

This model underneath is the Transformer model. Facebook AI research group did not make any serious changes in the network's architecture for the WMT19 competition. The main concern was cleaning the provided dataset because the incorrect data lowered the score. That is why they applied length filtering, keeping only sentences that were not bigger than 250 tokens. Moreover, they applied language identification models to filter out sentences with mixed languages. The total dataset was reduced by 30%.

Overall changes for WMT19 included the following techniques:
* Dataset reduction
* Increasing network capacity by increasing embed dimension, number of heads, and number of layers
* Using a larger FFN size (8192) gives a reasonable improvement in performance while maintaining a manageable network size 
* For fine-tuning, the authors used a combination of News-Commentary, newstest2013, newstest2015, and newstest2017.

We implemented the script that uses a `torch.hub` package. We have downloaded the checkpoint for model inference and used it for both CodaLab phases.

The script takes as input paths to: 
* model checkpoints
* input file with Russian sentences
* output file with translated sentences in English

### Results
* Development phase - **BLEU score = 51.81**
* Evaluation phase - **BLEU score = 50.96**

### References
1. [PyTorch transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
2. [PyTorch translation tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
3. [FairSeq](https://arxiv.org/pdf/1904.01038.pdf)
4. [Facebook FAIR’s WMT19 News Translation Task Submission](https://arxiv.org/pdf/1907.06616.pdf)
5. [Facebook AI blog post with WMT19 result](https://ai.facebook.com/blog/facebook-leads-wmt-translation-competition/)
6. [FairSeq WMT19 GitHub](https://github.com/pytorch/fairseq/tree/master/examples/wmt19)
7. [WMT19 RU->EN Model Checkpoint](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz)
8. [Our GitHub](https://github.com/BullDog57Rus/PMLDL-AssignmentIII)
