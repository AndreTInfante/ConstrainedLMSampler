What this is: 

A constrained sampler for a language model that forces it to reply in the form of a valid, existing wikipedia URL.

Dependencies:

This uses Falcon-7B-instruct, via HuggingFace. Must have Pytorch 2.0 and a working CUDA setup   

https://huggingface.co/tiiuae/falcon-7b-instruct
https://huggingface.co/blog/falcon

I'm running via Anaconda with the following packages installed (relevant ones grabbed from  conda list). I may have missed a couple, but following the errors should get you there.

tokenizers                0.13.3                   pypi_0    pypi
torch                     2.0.1                    pypi_0    pypi
torchaudio                2.0.2                    pypi_0    pypi
torchvision               0.15.2                   pypi_0    pypi
tqdm                      4.65.0                   pypi_0    pypi
transformers              4.30.2                   pypi_0    pypi
accelerate                0.20.3                   pypi_0    pypi
huggingface-hub           0.15.1                   pypi_0    pypi
pip                       23.1.2           py38haa95532_0
pytorch                   2.0.1           py3.8_cuda11.7_cudnn8_0    pytorch
pytorch-cuda              11.7                 h16d0643_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
xformers                  0.0.20                   pypi_0    pypi


Usage: 

ConstructURLTree.py builds the token tree from wiki_urls.txt. However, the project comes with a precomputer pickle of the tree in bigtree.p, so this shouldn't be necessary.

URLGenerator.py runs through a list of canned questions and does constrained sampling of the Falcon model (forced to choose a path through the token tree of valid URLs). It then defaults to a Q&A mode. 

URLGenerator.py takes a while to load, and really taxes system RAM due to the tree not being particularly efficient, so be patient.



