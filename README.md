# **Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events**
*Aditya Chinchure🔸,  Sahithya Ravi🔸,  Raymond Ng,  Vered Shwartz,  Boyang Li,  Leonid Sigal*
(🔸 indicates equal contribution)

[[paper](https://arxiv.org/abs/2412.05725)]
[[project](https://blackswan.cs.ubc.ca/)]
[[dataset](https://huggingface.co/collections/UBC-ViL/black-swan-abductive-and-defeasible-reasoning-67de1a4ab7ddc22edf0b0542)]
[[bibtex](#citing-black-swan)]


### Code

The code is organized as follows:

* `oops` contains setup code for extracting V_pre, V_main, and V_post from videos.
* `templates` contains data collection and evaluation templates for Cloud Research (or MTurk) participants.
* `inferece` contains code for running inference on the Black Swan dataset on multiple models
* `metric` contains code for computing metrics on the Black Swan dataset

You can set up the python environment by running the following command:
```bash
pip install -r requirements.txt
```

### Citing Black Swan
```tex
@misc{chinchure2024blackswanabductivedefeasible,
      title={Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events}, 
      author={Aditya Chinchure and Sahithya Ravi and Raymond Ng and Vered Shwartz and Boyang Li and Leonid Sigal},
      year={2024},
      eprint={2412.05725},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05725}, 
}
```
