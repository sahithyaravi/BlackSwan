# Black Swan Suite 

#### **Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events**
*Aditya Chinchure🔸,  Sahithya Ravi🔸,  Raymond Ng,  Vered Shwartz,  Boyang Li,  Leonid Sigal*
(🔸 indicates equal contribution)

### News
- 📚 Paper available on [arXiv](https://arxiv.org/abs/2412.05725)

### Dataset
To be released.

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