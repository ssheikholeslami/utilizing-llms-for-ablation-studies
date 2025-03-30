# Utilizing Large Language Models for Ablation Studies in Machine Learning and Deep Learning
Code and results from our paper, "[Utilizing Large Language Models for Ablation Studies in Machine Learning and Deep Learning](https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-360719)" (AblationMage), published in [EuroMLSys 2025](https://euromlsys.eu/).

## Setup Instructions

We have developed and tested AblationMage with Python 3.12.7. If you want to reproduce the experiments included in the paper, we recommend that you create a Python virtual environment with Python 3.12, e.g., with `python3.12 -m venv env-name`, activate the environment using `source env-name/bin/activate` and then clone the repo and use `pip install -r requirements.txt` to install the specific versions of the required libraries, including the libraries required to reproduce the experiment. However, if you want to use AblationMage for your own ablation studies, the most important requirements are the libraries related to the LLM APIs that you plan to use (e.g., OpenAI or Anthropic libraries).

Note that for our experiments we used Anthropic's `claude-3-5-sonnet-20241022` snapshot.
## Usage

After annotating your code with either *explicit* or *hint* annotations, you can pass the source code(s) and any other supporting documents to AblationMage for a *first call*:

`python ablationmage.py first-call HuggingFaceH4/zephyr-7b-beta -a anthropic -m claude-3-5-sonnet-20241022 -d your_base_code.py`

AblationMage's output from each call will be saved in a new file under `ablationmage_outputs`. This output includes instructions, as well as the code that you can copy into a new Python script and name it to, e.g., `ablation_code.py` and execute. Upon execution of the output code, copy the stack trace into a new file, e.g., `error_output.txt`, and do a *follow-up call* like this:

`python ablationmage.py followup-call HuggingFaceH4/zephyr-7b-beta -a anthropic -m claude-3-5-sonnet-20241022 -o error_output.txt -d ablation_code.py`

Note that for each follow-up call you should also provide the code that led to the error, i.e., the code output of AblationMage from the preceding step.


