# SLM/LLM Fine-tuning on Azure

This hands-on walks you through fine-tuning an open source SLM/LLM on Azure and serving the fine-tuned model on Azure. It is intended for Data Scientists and ML engineers who have experience with fine-tuning but are unfamiliar with Azure ML and Mlflow.
This hands-on is suitable for the following purposes:

1. Half-day workshop or 1-day workshop
2. Hackathon starter code
3. Reference guide for SLM fine-tuning&serving PoC/Prototype

## Use cases

- ### [Phi-3/Phi-3.5 Fine-tuning](phi3)

- ### [Florence-2 Fine-tuning on DoCVQA](florence2-VQA)

- ### [Azure OpenAI Fine-tuning](aoai)

## Requirements
Before starting, you should meet the following requirements:

- [Access to Azure OpenAI Service](https://go.microsoft.com/fwlink/?linkid=2222006)
- [Azure ML getting started](https://github.com/Azure/azureml-examples/tree/main/tutorials): Connect to [Azure ML] workspace and get your <WORKSPACE_NAME>, <RESOURCE_GROUP> and <SUBSCRIPTION_ID>.
- [Azure AI Studio getting started](https://aka.ms/azureaistudio): Create a project

- ***[Compute instance - for code development]*** A low-end instance without GPU is recommended: **[Standard_E2as_v4] (AMD 2 cores, 16GB RAM, 32GB storage) or **[Standard_DS11_v2]** (Intel 2 cores, 14GB RAM, 28GB storage, No GPUs)  
- ***[Compute cluster - for SLM/LLM fine-tuning]*** A single NVIDIA A100 GPU node (**[Standard_NC24ads_A100_v4]**) is recommended. If you do not have a dedicated quota or are on a tight budget, choose **[Low-priority VM]**.
- ***[Compute cluster - for SLM/LLM deployment]*** Two NVIDIA V100 GPU nodes (**[Standard_NC6s_v3]**) or two NVIDIA A100 GPU nodes (**[Standard_NC24ads_A100_v4]**) is recommended. 

**Note**
For managed onlie endpoints, [Azure ML reserves 20% of the quota for the deployment].[^1] If you request a given number of instances for those VM SKUs in a deployment, you must have a quota for `ceil(1.2 × number of instances requested for deployment) × number of cores for the VM SKU` available to avoid getting an error. For example, if you request 1 instances of a `Standard_NC6s_v3` VM (that comes with six cores) in a deployment, you should have a quota for 12 cores (ceil(1.2 × 1 instances) = 2, 2 × 6 cores) available.  


## How to get started

1. Create your compute instance. For code development, we recommend `Standard_DS11_v2` (2 cores, 14GB RAM, 28GB storage, No GPUs).
2. Open the terminal of the CI and run:
    ```shell
    git clone https://github.com/Azure/slm-innovator-lab.git
    conda activate azureml_py310_sdkv2
    pip install -r requirements.txt
    ```
3. Choose the model to use for your desired use case.
    - [Phi-3, Phi-3.5](phi3)
        - [Option 1. MLflow] Run [`1_training_mlflow.ipynb`](phi3/1_training_mlflow.ipynb) and [`2_serving.ipynb`](phi3/2_serving.ipynb), respectively.
        - [Option 2. Custom] Run [`1_training_custom.ipynb`](phi3/1_training_custom.ipynb) and [`2_serving.ipynb`](phi3/2_serving.ipynb), respectively.
        - _(Optional)_ If you are interested in LLM dataset preprocessing, see the hands-ons in `phi3/dataset-preparation` folder.
    - [Florence2-VQA](florence2-VQA)
        - Run [`1_training_mlflow.ipynb`](florence2-VQA/1_training_mlflow.ipynb) and [`2_serving.ipynb`](florence2-VQA/2_serving.ipynb), respectively.
    - Don't forget to edit the `config.yml`.

## References

- [Azure Machine Learning examples](https://github.com/Azure/azureml-examples)

### Phi-3/Phi-3.5
- [Finetune Small Language Model (SLM) Phi-3 using Azure ML](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/finetune-small-language-model-slm-phi-3-using-azure-machine/ba-p/4130399)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct): This is Microsoft's official Phi-3-mini-4k-instruct model.
- [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct): This is Microsoft's official Phi-3-mini-128k-instruct model.
- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct): This is Microsoft's official Phi-3.5-mini-instruct model.
- [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct): This is Microsoft's official Phi-3.5-MoE-instruct model.
- [Korean language proficiency evaluation for LLM/SLM models using KMMLU, CLIcK, and HAE-RAE dataset](https://github.com/daekeun-ml/evaluate-llm-on-korean-dataset)
- [daekeun-ml/Phi-3-medium-4k-instruct-ko-poc-v0.1](https://huggingface.co/daekeun-ml/Phi-3-medium-4k-instruct-ko-poc-v0.1)

### Florence-2
- [Fine-tuning Florence-2 for VQA (Visual Question Answering) using the Azure ML Python SDK and MLflow](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/fine-tuning-florence-2-for-vqa-visual-question-answering-using/ba-p/4181123)
- [Hugging Face Blog - Finetune Florence-2 on DoCVQA](https://huggingface.co/blog/finetune-florence2)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## License Summary

This sample code is provided under the MIT-0 license. See the LICENSE file.

[^1]: This extra quota is reserved for system-initiated operations such as OS upgrades and VM recovery, and it won't incur cost unless such operations run.