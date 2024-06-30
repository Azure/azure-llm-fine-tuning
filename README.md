# LLM Fine-tuning on Azure

This hands-on walks you through fine-tuning an open source LLM on Azure and serving the fine-tuned model on Azure. It is intended for Data Scientists and ML engineers who have experience with fine-tuning but are unfamiliar with Azure ML and Mlflow.
This hands-on is suitable for the following purposes:

1. Half-day workshop or 1-day workshop
2. Hackathon starter code
3. Reference guide for SLM fine-tuning&serving PoC/Prototype

## Use cases

- ### [Phi-3 Fine-tuning](phi3)

- ### [Florence2 Fine-tuning on DoCVQA](florence2-VQA)

## Requirements

Before starting, you have met the following requirements:

- [Azure ML getting started](https://github.com/Azure/azureml-examples/tree/main/tutorials): Connect to Azure ML workspace and get your <WORKSPACE_NAME>, <RESOURCE_GROUP> and <SUBSCRIPTION_ID>.
- [Azure ML CLI v2](https://learn.microsoft.com/en-us/azure/machine-learning/concept-v2?view=azureml-api-2#azure-machine-learning-cli-v2)
- ***[Compute instance - for code development]*** A low-end instance without GPU is recommended: `Standard_DS11_v2` (2 cores, 14GB RAM, 28GB storage, No GPUs).
- ***[Compute cluster - for LLM training]*** A single NVIDIA A100 GPU node (`Standard_NC24ads_A100_v4`) and a single NVIDIA V100 GPU node (`Standard_NC6s_v3`) is recommended. If you do not have a dedicated quota or are on a tight budget, choose Low-priority VM.

## How to get started 
1. Create your compute instance. For code development, we recommend `Standard_DS11_v2` (2 cores, 14GB RAM, 28GB storage, No GPUs).
2. Open the terminal of the CI and run: 
    ```shell
    git clone https://github.com/Azure/azure-llm-fine-tuning.git
    conda activate azureml_py310_sdkv2
    pip install -r requirements.txt
    ```
3. Choose the model to use for your desired use case.
    - [Phi-3](phi3)
        - [Option 1. MLflow] Run `1_training_mlflow.ipynb` and `2_serving.ipynb`, respectively.
        - [Option 2. Custom] Run `1_training_custom.ipynb` and `2_serving.ipynb`, respectively.
        - *(Optional)* If you are interested in LLM dataset preprocessing, see the hands-ons in `phi3/dataset-preparation` folder.
    - [Florence2-VQA](florence2-VQA)
        - Run `1_training_mlflow.ipynb` and `2_serving.ipynb`, respectively.
    - Don't forget to edit the `config.yml`.

## References

- [Azure Machine Learning examples](https://github.com/Azure/azureml-examples)

### Phi-3
- [Finetune Small Language Model (SLM) Phi-3 using Azure ML](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/finetune-small-language-model-slm-phi-3-using-azure-machine/ba-p/4130399)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct): This is Microsoft's official Phi-3-mini-4k-instruct model.
- [daekeun-ml/Phi-3-medium-4k-instruct-ko-poc-v0.1](https://huggingface.co/daekeun-ml/Phi-3-medium-4k-instruct-ko-poc-v0.1)

### Florence2
- [Hugging Face Blog - Finetune Florence2 on DoCVQA](https://huggingface.co/blog/finetune-florence2)

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