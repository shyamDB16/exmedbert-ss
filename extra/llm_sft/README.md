# Qwen SFT and serving example

This optional notebook demonstrates generic supervised fine-tuning and serving
for `Qwen/Qwen3.5-0.8B-Base` on Databricks serverless GPU compute:

1. Load the base model and tokenizer from Hugging Face.
2. Select 100 examples from `banghua/DL-SFT-Dataset`.
3. Run one epoch of full-parameter SFT with TRL on one H100.
4. Save and smoke-test the model with a local vLLM server.
5. Log and register the model in Unity Catalog.
6. Create and query a Custom Foundation Model serving endpoint.

It is not part of ExMed-BERT, does not use the ExMed-BERT Synthea snapshots,
and does not demonstrate reward modeling, RLHF, PPO, or GRPO.

## Before running

- Use a Databricks serverless GPU environment with the notebook's required
  Transformers, TRL, vLLM, MLflow, and Databricks SDK versions.
- Set `UC_MODEL_NAME` to a writable `catalog.schema.model` name.
- Choose a unique `ENDPOINT_NAME`.
- Review endpoint workload sizing and permissions.
- Remember that the deployment cell creates a potentially billable endpoint;
  delete it when it is no longer needed.

Training uses `report_to=[]`, so the notebook does not currently log SFT loss
and hyperparameters to MLflow. Its prompt checks are illustrative rather than a
quality evaluation. Add a held-out dataset and explicit evaluation before using
the result beyond a platform demo.

The committed notebook has outputs and execution metadata cleared. Run it from
the beginning in your own workspace:

- [sgc-qwen3_sft_train_serve.ipynb](sgc-qwen3_sft_train_serve.ipynb)
