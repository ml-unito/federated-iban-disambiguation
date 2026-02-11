# flower-mlp: A Flower / PyTorch app

## Create virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Logger informations

In the config/flower_exp_bertmlp.yaml file, you can find the logger section that contains

```yaml
enabled: false
```

if set to true, WandB will be used as logger, and  

```yaml
project: #YOUR WANDB PROJECT NAME#
entity: #YOUR WANDB ENTITY NAME#
```
should be setted.

## seed information

In the config/flower_exp_bertmlp.yaml file, seed should be changed accordingly to the ones in the dataset folders names.

### Run with 4 clients

In the `flower-mlp` directory, you need to open six terminals: 1 for server process, 4 for clients process, 1 for Flower App. 

In the **server terminal**, start the SuperLink process in insecure mode:
```bash
uv run flower-superlink --insecure
```

After that, you launch four SuperNodes (**clients**) and connect them to the SuperLink (server):
- in the first clients terminal, run this command:
  ```bash
  uv run flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config "partition-id=0 num-partitions=4"
  ```
- in the second clients terminal, run this command:
  ```bash
  uv run flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config "partition-id=1 num-partitions=4"
  ```
- in the third clients terminal, run this command:
  ```bash
  uv run flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config "partition-id=2 num-partitions=4"
  ```
- in the fourth clients terminal, run this command:
  ```bash
  uv run flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9097 \
    --node-config "partition-id=3 num-partitions=4"
  ```

As a final step, in the last terminal run the **Flower App** and follow the ServerApp logs to track the execution of the run:
```bash
uv run flwr run . local-deployment --stream
```

At the end of federation the global model will be saved in the _.out/_ directory.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)


