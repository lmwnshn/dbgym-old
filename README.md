# README

> The goal of a database gym is to simplify ML model training and evaluation
> to accelerate autonomous DBMS research.

The vast majority of the code in your typical autonomous DBMS research paper tends
to be completely unrelated to the ML (TODO: add graphic?).
Instead, you spend all your time futzing around with instrumenting X, passing data
and training options through to Y, trying to understand ad-hoc training data formats,
and so on.

The database gym ("dbgym", aka "gym") tries to solve that. dbgym aims to standardize:

- Training data collection
- Training data format
- Training models for autonomous databases
- Using trained models (inference) to apply actions

For more details, see (TODO: CIDR paper if accepted).

The goal is to make using autonomous database models as easy as:

```python
# Specify the historical workloads and state.
gym_spec = GymSpec(
    historical_workloads=workloads,
    historical_state=state,
)
# Create the gym environment.
env = gym.make(
    "dbgym/DbGym-v0",
    gym_spec=gym_spec,
    # Additional arguments.
)
# Collect initial observations.
observations, info = env.reset(seed=15721)
# Create or load a model.
model = load_model(model_type)
model.initialize(observations, info)
# Pick actions based on the model.
for _ in range(1000):
    action = model.pick_action(env.action_space, observations)
    observations, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observations, info = env.reset(seed = 15721)
```

## Adding a new model

Adding a new model has two steps.

1. Define the format of your model's training data.
2. Define the model itself.

You will probably want to start by familiarizing yourself with `main.py` and how a normal OpenAI gym works.

### Defining training data format

You will want to familiarize yourself with https://www.gymlibrary.dev/api/spaces/

If you can reuse an existing observation space, do so.
Otherwise, see `/dbgym/spaces/observations/qppnet/features.py` for an example.

We choose to comply strictly with OpenAI's space format because this unlocks the possibility of
integrating with other RL libraries in the future, that may be able to directly operate on
well-defined observations.

### Defining the model

You will want to add your model under `/models/`, see `/models/qppnet.py` as an example.

If your model needs multiple files, please create a submodule under `/models/my_new_model/`.


# TODO

- I am thinking about a way to abstract out the experiment sweeps from `main.py`.
- I am thinking about ways to improve the workload runner, both from a code cleanliness and runtime standpoint.