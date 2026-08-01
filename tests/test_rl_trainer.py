import json

import jax.numpy as jnp
import numpy as np
import pytest

from taktiny import nn
from taktiny.trainer import (
    DatasetConfig,
    PolicyRuntime,
    RLBaseTrainer,
    Rollout,
    Trainer,
    TrainingConfig,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(jnp.asarray(0.0))


class FakePolicyRuntime:
    def __init__(self, model, *, policy_version=0):
        self.model = model
        self.policy_version = policy_version
        self.sync_calls = []
        self.generate_calls = []
        self.fail_sync = False
        self.returned_version = None

    def sync(self, *, policy_version=None, **kwargs):
        self.sync_calls.append({
            'policy_version': policy_version,
            'weight': float(self.model.weight.value),
            'kwargs': dict(kwargs),
        })
        if self.fail_sync:
            raise RuntimeError('sync failed')
        if self.returned_version is not None:
            policy_version = self.returned_version
        self.policy_version = policy_version
        return policy_version

    def generate(self, prompt_ids):
        prompt_ids = np.asarray(prompt_ids)
        self.generate_calls.append({
            'prompt_ids': prompt_ids.copy(),
            'policy_version': self.policy_version,
            'weight': float(self.model.weight.value),
        })
        return prompt_ids + 1


class TinyOnlineTrainer(RLBaseTrainer):
    def generate_rollouts(self, prompt_batch):
        return self.runtime.generate(prompt_batch['prompt_ids'])

    def build_rollouts(self, prompt_batch, generated):
        return (
            self.create_rollout(
                prompt_ids=prompt_batch['prompt_ids'],
                output_ids=generated,
                metadata={
                    'x': prompt_batch['x'],
                    'y': prompt_batch['y'],
                },
            ),
        )

    def compute_rewards(self, rollouts):
        return tuple(
            float(np.asarray(rollout.output_ids).mean())
            for rollout in rollouts
        )

    def prepare_training_batch(self, rollouts, rewards):
        assert len(rollouts) == len(rewards) == 1
        return {
            'x': rollouts[0].metadata['x'],
            'y': rollouts[0].metadata['y'],
            'reward': np.asarray(rewards, dtype=np.float32),
        }


def squared_error(model, batch):
    prediction = model.weight.value * batch['x']
    return jnp.mean(jnp.square(prediction - batch['y']))


def non_finite_loss(model, batch):
    del model, batch
    return jnp.asarray(jnp.nan)


def prompt_batch(index=0):
    return {
        'prompt_ids': np.asarray([index + 1], dtype=np.int32),
        'x': np.asarray([1.0], dtype=np.float32),
        'y': np.asarray([index + 2.0], dtype=np.float32),
    }


def make_trainer(
    model,
    *,
    loss_fn=squared_error,
    runtime=None,
    max_steps=1,
    batch_count=1,
    gradient_accumulation_steps=1,
    prefetch_size=2,
):
    trainer_type = TinyOnlineTrainer if (
        runtime is not None
        or isinstance(model, FakePolicyRuntime)
    ) else RLBaseTrainer
    return trainer_type(
        model,
        TrainingConfig(
            max_steps=max_steps,
            learning_rate=0.1,
            log_interval=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        DatasetConfig(
            [prompt_batch(index) for index in range(batch_count)],
            prefetch_size=prefetch_size,
        ),
        loss_fn=loss_fn,
        runtime=runtime,
    )


def test_policy_runtime_protocol_is_structural():
    runtime = FakePolicyRuntime(TinyPolicy())

    assert isinstance(runtime, PolicyRuntime)


def test_rl_base_uses_the_trainer_train_api():
    assert RLBaseTrainer.train is Trainer.train


def test_rl_trainer_unwraps_runtime_model():
    policy = TinyPolicy()
    runtime = FakePolicyRuntime(policy, policy_version=3)
    trainer = make_trainer(runtime)

    assert trainer.model is policy
    assert trainer.runtime is runtime
    assert trainer.has_runtime
    assert trainer.policy_version == 3
    assert trainer.policy_dirty


def test_rl_trainer_accepts_runtime_separately():
    policy = TinyPolicy()
    runtime = FakePolicyRuntime(policy)
    trainer = make_trainer(policy, runtime=runtime)

    assert trainer.model is policy
    assert trainer.runtime is runtime


def test_rl_trainer_rejects_runtime_for_another_model():
    policy = TinyPolicy()
    runtime = FakePolicyRuntime(TinyPolicy())

    with pytest.raises(
        ValueError,
        match='runtime.model must be the same object',
    ):
        make_trainer(policy, runtime=runtime)


def test_train_runs_rollout_reward_update_and_sync_lifecycle():
    runtime = FakePolicyRuntime(TinyPolicy())
    trainer = make_trainer(
        runtime,
        max_steps=2,
        batch_count=2,
        prefetch_size=8,
    )

    result = trainer.train()

    assert result is None
    assert trainer.global_step == 2
    assert trainer.policy_version == 3
    assert not trainer.policy_dirty
    assert [
        call['policy_version']
        for call in runtime.generate_calls
    ] == [1, 2]
    assert [
        call['policy_version']
        for call in runtime.sync_calls
    ] == [1, 2, 3]
    assert runtime.sync_calls[0]['weight'] == 0.0
    assert runtime.sync_calls[1]['weight'] != 0.0
    assert (
        runtime.sync_calls[-1]['weight']
        == float(trainer.model.weight.value)
    )


def test_gradient_accumulation_reuses_policy_for_microbatches():
    runtime = FakePolicyRuntime(TinyPolicy())
    trainer = make_trainer(
        runtime,
        max_steps=2,
        batch_count=4,
        gradient_accumulation_steps=2,
        prefetch_size=8,
    )

    trainer.train()

    assert [
        call['policy_version']
        for call in runtime.generate_calls
    ] == [1, 1, 2, 2]
    assert [
        call['policy_version']
        for call in runtime.sync_calls
    ] == [1, 2, 3]


def test_skipped_optimizer_update_does_not_publish_new_policy():
    runtime = FakePolicyRuntime(TinyPolicy())
    trainer = make_trainer(
        runtime,
        loss_fn=non_finite_loss,
    )

    trainer.train()

    assert trainer.last_update_skipped
    assert not trainer.policy_dirty
    assert trainer.policy_version == 1
    assert len(runtime.sync_calls) == 1
    assert len(runtime.generate_calls) == 1


def test_failed_initial_sync_stops_train_before_generation():
    runtime = FakePolicyRuntime(TinyPolicy(), policy_version=4)
    runtime.fail_sync = True
    trainer = make_trainer(runtime)

    with pytest.raises(RuntimeError, match='sync failed'):
        trainer.train()

    assert trainer.policy_version == 4
    assert trainer.policy_dirty
    assert runtime.generate_calls == []


def test_policy_sync_rejects_unexpected_runtime_version():
    runtime = FakePolicyRuntime(TinyPolicy())
    runtime.returned_version = 7
    trainer = make_trainer(runtime)

    with pytest.raises(
        RuntimeError,
        match='expected 1, got 7',
    ):
        trainer.train()

    assert trainer.policy_version == 0
    assert trainer.policy_dirty
    assert runtime.generate_calls == []


def test_rollout_creation_and_version_validation():
    runtime = FakePolicyRuntime(TinyPolicy())
    trainer = make_trainer(runtime)
    trainer.sync_policy()
    rollout = trainer.create_rollout(
        prompt_ids=[1, 2],
        output_ids=[3, 4],
        logprobs=[-0.2, -0.1],
        finish_reason='stop',
        metadata={'reward': 1.0},
    )

    assert rollout.policy_version == 1
    assert trainer.validate_rollouts(rollout) == (rollout,)

    stale = Rollout(
        prompt_ids=[1],
        output_ids=[2],
        policy_version=0,
    )
    with pytest.raises(ValueError, match='Stale rollout'):
        trainer.validate_rollouts([stale])


def test_rl_base_without_runtime_uses_normal_trainer_loop():
    trainer = make_trainer(TinyPolicy())

    trainer.train()

    assert trainer.global_step == 1
    assert not trainer.has_runtime
    with pytest.raises(RuntimeError, match='policy runtime is required'):
        trainer.sync_policy()


def test_rl_trainer_state_restores_version_and_dirties_new_runtime(
    tmp_path,
):
    runtime = FakePolicyRuntime(TinyPolicy())
    trainer = make_trainer(runtime)
    trainer.sync_policy()
    state = trainer._trainer_state(
        step=4,
        epoch=1,
        step_in_epoch=2,
    )
    checkpoint = tmp_path / 'checkpoint-4'
    checkpoint.mkdir()
    with (checkpoint / 'trainer_state.json').open('w') as state_file:
        json.dump(state, state_file)

    restored_runtime = FakePolicyRuntime(TinyPolicy())
    restored = make_trainer(restored_runtime)
    restored._load_resume_state(checkpoint)

    assert restored.policy_version == 1
    assert restored.policy_dirty


@pytest.mark.parametrize(
    ('kwargs', 'error', 'message'),
    [
        (
            {'policy_version': -1},
            ValueError,
            'policy_version must be a non-negative integer',
        ),
        (
            {'policy_version': 0, 'finish_reason': 1},
            TypeError,
            'finish_reason must be a string or None',
        ),
        (
            {'policy_version': 0, 'metadata': []},
            TypeError,
            'metadata must be a mapping',
        ),
    ],
)
def test_rollout_validates_metadata(kwargs, error, message):
    with pytest.raises(error, match=message):
        Rollout(prompt_ids=[1], output_ids=[2], **kwargs)
